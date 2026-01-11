import os
import json
import torch
import re
import ast
import jieba
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from peft import PeftModel
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

# ================= 配置区域 =================
# 1. 模型与适配器路径
BASE_MODEL_PATH = "/public/home/lilingzhi/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "qwen_lora_outputs_ddp"

# 2. 验证集路径
# 如果找不到，可以尝试写绝对路径
VALID_FILE = "NLPCC-2025-Shared-Task-7-main/data/test_gt/classification and mitigation/biased.json"

# 3. 结果保存路径
RESULT_FILE = "benchmark_results_test.json"

# ================= 初始化 NLTK =================
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    pass

# ================= 核心算法函数 =================

def my_rouge_l(ref_tokens, cand_tokens):
    """手写 ROUGE-L (LCS算法)"""
    if not ref_tokens or not cand_tokens: return 0.0
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    prec = lcs_len / n if n > 0 else 0
    rec = lcs_len / m if m > 0 else 0
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def clean_and_parse(text_str):
    """强力清洗并解析模型输出的 JSON"""
    if not isinstance(text_str, str): return {}
    start = text_str.find("{")
    end = text_str.rfind("}") + 1
    if start == -1 or end == -1 or start >= end: return None
    clean_str = text_str[start:end]
    pattern = r'("[^"]*")|#.*'
    clean_str = re.sub(pattern, lambda m: m.group(1) if m.group(1) else "", clean_str)
    # 替换 Python 关键字
    python_style_str = (clean_str
                        .replace("true", "True")
                        .replace("false", "False")
                        .replace("null", "None"))
    try: return json.loads(clean_str)
    except: pass
    try: return ast.literal_eval(python_style_str)
    except: pass
    return None

def tokenize_zh(text):
    return list(jieba.cut(text))

# ================= 主流程 =================

def main():
    # ✅ 修复：使用局部变量 target_file，避免 UnboundLocalError
    target_file = VALID_FILE
    
    # 1. 加载验证集
    print(f"=== 1. 正在加载验证集: {target_file} ===")
    
    if not os.path.exists(target_file):
        # 尝试拼接绝对路径作为备选
        abs_path = os.path.join(os.getcwd(), target_file)
        if os.path.exists(abs_path):
            target_file = abs_path
        else:
            raise FileNotFoundError(f"❌ 找不到文件: {target_file}，请检查路径！")
            
    with open(target_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    print(f"✅ 成功加载 {len(val_data)} 条验证样本")

    # 2. 加载模型
    print("\n=== 2. 正在加载模型与适配器 ===")
    config = Qwen2Config.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("✅ 模型加载完毕")

    # 3. 开始推理与实时评估
    print("\n=== 3. 开始基准测试 (Benchmark) ===")
    
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert in gender bias mitigation. Please analyze the following text for gender bias, provide a step-by-step chain-of-thought analysis, classify the bias type, and provide a rewritten version if bias exists.

### Input:
{}

### Response:
"""
    
    results = []
    y_true_cls, y_pred_cls = [], []
    scores_bleu, scores_rouge, scores_meteor = [], [], [] # 补全 meteor 列表
    parse_fail = 0
    
    for item in tqdm(val_data, desc="推理进度"):
        input_text = item['ori_sentence']
        
        # 构造 Prompt
        prompt = alpaca_prompt.format(input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.3, 
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_text = response.split("### Response:")[-1].strip()
        
        # 解析结果
        gen_json = clean_and_parse(gen_text)
        
        # 保存详细结果
        results.append({
            "original_text": input_text,
            "ground_truth_label": item.get('bias_labels'),
            "ground_truth_edit": item.get('edit_sentence'),
            "model_output_raw": gen_text,
            "model_parsed": gen_json
        })
        
        if not gen_json:
            parse_fail += 1
            continue
            
        # --- 实时计算指标 ---
        # Task 2: Classification
        gt_l = item.get('bias_labels', [0,0,0])
        pred_l = gen_json.get('bias_labels', [0,0,0])
        
        if not isinstance(gt_l, list) or len(gt_l) < 3: gt_l = [0,0,0]
        if not isinstance(pred_l, list) or len(pred_l) < 3: pred_l = [0,0,0]
        
        y_true_cls.append(gt_l[:3])
        y_pred_cls.append(pred_l[:3])
        
        # Task 3: Mitigation
        ref_text = str(item.get('edit_sentence', ''))
        cand_text = str(gen_json.get('edit_sentence', ''))
        
        if ref_text and cand_text:
            scores_bleu.append(sentence_bleu([list(ref_text)], list(cand_text)))
            ref_tok = tokenize_zh(ref_text)
            cand_tok = tokenize_zh(cand_text)
            
            try:
                scores_meteor.append(meteor_score([ref_tok], cand_tok))
            except:
                pass
                
            scores_rouge.append(my_rouge_l(ref_tok, cand_tok))

    # 4. 保存详细数据
    with open(RESULT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 5. 输出最终报告
    print("\n" + "="*50)
    print(f"🏆 验证集最终成绩 (样本数: {len(val_data)})")
    print(f"解析成功率: {len(val_data) - parse_fail} / {len(val_data)}")
    print("="*50)
    
    if len(y_true_cls) > 0:
        y_true_cls = np.array(y_true_cls)
        y_pred_cls = np.array(y_pred_cls)
        f1_list = []
        for i in range(3):
            f1 = f1_score(y_true_cls[:, i], y_pred_cls[:, i], average='macro', zero_division=0)
            f1_list.append(f1)
        print(f"【Task 2 - 偏见分类】 Macro-F1: {np.mean(f1_list):.4f}")
    
    if scores_bleu:
        meteor_avg = np.mean(scores_meteor) if scores_meteor else 0.0
        print(f"【Task 3 - 偏见缓解】")
        print(f"  BLEU:    {np.mean(scores_bleu):.4f}")
        print(f"  METEOR:  {meteor_avg:.4f}")
        print(f"  ROUGE-L: {np.mean(scores_rouge):.4f}")
    
    print("="*50)
    print(f"结果文件已保存至: {RESULT_FILE}")

if __name__ == "__main__":
    main()