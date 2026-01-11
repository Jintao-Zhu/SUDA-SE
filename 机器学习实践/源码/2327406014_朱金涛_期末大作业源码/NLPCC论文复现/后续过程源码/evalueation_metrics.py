import json
import re
import ast
import numpy as np
import jieba 
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm

# ==================== 初始化 ====================
try:
    # 尝试加载 wordnet (如果没网会跳过，不影响其他指标)
    nltk.data.find('corpora/wordnet')
except LookupError:
    pass

# ==================== 核心算法函数 ====================
def my_rouge_l(ref_tokens, cand_tokens):
    """手写 ROUGE-L (LCS算法)，解决第三方库中文兼容性问题"""
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
    """强力清洗函数 (用于处理模型生成的 Raw String)"""
    if not isinstance(text_str, str): return {}
    start = text_str.find("{")
    end = text_str.rfind("}") + 1
    if start == -1 or end == -1 or start >= end: return None
    clean_str = text_str[start:end]
    pattern = r'("[^"]*")|#.*'
    clean_str = re.sub(pattern, lambda m: m.group(1) if m.group(1) else "", clean_str)
    python_style_str = (clean_str.replace("true", "True").replace("false", "False").replace("null", "None"))
    try: return json.loads(clean_str)
    except: pass
    try: return ast.literal_eval(python_style_str)
    except: pass
    return None

def tokenize_zh(text):
    return list(jieba.cut(text))

# ==================== 主逻辑 ====================
RESULTS_FILE = "benchmark_results_test.json"

print(f"正在读取文件: {RESULTS_FILE}")
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

y_true_cls, y_pred_cls = [], []
scores_bleu, scores_meteor, scores_rouge = [], [], []
valid_count = 0

print(f"正在评估 {len(data)} 条测试集样本...")

for idx, item in enumerate(tqdm(data)):
    # =======================================================
    # 🛠️ 关键修复：从分散的字段中重建 Ground Truth
    # =======================================================
    gt_labels = item.get('ground_truth_label')
    gt_edit = item.get('ground_truth_edit')
    
    # 获取模型预测 (优先使用已解析好的 model_parsed)
    gen_json = item.get('model_parsed')
    if not gen_json:
        # 如果之前解析失败，再试一次清洗解析
        gen_json = clean_and_parse(item.get('model_output_raw', ''))

    # 只要 Ground Truth 有标签，我们就进行评估
    if gt_labels is None: 
        continue
        
    valid_count += 1

    try:
        # --- Task 2: Classification ---
        # 容错：如果模型没生成 bias_labels，或者格式不对，默认为 [0,0,0]
        if gen_json and isinstance(gen_json.get('bias_labels'), list) and len(gen_json['bias_labels']) >= 3:
            pred_l = gen_json['bias_labels'][:3]
        else:
            pred_l = [0, 0, 0] # 惩罚：格式错误算全错
            
        y_true_cls.append(gt_labels[:3])
        y_pred_cls.append(pred_l)

        # --- Task 3: Mitigation ---
        # Ground Truth 改写句
        ref_text = str(gt_edit) if gt_edit else ""
        
        # 模型生成改写句
        cand_text = ""
        if gen_json:
            cand_text = str(gen_json.get('edit_sentence', ''))
        
        # 只有当参考答案存在时才计算生成指标
        if ref_text:
            # 1. BLEU
            # 防止空字符串报错
            if not cand_text: cand_text = " " 
            scores_bleu.append(sentence_bleu([list(ref_text)], list(cand_text)))
            
            # 分词 (用于 METEOR 和 ROUGE)
            ref_tok = tokenize_zh(ref_text)
            cand_tok = tokenize_zh(cand_text)
            
            # 2. METEOR
            try:
                scores_meteor.append(meteor_score([ref_tok], cand_tok))
            except:
                pass # NLTK 没网或报错就跳过

            # 3. ROUGE-L
            scores_rouge.append(my_rouge_l(ref_tok, cand_tok))
            
    except Exception as e:
        # print(f"样本 {idx} 计算出错: {e}")
        pass

# ==================== 输出最终成绩单 ====================
print("\n" + "="*50)
print("📊 测试集最终成绩单 (Test Set)")
print("="*50)
print(f"有效评估样本数: {valid_count} / {len(data)}")

if valid_count > 0:
    # Task 2
    y_true_cls = np.array(y_true_cls)
    y_pred_cls = np.array(y_pred_cls)
    f1_list = []
    for i in range(3):
        f1 = f1_score(y_true_cls[:, i], y_pred_cls[:, i], average='macro', zero_division=0)
        f1_list.append(f1)
    print(f"【Task 2 - 分类】 Macro-F1: {np.mean(f1_list):.4f}")
    print(f"   (注: 论文SOTA为 0.509)")

    # Task 3
    meteor_avg = np.mean(scores_meteor) if scores_meteor else 0.0
    print(f"\n【Task 3 - 缓解】")
    print(f"  BLEU:      {np.mean(scores_bleu):.4f}")
    print(f"  METEOR:    {meteor_avg:.4f}")
    print(f"  ROUGE-L:   {np.mean(scores_rouge):.4f}")
    print(f"   (注: 论文BLEU为0.013, ROUGE为0.453)")
    print("="*50)
else:
    print("❌ 依然没有有效数据，请检查 benchmark_results_test.json 内容是否正常。")