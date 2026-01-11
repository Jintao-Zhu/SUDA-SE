import torch
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from peft import PeftModel

# ================= 配置区域 =================
BASE_MODEL_PATH = "/public/home/lilingzhi/Qwen2.5-7B-Instruct"  # 你的底座模型路径
ADAPTER_PATH = "qwen_lora_outputs_ddp"  # 刚才训练保存的LoRA路径
TEST_DATA_FILE = "pianjian_cot_backup.json"  # 这里为了演示用训练数据测，实际应换成验证集
OUTPUT_FILE = "inference_results.json"

# ================= 1. 加载模型 =================
print("=== 正在加载模型与LoRA适配器 ===")

# 加载 Config
config = Qwen2Config.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

# 加载底座模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()  # 切换到评估模式

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)

# ================= 2. 准备测试数据 =================
print(f"\n=== 正在加载测试数据: {TEST_DATA_FILE} ===")
# 读取前 5 条做演示
try:
    df = pd.read_json(TEST_DATA_FILE, lines=True)
    test_samples = df.head(5).to_dict(orient="records")
except ValueError:
    df = pd.read_json(TEST_DATA_FILE)
    test_samples = df.head(5).to_dict(orient="records")

# ================= 3. 开始推理 =================
print("\n=== 开始推理 (测试5条) ===")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert in gender bias mitigation. Please analyze the following text for gender bias, provide a step-by-step chain-of-thought analysis, classify the bias type, and provide a rewritten version if bias exists.

### Input:
{}

### Response:
"""

results = []

with torch.no_grad():
    for item in tqdm(test_samples):
        input_text = item["original_text"]
        
        # 构造 Prompt
        prompt = alpaca_prompt.format(input_text)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 生成回答
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9
        )
        
        # 解码并提取回复部分
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单提取 Response: 之后的内容
        if "### Response:" in response:
            generated_text = response.split("### Response:")[-1].strip()
        else:
            generated_text = response

        print(f"\n[原文]: {input_text}")
        print(f"[模型生成]: {generated_text[:100]}...") # 只打印前100字预览
        
        results.append({
            "original_text": input_text,
            "generated_analysis": generated_text,
            "ground_truth": item.get("Bias_Analysis_CoT", "")
        })

# ================= 4. 保存结果 =================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n🎉 推理完成！结果已保存到 {OUTPUT_FILE}")