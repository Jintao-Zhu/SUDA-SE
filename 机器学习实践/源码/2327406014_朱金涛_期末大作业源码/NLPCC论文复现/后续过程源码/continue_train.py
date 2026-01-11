import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel # 确保导入 PeftModel

# ================= 配置区域 =================
MODEL_PATH = "/public/home/lilingzhi/Qwen2.5-7B-Instruct"
DATA_FILE = "pianjian_cot_backup.json" 

# ⚠️ 关键：继续使用旧的输出目录，因为模型权重就在这里
OUTPUT_DIR = "qwen_lora_outputs_ddp" 
# 找到最大的 checkpoint 目录路径 (模型权重就在这里)
LATEST_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint-93") 

# ================= 1. 加载数据 =================
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    print(f"=== 正在准备断点续训 (目标：续跑 7 轮) ===")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ 找不到文件: {DATA_FILE}")

try:
    df = pd.read_json(DATA_FILE, lines=True)
except ValueError:
    df = pd.read_json(DATA_FILE)
dataset = Dataset.from_pandas(df)

# ================= 2. 加载模型 (关键修复：手动加载权重) =================
try:
    if local_rank == 0:
        print(f"✅ 找到 Checkpoint: {LATEST_CHECKPOINT_PATH}")
        
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config=config, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token 
    
    # 1. 加载底座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16, 
        local_files_only=True
    )

    # 2. 从 Checkpoint 路径加载 PEFT 适配器权重 (模型现在已经是最新的了)
    model = PeftModel.from_pretrained(base_model, LATEST_CHECKPOINT_PATH)
    
    if local_rank == 0:
        print("✅ 模型权重加载成功！")

except Exception as e:
    raise RuntimeError(f"❌ 模型加载失败: {e}")

# ================= 3. 配置 LoRA (保持不变) =================
# PEFT 适配器已经加载，这里只需要配置参数，不再调用 get_peft_model
model.gradient_checkpointing_enable() 
model.enable_input_require_grads()

if local_rank == 0:
    model.print_trainable_parameters()
    print("当前训练状态: 从第 4 轮开始运行...")

# ================= 4. 数据格式化 (保持不变) =================
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an expert in gender bias mitigation. Please analyze the following text for gender bias, provide a step-by-step chain-of-thought analysis, classify the bias type, and provide a rewritten version if bias exists.

### Input:
{}

### Response:
{}"""

def preprocess_function(examples):
    inputs = examples["original_text"]
    targets = examples["Bias_Analysis_CoT"]
    model_inputs = []
    for i in range(len(inputs)):
        prompt = alpaca_prompt.format(inputs[i], "")
        full_text = prompt + str(targets[i]) + tokenizer.eos_token
        tokenized = tokenizer(full_text, truncation=True, max_length=2048, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        model_inputs.append(tokenized)
    return {k: [d[k] for d in model_inputs] for k in model_inputs[0].keys()}

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# ================= 5. 开始续训 (最终修复) =================
if local_rank == 0:
    print("\n=== 启动续训 (目标：再跑 7 轮) ===")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8, 
    learning_rate=2e-4,
    num_train_epochs=7, # 跑剩下的7轮
    logging_steps=10,
    
    # ❌ 核心修改：关闭 fp16，改用 fp32
    fp16=False, 
    
    save_strategy="epoch",
    save_total_limit=2,
    optim="adamw_torch",
    report_to="none",
    ddp_find_unused_parameters=False, 
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

# 🔥 核心启动：不使用 resume_from_checkpoint=True
# 因为模型已经手动加载了最新的权重，直接开始新的 Trainer 即可
trainer.train()

# ================= 6. 保存最终结果 =================
if local_rank == 0:
    print(f"\n=== 10轮训练完成，保存最终模型到 {OUTPUT_DIR} ===")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("🎉 续训任务圆满结束！")