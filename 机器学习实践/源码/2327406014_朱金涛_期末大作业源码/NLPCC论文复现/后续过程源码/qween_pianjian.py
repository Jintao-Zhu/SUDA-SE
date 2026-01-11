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
from peft import LoraConfig, get_peft_model, TaskType

# ================= 配置区域 =================
# ✅ 模型本地绝对路径 (这是你刚刚下载好的位置)
MODEL_PATH = "/public/home/lilingzhi/Qwen2.5-7B-Instruct"

# ✅ 数据文件 (就在当前目录下)
DATA_FILE = "pianjian_cot_backup.json" 

# 输出目录
OUTPUT_DIR = "qwen_lora_outputs_ddp"

# ================= 1. 环境与数据检查 =================
# 只在主进程打印日志，防止双卡刷屏
local_rank = int(os.environ.get("LOCAL_RANK", 0))

if local_rank == 0:
    print("=== 环境检查 ===")
    if torch.cuda.is_available():
        print(f"✅ 发现 GPU 数量: {torch.cuda.device_count()}")
        print(f"✅ 当前显卡: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("❌ 未检测到 GPU！")
    
    print(f"\n=== 正在加载数据: {DATA_FILE} ===")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ 找不到文件: {DATA_FILE}")

try:
    # 尝试按 JSON Lines 格式加载
    df = pd.read_json(DATA_FILE, lines=True)
except ValueError:
    # 失败则尝试标准 JSON
    df = pd.read_json(DATA_FILE)

dataset = Dataset.from_pandas(df)

if local_rank == 0:
    print(f"✅ 成功读取 {len(dataset)} 条数据")

# ================= 2. 加载模型 (原生 DDP 模式) =================
if local_rank == 0:
    print(f"\n=== 正在从本地加载模型: {MODEL_PATH} ===")

try:
    # 🛠️ 关键修复：手动加载 Config，防止自动识别出错
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    
    # 加载 Tokenizer
    # 注意：不再使用 trust_remote_code=True，避免加载错误代码
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, 
        config=config,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token 

    # 加载模型
    # ⚠️ 关键：DDP 模式下绝对不能写 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.float16, # V100 完美支持 FP16
        local_files_only=True
    )
    
    if local_rank == 0:
        print("✅ 模型加载成功！")

except Exception as e:
    raise RuntimeError(f"❌ 模型加载失败！请检查路径是否正确。\n错误信息: {e}")

# ================= 3. 配置 LoRA =================
if local_rank == 0:
    print("\n=== 正在配置 LoRA ===")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 开启梯度检查点 (大幅节省显存，防止 OOM)
model.gradient_checkpointing_enable() 
model.enable_input_require_grads()

if local_rank == 0:
    model.print_trainable_parameters()

# ================= 4. 数据格式化 =================
if local_rank == 0:
    print("\n=== 正在格式化数据 ===")

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
        
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        model_inputs.append(tokenized)
        
    return {k: [d[k] for d in model_inputs] for k in model_inputs[0].keys()}

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# ================= 5. 开始训练 (双卡参数优化) =================
if local_rank == 0:
    print("\n=== 开始双卡训练 ===")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # 显存策略：V100 32G 很大，但也经不住 Qwen-7B 随便造
    # 单卡 batch size 设为 2，配合梯度累积 8
    # 总 batch size = 2 (单卡) * 2 (卡数) * 8 (累积) = 32
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=8, 
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    fp16=True, # 开启半精度加速
    save_strategy="epoch",
    optim="adamw_torch", # 使用原生优化器，不依赖 bitsandbytes
    report_to="none",
    ddp_find_unused_parameters=False, # DDP 必须参数
    gradient_checkpointing=True, # 必须开启，否则容易 OOM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
)

trainer.train()

# ================= 6. 保存模型 =================
# 只在主进程保存，防止写冲突
if local_rank == 0:
    print(f"\n=== 训练完成，正在保存到 {OUTPUT_DIR} ===")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("🎉 全部完成！")