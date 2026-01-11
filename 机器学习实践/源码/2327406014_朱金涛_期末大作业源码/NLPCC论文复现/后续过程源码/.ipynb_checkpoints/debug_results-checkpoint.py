import json

# 读取结果文件
FILE_PATH = "benchmark_results_test.json"

try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📂 文件共有 {len(data)} 条数据")
    
    if len(data) > 0:
        item = data[0]
        print("\n🔍 --- 第一条数据样本 ---")
        print(f"🔑 包含的键 (Keys): {list(item.keys())}")
        
        print("\n1️⃣ [Ground Truth] (标准答案):")
        print(item.get('ground_truth'))
        
        print("\n2️⃣ [Model Output Raw] (模型生成的原始文本):")
        raw_out = item.get('model_output_raw', '')
        print(f"'{raw_out}'")  # 加引号以便看清是否有空格或换行
        
        print("\n3️⃣ [Model Parsed] (代码解析后的JSON):")
        print(item.get('model_parsed'))
        
        # 尝试现场解析
        print("\n🛠️ [现场解析测试]:")
        start = raw_out.find("{")
        end = raw_out.rfind("}") + 1
        print(f"  - 找到大括号位置: Start={start}, End={end}")
        if start != -1 and end != -1:
            print(f"  - 截取内容: {raw_out[start:end]}")
        else:
            print("  - ⚠️ 警告: 未找到成对的大括号 {}，无法解析为 JSON！")

    else:
        print("⚠️ 文件是空的！")

except FileNotFoundError:
    print(f"❌ 找不到文件: {FILE_PATH}")