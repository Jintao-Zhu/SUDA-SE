import pandas as pd
import json
import torch
import dashscope
from transformers import BertTokenizer, BertForTokenClassification
from py2neo import Graph
from http import HTTPStatus

# ==================== 配置区域 (请修改这里) ====================
# 1. 填入你的阿里 DashScope API Key
dashscope.api_key = "sk-1a5fe8ff79f24ba88eef4ef7c52f60c1"  # <--- 把你的Key粘贴在这里

# 2. 你的 Neo4j 密码
NEO4J_PASSWORD = "zjt20050213"  # <--- 填入你的Neo4j密码

# 3. Excel 文件配置
EXCEL_FILE = 'D:\\课程资料\\机器学习实践\\期末大作业\\期末大作业\\医疗问答\\questions.csv'  # 假设你的文件名是这个，如果是csv请改后缀
COLUMN_NAME = 'content'       # <--- 请确认你Excel里存放问题的列名(表头)叫什么？

# 4. 模型选择 (这里指定了使用 qwen-plus)
LLM_MODEL = "qwen-plus"
# ============================================================

# 连接 Neo4j
try:
    graph = Graph("bolt://localhost:7687", auth=("neo4j", NEO4J_PASSWORD))
    print("✅ Neo4j 连接成功")
except Exception as e:
    print(f"❌ Neo4j 连接失败: {e}")
    exit()

# 加载 NER 模型
print("正在加载 NER 模型...")
MODEL_PATH = './saved_model'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # 加载标签映射 (需要与训练时一致)
    # 这里我们硬编码训练时的标签列表，确保对应正确
    # 注意：如果你的标签列表顺序变了，这里需要调整。最稳妥的是保存训练时的tag2id。
    # 根据你之前的截图，这是你训练时的标签顺序：
    labels_list = ['B-检查项目', 'B-治疗方法', 'B-疾病', 'B-疾病症状', 'B-科目', 'B-药品', 'B-药品商', 'B-食物', 
                   'I-检查项目', 'I-治疗方法', 'I-疾病', 'I-疾病症状', 'I-科目', 'I-药品', 'I-药品商', 'I-食物', 'O']
    id2tag = {i: tag for i, tag in enumerate(labels_list)}
    print("✅ NER 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# NER 推理函数
def extract_entities(text):
    tokens = [tokenizer.cls_token_id]
    token_list = []
    for char in text:
        token_list.append(char)
        tokens.extend(tokenizer.encode(char, add_special_tokens=False))
    tokens.append(tokenizer.sep_token_id)
    
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=2).cpu().numpy()[0]
    
    # 解析 BIO 标签
    entities = {}
    curr_entity = ""
    curr_type = ""
    
    # 为了对齐，去掉头尾的 [CLS] [SEP]
    preds = preds[1:-1]
    
    # 简单的实体提取逻辑
    for char, tag_id in zip(token_list, preds):
        if tag_id >= len(labels_list): continue
        tag = id2tag[tag_id]
        
        if tag.startswith("B-"):
            if curr_entity: # 保存上一个
                if curr_type not in entities: entities[curr_type] = []
                entities[curr_type].append(curr_entity)
            curr_entity = char
            curr_type = tag.split("-")[1]
        elif tag.startswith("I-") and curr_type == tag.split("-")[1]:
            curr_entity += char
        else:
            if curr_entity:
                if curr_type not in entities: entities[curr_type] = []
                entities[curr_type].append(curr_entity)
            curr_entity = ""
            curr_type = ""
            
    if curr_entity:
        if curr_type not in entities: entities[curr_type] = []
        entities[curr_type].append(curr_entity)
        
    return entities

# 大模型调用函数
def call_llm(prompt):
    messages = [{'role': 'system', 'content': '你是一个专业的医疗知识图谱问答助手。'},
                {'role': 'user', 'content': prompt}]
    
    try:
        response = dashscope.Generation.call(
            model=LLM_MODEL,  # <--- 这里指定了使用 qwen-plus
            messages=messages,
            result_format='message',
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0]['message']['content']
        else:
            return f"Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Exception: {e}"

# 解析意图并查询 Neo4j
def execute_query(intent_str):
    # PPT要求的格式是: "1 疾病名称 属性" 或 "2 疾病名称 关系 实体类别"
    # 我们需要解析这个字符串并生成 Cypher
    results = []
    queries = intent_str.split(',') # 多个查询用逗号隔开
    
    for q in queries:
        q = q.strip()
        parts = q.split()
        if len(parts) < 3: continue
        
        q_type = parts[0]
        name = parts[1]
        
        cypher = ""
        try:
            if q_type == '1': # 查询属性
                attr = parts[2]
                # 属性名映射 (PPT 32页对应)
                attr_map = {
                    "疾病简介": "desc", "疾病病因": "cause", "预防措施": "prevent",
                    "治疗周期": "cure_lasttime", "治愈概率": "cured_prob", "疾病易感人群": "easy_get"
                }
                db_attr = attr_map.get(attr, attr) # 找不到就用原名
                cypher = f"MATCH (n:Disease {{name: '{name}'}}) RETURN n.{db_attr} as result"
                
            elif q_type == '2': # 查询关系
                # 格式: 2 疾病名称 关系名称 实体类别
                if len(parts) >= 4:
                    rel_name = parts[2]
                    target_label = parts[3]
                    # 实体类别映射 (英文)
                    label_map = {
                        "药品": "Drug", "食物": "Food", "检查项目": "Check", 
                        "科目": "Department", "疾病症状": "Symptom", "治疗方法": "CureWay", "疾病": "Disease"
                    }
                    target_label_en = label_map.get(target_label, "Node")
                    cypher = f"MATCH (n:Disease {{name: '{name}'}})-[:{rel_name}]->(m:{target_label_en}) RETURN m.name as result"
            
            if cypher:
                print(f"  [Cypher]: {cypher}")
                data = graph.run(cypher).data()
                results.append(data)
                
        except Exception as e:
            print(f"  [Query Error]: {e}")
            
    return results

# ==================== 主流程 ====================
def main():
    # 1. 读取 Excel
    print(f"正在读取 {EXCEL_FILE}...")
    try:
        df = pd.read_csv(EXCEL_FILE)
        # 如果找不到指定列，尝试用第一列
        if COLUMN_NAME not in df.columns:
            print(f"⚠️ 警告: 未找到列名 '{COLUMN_NAME}'，将默认使用第一列作为问题列。")
            questions = df.iloc[:, 0].astype(str).tolist()
        else:
            questions = df[COLUMN_NAME].astype(str).tolist()
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 只取前 100 条 (PPT要求)
    questions = questions[:100]
    final_output = []

    # 2. 循环处理
    print(f"开始处理 {len(questions)} 个问题...")
    
    # PPT Source 49 提供的 Prompt 模版
    PROMPT_TEMPLATE = """
    现在，你是一个机器人医生，用户对你输入问题，你需要精准的理解问题的内容，根据其含义构建Neo4j数据库的查询语句...
    (此处省略长 Prompt，为了代码整洁，我们用简化的核心逻辑，实际运行时请把 PPT 49 页完整的 Prompt 文字贴在这里，或者直接使用下面的精简版)
    
    提示:目前我的图数据库中有8类实体: 疾病、药品、药品商、疾病症状、食物、检查项目、治疗方法、科目。
    查询语句格式应为: 
    类型1(查询属性): "1 疾病名称 属性名" (属性包括: 疾病简介, 疾病病因, 预防措施, 治疗周期, 治愈概率, 疾病易感人群)
    类型2(查询关系): "2 疾病名称 关系名称 实体类别" (关系包括: 疾病使用药品, 疾病宜吃食物, 疾病忌吃食物, 疾病所需检查, 疾病所属科目, 疾病的症状, 治疗的方法, 疾病并发疾病)
    
    用户问题: {question}
    
    请直接输出查询语句，不要输出其他废话。如果有多个查询用逗号隔开。
    例如: 1 口臭 疾病简介, 2 口臭 治疗的方法 治疗方法
    """

    for i, q in enumerate(questions):
        print(f"\n--- 处理第 {i+1} 条: {q} ---")
        
        # 步骤 A: NER 识别
        entities = extract_entities(q)
        print(f"  [NER]: {entities}")
        
        # 步骤 B: 构造 Prompt 并调用 LLM 获取意图
        # 将 NER 结果也放入 Prompt 辅助模型 (可选)
        prompt = PROMPT_TEMPLATE.format(question=q)
        llm_res = call_llm(prompt)
        print(f"  [LLM Intent]: {llm_res}")
        
        # 步骤 C: 解析意图并查询 Graph
        db_results = execute_query(llm_res)
        print(f"  [DB Result]: {db_results}")
        
        # 步骤 D: 保存结果
        item = {
            "id": i,
            "question": q,
            "ner_results": entities,
            "intent_raw": llm_res,
            "query_results": db_results
        }
        final_output.append(item)

    # 3. 写入文件
    with open('final_result.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print("\n🎉 全部完成！结果已保存为 final_result.json")

if __name__ == "__main__":
    main()