import streamlit as st
import torch
import dashscope
from transformers import AutoTokenizer, AutoModelForTokenClassification
from py2neo import Graph
from http import HTTPStatus

# ================= 配置区域 =================
dashscope.api_key = "sk-1a5fe8ff79f24ba88eef4ef7c52f60c1" # <--- 填你的 Key
NEO4J_PASSWORD = "zjt20050213"                       # <--- 填你的密码
MODEL_PATH = './saved_model'
# ============================================

# 页面标题
st.set_page_config(page_title="医疗知识图谱问答系统", page_icon="🏥")
st.title("🏥 智能医疗问答助手")
st.markdown("基于 **知识图谱 (Neo4j)** + **大模型 (Qwen-Plus)** + **NER (BERT)**")

# 1. 初始化连接 (加缓存，只加载一次)
@st.cache_resource
def init_resources():
    # 连接 Neo4j
    try:
        graph = Graph("bolt://localhost:7687", auth=("neo4j", NEO4J_PASSWORD))
    except:
        return None, None, None, None
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    # 标签映射
    labels_list = ['B-检查项目', 'B-治疗方法', 'B-疾病', 'B-疾病症状', 'B-科目', 'B-药品', 'B-药品商', 'B-食物', 
                   'I-检查项目', 'I-治疗方法', 'I-疾病', 'I-疾病症状', 'I-科目', 'I-药品', 'I-药品商', 'I-食物', 'O']
    id2tag = {i: tag for i, tag in enumerate(labels_list)}
    
    return graph, tokenizer, model, id2tag, device

graph, tokenizer, model, id2tag, device = init_resources()

if graph is None:
    st.error("❌ 数据库或模型加载失败，请检查后台日志。")
    st.stop()

# 侧边栏
with st.sidebar:
    st.success("✅ 系统状态：在线")
    st.info("💡 提示：试着问问 '感冒了吃什么药？' 或 '高血压有什么症状？'")

# 2. 核心函数 (直接复用你之前的逻辑)
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
    preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0][1:-1]
    
    entities = {}
    curr_entity = ""
    curr_type = ""
    for char, tag_id in zip(token_list, preds):
        if tag_id >= len(id2tag): continue
        tag = id2tag[tag_id]
        if tag.startswith("B-"):
            if curr_entity:
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
            curr_entity = ""; curr_type = ""
    if curr_entity:
        if curr_type not in entities: entities[curr_type] = []
        entities[curr_type].append(curr_entity)
    return entities

def get_answer(question):
    # NER
    entities = extract_entities(question)
    
    # LLM Intent
    prompt = f"""
    现在，你是一个机器人医生。目前我的图数据库中有8类实体: 疾病、药品、药品商、疾病症状、食物、检查项目、治疗方法、科目。
    查询语句格式应为: 
    类型1(查询属性): "1 疾病名称 属性名" (属性包括: 疾病简介, 疾病病因, 预防措施, 治疗周期, 治愈概率, 疾病易感人群)
    类型2(查询关系): "2 疾病名称 关系名称 实体类别" (关系包括: 疾病使用药品, 疾病宜吃食物, 疾病忌吃食物, 疾病所需检查, 疾病所属科目, 疾病的症状, 治疗的方法, 疾病并发疾病)
    用户问题: {question}
    请直接输出查询语句，多个用逗号隔开。
    """
    messages = [{'role': 'system', 'content': '你是一个专业的医疗知识图谱问答助手。'}, {'role': 'user', 'content': prompt}]
    resp = dashscope.Generation.call(model="qwen-plus", messages=messages, result_format='message')
    intent_str = resp.output.choices[0]['message']['content']
    
    # Graph Query
    results = []
    queries = intent_str.split(',')
    for q in queries:
        parts = q.strip().split()
        if len(parts) < 3: continue
        cypher = ""
        try:
            name = parts[1]
            if parts[0] == '1':
                attr_map = {"疾病简介": "desc", "疾病病因": "cause", "预防措施": "prevent", "治疗周期": "cure_lasttime", "治愈概率": "cured_prob"}
                attr = attr_map.get(parts[2], parts[2])
                cypher = f"MATCH (n:Disease {{name: '{name}'}}) RETURN n.{attr} as result"
            elif parts[0] == '2' and len(parts) >= 4:
                rel = parts[2]
                target_map = {"药品": "Drug", "食物": "Food", "检查项目": "Check", "科目": "Department", "疾病症状": "Symptom", "治疗方法": "CureWay", "疾病": "Disease"}
                target = target_map.get(parts[3], "Node")
                cypher = f"MATCH (n:Disease {{name: '{name}'}})-[:{rel}]->(m:{target}) RETURN m.name as result"
            
            if cypher:
                data = graph.run(cypher).data()
                if data: results.append(f"【{name}】: {str([d['result'] for d in data])}")
        except: continue
        
    return entities, intent_str, results

# 3. 聊天界面逻辑
# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您的医疗问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 处理并回复
    with st.chat_message("assistant"):
        with st.status("正在思考中...", expanded=True):
            st.write("🔍 正在进行命名实体识别...")
            entities, intent, answers = get_answer(prompt)
            st.write(f"🏷️ 识别实体: {entities}")
            
            st.write("🧠 正在分析意图 (LLM)...")
            st.write(f"🎯 查询指令: {intent}")
            
            st.write("🕸️ 正在查询知识图谱...")
            
        if answers:
            response = "根据知识库查询，结果如下：\n\n" + "\n".join(answers)
        else:
            response = "抱歉，知识库中暂时没有查到相关信息，或者该问题不属于医疗知识图谱范围。"
            
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})