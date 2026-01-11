import json
from py2neo import Graph

# ================= 配置区域 =================
uri = "bolt://localhost:7687"
username = "neo4j"
password = "zjt20050213" 
file_path = r'D:\课程资料\机器学习实践\期末大作业\期末大作业\医疗问答\medical.json'
BATCH_SIZE = 1000  # 每次打包处理 1000 条（如果显存不够报错，可调小到 500）
# ===========================================

def load_data_robust(filepath):
    """(保持之前的强力读取函数不变)"""
    print(f"正在读取文件: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    data = []
    pos = 0
    total_len = len(content)
    while pos < total_len:
        while pos < total_len and content[pos].isspace(): pos += 1
        if pos >= total_len: break
        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            data.append(obj)
            pos = end_pos
            while pos < total_len and (content[pos].isspace() or content[pos] == ','): pos += 1
        except json.JSONDecodeError:
            pos += 1
    return data

# ================= 主程序 =================

print("1. 正在连接 Neo4j 数据库...")
try:
    graph = Graph(uri, auth=(username, password))
    
    # ⚠️ 这一步如果数据库本身有百万数据，会非常慢。
    # 如果卡在这里，建议直接去 Neo4j 安装目录删掉 data/databases/neo4j 文件夹重建
    print("   正在清空数据库 (如果数据量大可能会卡几分钟)...") 
    graph.delete_all() 
    print("   数据库已清空。")

    # 创建索引 (这是速度的关键，绝对不能少)
    print("2. 正在创建唯一索引...")
    for label in ["Disease", "Producer", "Symptom", "Department", "Check", "CureWay", "Food", "Drug"]:
        try:
            # 兼容不同版本的 Neo4j 语法
            graph.run(f"CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.name IS UNIQUE")
        except Exception:
            try:
                graph.run(f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.name IS UNIQUE")
            except: pass
    print("   索引创建完成。")

except Exception as e:
    print(f"连接或初始化失败: {e}")
    exit()

# 读取数据
data = load_data_robust(file_path)
total_count = len(data)
if not data:
    print("未读取到数据，退出。")
    exit()

print(f"3. 开始导入数据 (共 {total_count} 条)，采用 UNWIND 批处理模式...")

# ================= 核心：Cypher 批处理语句 =================
# 这条语句极其复杂，但效率极高。它在数据库内部完成循环。
cypher_query = """
UNWIND $batch_data AS row

// 1. 创建疾病节点
MERGE (d:Disease {name: row.name})
SET d.desc = row.desc,
    d.cause = row.cause,
    d.prevent = row.prevent,
    d.cure_lasttime = row.cure_lasttime,
    d.cured_prob = row.cured_prob,
    d.easy_get = row.easy_get

// 2. 处理症状 (Symptom)
FOREACH (item IN row.symptom | 
    MERGE (n:Symptom {name: item}) 
    MERGE (d)-[:疾病的症状]->(n))

// 3. 处理所属科室 (Department)
FOREACH (item IN row.cure_department | 
    MERGE (n:Department {name: item}) 
    MERGE (d)-[:疾病所属科目]->(n))

// 4. 处理所需检查 (Check)
FOREACH (item IN row.check | 
    MERGE (n:Check {name: item}) 
    MERGE (d)-[:疾病所需检查]->(n))

// 5. 处理治疗方法 (CureWay)
FOREACH (item IN row.cure_way | 
    MERGE (n:CureWay {name: item}) 
    MERGE (d)-[:治疗的方法]->(n))

// 6. 处理宜吃食物 (Food) - 合并两个字段
FOREACH (item IN row.do_eat + row.recommand_eat | 
    MERGE (n:Food {name: item}) 
    MERGE (d)-[:疾病宜吃食物]->(n))

// 7. 处理忌吃食物 (Food)
FOREACH (item IN row.not_eat | 
    MERGE (n:Food {name: item}) 
    MERGE (d)-[:疾病忌吃食物]->(n))

// 8. 处理药品 (Drug) - 合并两个字段
FOREACH (item IN row.common_drug + row.recommand_drug | 
    MERGE (n:Drug {name: item}) 
    MERGE (d)-[:疾病使用药品]->(n))

// 9. 处理并发症 (Disease)
FOREACH (item IN row.acompany | 
    MERGE (n:Disease {name: item}) 
    MERGE (d)-[:疾病并发疾病]->(n))

// 10. 特殊处理：药品详情 (Drug -> Producer)
// 使用 CASE WHEN 模拟 IF 逻辑，因为 Cypher 的 FOREACH 比较死板
FOREACH (detail IN row.drug_detail |
    FOREACH (_ IN CASE WHEN size(split(detail, ',')) >= 2 THEN [1] ELSE [] END |
        MERGE (drug:Drug {name: split(detail, ',')[0]})
        MERGE (producer:Producer {name: split(detail, ',')[1]})
        MERGE (producer)-[:生产]->(drug)
    )
)
"""

# ================= 执行批处理 =================
processed = 0

# 将数据切片，每 BATCH_SIZE 条作为一组
for i in range(0, total_count, BATCH_SIZE):
    batch = data[i : i + BATCH_SIZE]
    
    # 这里做一点小清洗，防止 None 值导致 Cypher 报错
    # 确保所有列表字段至少是空列表 [] 而不是 None
    cleaned_batch = []
    for item in batch:
        clean_item = item.copy()
        # 补全缺失的列表字段，防止 FOREACH 报错
        for list_field in ['symptom', 'cure_department', 'check', 'cure_way', 
                           'do_eat', 'recommand_eat', 'not_eat', 'common_drug', 
                           'recommand_drug', 'acompany', 'drug_detail']:
            if not isinstance(clean_item.get(list_field), list):
                clean_item[list_field] = []
        cleaned_batch.append(clean_item)

    try:
        # 发送整个 batch 给 Neo4j
        graph.run(cypher_query, batch_data=cleaned_batch)
        processed += len(batch)
        print(f"已处理 {processed}/{total_count} 条...")
    except Exception as e:
        print(f"警告：批次 {i} 处理出错，错误信息: {e}")

print("任务完成！")

# 统计部分
print("\n========== 最终统计 ==========")
try:
    n_nodes = graph.run("MATCH (n) RETURN count(n)").evaluate()
    n_rels = graph.run("MATCH ()-[r]->() RETURN count(r)").evaluate()
    print(f"总节点数: {n_nodes}")
    print(f"总关系数: {n_rels}")
except:
    pass