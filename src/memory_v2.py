from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .config import get_user_memory, openai_client, llm, global_memory, get_collection_name, init_user_collection
from .config import config, qdrant_client, embedder_info
from uuid import uuid4
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchText
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import requests
from typing import List

def creat_reflection_prompt():
    reflection_prompt_template = """
    You are analyzing conversations about research papers to create memories that will help guide future interactions. Your task is to extract key elements that would be most helpful when encountering similar academic discussions in the future.

    Review the conversation and create a memory reflection following these rules:

    1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
    2. Be extremely concise - each string should be one clear, actionable sentence
    3. Focus only on information that would be useful for handling similar future conversations
    4. Context_tags should be specific enough to match similar situations but general enough to be reusable

    Output valid JSON in exactly this format:
    {{
        "context_tags": [              // 2-4 keywords that would help identify similar future conversations
            string,                    // Use field-specific terms like "deep_learning", "methodology_question", "results_interpretation"
            ...
        ],
        "conversation_summary": string, // One sentence describing what the conversation accomplished
        "what_worked": string,         // Most effective approach or strategy used in this conversation
        "what_to_avoid": string        // Most important pitfall or ineffective approach to avoid
    }}

    Examples:
    - Good context_tags: ["transformer_architecture", "attention_mechanism", "methodology_comparison"]
    - Bad context_tags: ["machine_learning", "paper_discussion", "questions"]

    - Good conversation_summary: "Explained how the attention mechanism in the BERT paper differs from traditional transformer architectures"
    - Bad conversation_summary: "Discussed a machine learning paper"

    - Good what_worked: "Using analogies from matrix multiplication to explain attention score calculations"
    - Bad what_worked: "Explained the technical concepts well"

    - Good what_to_avoid: "Diving into mathematical formulas before establishing user's familiarity with linear algebra fundamentals"
    - Bad what_to_avoid: "Used complicated language"

    Additional examples for different research scenarios:

    Context tags examples:
    - ["experimental_design", "control_groups", "methodology_critique"]
    - ["statistical_significance", "p_value_interpretation", "sample_size"]
    - ["research_limitations", "future_work", "methodology_gaps"]

    Conversation summary examples:
    - "Clarified why the paper's cross-validation approach was more robust than traditional hold-out methods"
    - "Helped identify potential confounding variables in the study's experimental design"

    What worked examples:
    - "Breaking down complex statistical concepts using visual analogies and real-world examples"
    - "Connecting the paper's methodology to similar approaches in related seminal papers"

    What to avoid examples:
    - "Assuming familiarity with domain-specific jargon without first checking understanding"
    - "Over-focusing on mathematical proofs when the user needed intuitive understanding"

    Do not include any text outside the JSON object in your response.

    Here is the prior conversation:

    {conversation}
    """
    reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
    return reflection_prompt | llm | RobustJsonParser()

class RobustJsonParser(JsonOutputParser):
    def parse(self, text: str):
        try:
            # 尝试提取第一个完整JSON对象
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except Exception as e:
            return {"error": f"解析失败: {str(e)}", "raw": text}

def format_conversation(messages):
    
    # Create an empty list placeholder
    conversation = []
    
    # Start from index 1 to skip the first system message
    for message in messages[1:]:
        conversation.append(f"{message.type.upper()}: {message.content}")
    
    # Join with newlines
    return "\n".join(conversation)

def embed_text(text: str) -> List[float]:
    """使用 Ollama 生成向量（保持与原始配置相同）"""
    response = requests.post(
        "http://localhost:11434/api/embeddings",  # 使用 HTTP 协议
        json={"model": "nomic-embed-text", "text": text}
    )
    if response.status_code != 200:
        raise Exception(f"Ollama API 请求失败: {response.text}")
    return response.json().get("embedding", [])

# 增加情景记忆
def add_episodic_memory(messages, user_id="default_user"):
    # 初始化用户集合
    collection_name = get_collection_name(user_id)
    if not qdrant_client.collection_exists(collection_name):
        init_user_collection(user_id)  # 确保调用初始化
        print("/n 初始化")
    else:
        print("/n 已初始化")

    # 生成嵌入向量
    conversation = format_conversation(messages)
    reflection = creat_reflection_prompt().invoke({"conversation": conversation})
    print("/n",reflection)
    
    summary = reflection.get('conversation_summary', "")
    embedding = embed_text([summary])[0]

    # 构造Qdrant数据点
    point = PointStruct(
        id=str(uuid4()),
        vector=embedding,
        payload={
            "conversation": conversation,
            "context_tags": reflection.get('context_tags', []),
            "conversation_summary": reflection.get('conversation_summary', ""),
            "what_worked": reflection.get('what_worked', ""),
            "what_to_avoid": reflection.get('what_to_avoid', "")
        }
    )

    # 批量插入
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[point]
    )


def episodic_recall(query: str, user_id: str = "default_user", alpha=0.5):
    collection_name = get_collection_name(user_id)
    
    # 生成双路查询条件
    vector = embedder_info.embed_query(query)
    bm25_filter = Filter(
        must=[FieldCondition(key="conversation", match=MatchText(text=query))]
    )

    # 混合检索实现
    vector_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=5
    )
    
    keyword_results = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=bm25_filter,
        limit=5
    )

    # 结果融合算法
    combined = hybrid_merge(
        vector_results, 
        keyword_results,
        alpha=alpha
    )
    return combined[:3]  # 返回Top3结果

def hybrid_merge(vector_res, keyword_res, alpha):
    # 实现得分加权融合算法
    scores = {}
    for idx, item in enumerate(vector_res):
        scores[item.id] = alpha * (1 - item.score)  # Qdrant返回余弦相似度得分
    
    for idx, item in enumerate(keyword_res):
        bm25_score = (idx + 1) / len(keyword_res)  # 简化的BM25得分估算
        if item.id in scores:
            scores[item.id] += (1 - alpha) * bm25_score
        else:
            scores[item.id] = (1 - alpha) * bm25_score
    
    # 合并去重并排序
    all_items = {item.id: item for item in vector_res + keyword_res}
    sorted_items = sorted(
        all_items.values(),
        key=lambda x: scores.get(x.id, 0),
        reverse=True
    )
    return sorted_items


def episodic_system_prompt(query: str, user_id: str):
    memories = episodic_recall(query, user_id)
    if not memories:
        return SystemMessage(content="You are a helpful AI Assistant.")
    
    current_memory = memories[0].payload
    previous_convos = [m.payload["conversation_summary"] for m in memories[1:4]]
    
    prompt_template = f"""
    You are a helpful AI Assistant with conversation memory:
    
    Current Context Tags: {', '.join(current_memory['context_tags'])}
    Key Insight: {current_memory['what_worked']}
    Avoid: {current_memory['what_to_avoid']}
    Recent History: {' | '.join(previous_convos)}
    """
    return SystemMessage(content=prompt_template)