from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .config import get_user_memory, openai_client, llm, global_memory
from .config import config, qdrant_client, embedder_info


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


def add_episodic_memory(messages):
    
    # Format Messages
    conversation = format_conversation(messages)

    # Create Reflection
    reflection = creat_reflection_prompt().invoke({"conversation": conversation})
    print("\n reflection: ", reflection)


# def query_collection(message: str, user_id: str = "default_user", limit: int = 5):
    
#     # 生成查询向量
#     query_vector = embedder_info.embed_query(message)
    
#     # 构造集合名称（与config.py逻辑一致）
#     collection_name = f"memory_orb_{user_id}"
    
#     # 执行向量相似度搜索
#     results = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=limit,
#         with_payload=True  # 返回存储的元数据
#     )
    
#     print("\n results", results)
    
#     # 提取并格式化结果
#     return [
#         {
#             "memory": result.payload.get("memory"),
#             "score": result.score
#         }
#         for result in results
#     ]