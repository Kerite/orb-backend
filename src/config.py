# Configuration information
from openai import OpenAI
from mem0 import Memory
from qdrant_client import QdrantClient
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

load_dotenv(verbose=True)

# 全局用户对话记忆（Key: user_id, Value: List[Message]）
global_memory = {}

# API configuration
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_API_BASE")

if API_KEY is None or BASE_URL is None:
    raise ValueError("Please set the OPENAI_API_KEY and OPENAI_API_BASE environment variables.")

llm = ChatOpenAI(
    base_url="https://api.openai.com/v1",
    api_key=API_KEY,
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1024
)

# 基础集合名称前缀
BASE_COLLECTION_NAME = "memory_orb"

# 根据用户ID生成集合名称
def get_collection_name(user_id="default_user"):
    return f"{BASE_COLLECTION_NAME}_{user_id}"

# Configuration information
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": API_KEY,
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 1.0
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": BASE_COLLECTION_NAME,  # 默认集合名称，将根据用户ID动态替换
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1024,
        }
    },
    "version": "v1.1",
}

# Initialize client
openai_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Create a direct Qdrant client instance for snapshot operations
qdrant_client = QdrantClient(
    host=config["vector_store"]["config"]["host"],
    port=config["vector_store"]["config"]["port"]
    # api_key=config["vector_store"]["config"]["api_key"]
)

embedder_info = OpenAIEmbeddings(
    model=config["embedder"]["config"]["model"],
)

# 获取用户特定的内存配置
def get_user_config(user_id="default_user"):
    user_config = config.copy()
    user_config["vector_store"] = config["vector_store"].copy()
    user_config["vector_store"]["config"] = config["vector_store"]["config"].copy()
    user_config["vector_store"]["config"]["collection_name"] = get_collection_name(user_id)
    return user_config

# 获取用户特定的内存实例
def get_user_memory(user_id="default_user"):
    user_config = get_user_config(user_id)
    return Memory.from_config(user_config)

# 默认内存对象
memory = Memory.from_config(config)

# 定义集合参数
COLLECTION_NAME = "episodic_memory"
VECTOR_SIZE = 1024  # 与mxbai-embed-large模型输出维度一致

payload_schema = {
    "conversation": str,             # 对话内容（TEXT）
    "context_tags": list[str],       # 上下文标签（TEXT5_ARRAY）
    "conversation_summary": str,     # 摘要文本（TEXT）
    "what_worked": str,              # 有效策略（TEXT）
    "what_to_avoid": str             # 需避免内容（TEXT）
}

# 定义集合参数
COLLECTION_NAME = "episodic_memory"
VECTOR_SIZE = 1024  # 与mxbai-embed-large模型输出维度一致

def init_user_collection(user_id: str):
    collection_name = get_collection_name(user_id)
    qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=config["vector_store"]["config"]["embedding_model_dims"],
        distance=Distance.COSINE
    )
)