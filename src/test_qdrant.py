from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333, timeout=60)

try:
    print("Collections:", client.get_collections())
except Exception as e:
    print("连接失败:", str(e))