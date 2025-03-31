# README.md

## WELCOME

Welcome to Memory Orb, we hope you enjoy this brainstorming journey

## Introduction

In this example, you will have a conversation with the US Treasury Secretary to get their latest thinking and gain investment insights

### Example Questions

## Getting Started

### 1. Clone the Code

```bash
git clone https://github.com/memory-orb/orb-back.git
cd orb-back
```

### 2. Configure the Backend

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Install Qdrant Database

We recommend installing Qdrant using Docker:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

#### Install Ollama Embedding Model

```bash
ollama pull mxbai-embed-large
```

#### Configure API Key

Modify the API key in src/config.py:

```python
API_KEY = "your DeepSeek API-key"

BASE_URL = "https://api.deepseek.com"
```


### 3. Run the Backend Server

```bash
python run_memory_orb.py --port 5002
```

By default, the backend service will run on port 5002. You can specify a different port using the `--port` parameter.

### 5. Usage Instructions

1. Type messages in the chat window to converse with the AI
2. The system will automatically store conversation content as memories
3. Use the "Save Memory" button to export memory snapshots
4. Use the "Load Memory" button to import previously saved memory snapshots

## Project Structure

- ```bash
  - `src/` (Python backend code)
    - `api.py` (Flask API interface)
    - `chat.py` (Chat functionality implementation)
    - `config.py` (Configuration information)
    - `memory_store.py` (Memory storage and management)
    - `main.py` (Program entry point)
  

## Other Configurations

### LLM

#### openAI

```python
API_KEY = "你的密钥"

BASE_URL = "https://api.openai.com"

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    }
}
```

#### Google AI

```python
API_KEY = "你的密钥"

BASE_URL = "https://api.gemini.com"
config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "gemini/gemini-pro",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    }
}
```

## Troubleshooting

1. If the frontend cannot connect to the backend, please check:
   - Whether the backend service is running correctly
   - Whether the API address is configured correctly
   - Whether there are CORS restrictions
2. If the memory feature is not working, please check:
   - Whether the Qdrant database is running correctly
   - Whether the connection information in the configuration file is correct
3. If AI replies fail, please check:
   - Whether the API key is valid
   - Whether the network connection is normal

We hope you enjoy using the application!