from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import traceback  
from .memory_store import export_qdrant_snapshot, import_qdrant_snapshot
from .chat import chat_with_memories
from .config import get_user_memory, openai_client, llm, global_memory
import logging
from flask import Response, stream_with_context
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .memory_v2 import add_episodic_memory

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initializing the Flask application
app = Flask(__name__)
CORS(app)  

@app.route('/api/export-memory', methods=['POST'])
def export_memory():
    """Export memory snapshot and return file download"""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'default_user')
        
        # 使用用户特定集合导出快照
        snapshot_path = export_qdrant_snapshot(user_id=user_id)
        
        if not snapshot_path or not os.path.exists(snapshot_path):
            return jsonify({"error": "Snapshot export failed"}), 500
            
        # Return to file download
        return send_file(
            snapshot_path,
            as_attachment=True,
            download_name=os.path.basename(snapshot_path),
            mimetype='application/octet-stream'
        )
    except Exception as e:
        print(f"Export Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/import-memory', methods=['POST'])
def import_memory():
    """Importing a memory snapshot from an uploaded file"""
    temp_file_path = None
    try:
        # 获取用户ID
        user_id = request.form.get('user_id', 'default_user')
        
        # 检查是否有上传的文件
        if 'snapshot' not in request.files:
            print("Uploaded file not found")
            return jsonify({"error": "Uploaded file not found"}), 400
            
        file = request.files['snapshot']
        
        # Check the file name
        if file.filename == '':
            print("No file selected")
            logger.debug("No file selected: %s", file.filename)
            return jsonify({"error": "No file selected"}), 400
            
        print(f"Received file: {file.filename}")
        
        # Save temporary files
        temp_file_path = tempfile.mktemp(suffix='.snapshot')
        file.save(temp_file_path)
        
        file_size = os.path.getsize(temp_file_path)
        print(f"Saved temporary file: {temp_file_path}, size: {file_size} bytes")
        
        # Importing a Snapshot
        success = import_qdrant_snapshot(temp_file_path, user_id=user_id)
        
        if success:
            print("Import Success")
            return jsonify({"message": "Memory snapshot imported successfully"})
        else:
            print("Import failed")
            return jsonify({"error": "Memory snapshot import failed"}), 500
    except Exception as e:
        print(f"An error occurred during the import process: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        # Make sure to delete temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Temporary file deleted: {temp_file_path}")
            except Exception as e:
                print(f"Failed to delete temporary file: {str(e)}")


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests and return AI responses in a streaming manner"""
    try:
        # Get request data
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Message cannot be empty"}), 400
            
        message = data['message']
        user_id = data.get('user_id', 'default_user')
        
        print(f"Received chat message from user {user_id}: {message}")
        
        # Use a generator function for streaming response
        def generate():
            try:
                # 为特定用户获取内存实例
                user_memory = get_user_memory(user_id)
                
                # 获取相关内存
                relevant_memories = user_memory.search(query=message, user_id=user_id, limit=10)
                memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
                                
                # 生成助手响应
                system_prompt = f"You are a helpful AI. Answer the question based on query and memories.\nUser Memories:\n{memories_str}"
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
                
                # Use streaming output
                stream = openai_client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=messages,
                    stream=True
                )
                
                # Collect the complete response for storage
                assistant_response = ""
                
                for chunk in stream:
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                        if content:
                            # Send data in SSE format
                            yield f"data: {json.dumps({'content': content})}\n\n"
                            assistant_response += content
                
                # Create new conversation memory
                messages.append({"role": "assistant", "content": assistant_response})
                user_memory.add(messages, user_id=user_id)
                
                # Send end marker
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
        return Response(stream_with_context(generate()), 
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", 
                                "X-Accel-Buffering": "no",
                                "Access-Control-Allow-Origin": "*"})
        
    except Exception as e:
        print(f"Error handling chat request: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
# 增加工作记忆
@app.route('/api/chatV2', methods=['POST'])
def chatV2():
    """Handle chat requests and return AI responses in a streaming manner"""
    try:
        # Get request data
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "Message cannot be empty"}), 400
            
        message = data['message']
        user_id = data.get('user_id', 'default_user')
        
        print(f"Received chat message from user {user_id}: {message}")
        
        # 使用langchain请求ds返回数据
        # 将对话输入和输出append到message
        
        # Use a generator function for streaming response
        def generate():
            try:
                
                # 初始化或获取用户对话历史
                if user_id not in global_memory:
                    system_prompt = SystemMessage(
                        content="You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence."
                    )
                    global_memory[user_id] = [system_prompt]
                    
                messages = global_memory[user_id]
                # 添加用户进行对话
                user_message = HumanMessage(content=message)
                messages.append(user_message)
                
                response = llm.invoke(messages)
                print("\nAI Message: ", response.content)
                yield f"data: {json.dumps({'content': response.content})}\n\n"
                
                # 添加AI返回对话
                messages.append(response)
                # user_memory.add(messages, user_id=user_id)
                
                for i, msg in enumerate(messages, start=0):
                    print(f"Message {i} - {msg.type.upper()}: {msg.content}")
                    
                # Send end marker
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            
        return Response(stream_with_context(generate()), 
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", 
                                "X-Accel-Buffering": "no",
                                "Access-Control-Allow-Origin": "*"})
        
    except Exception as e:
        print(f"Error handling chat request: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/api/save_episodic_memory', methods=['GET'])
def episodic_memory():
    try:
        # 获取用户id
        # data = request.json
        # if not data or 'message' not in data:
        #     return jsonify({"error": "Message cannot be empty"}), 400
        
        # user_id = data.get('user_id', 'default_user')
        
        # # 获取用户所有对话
        # if user_id not in global_memory:
        #     return jsonify({"error": "Message cannot be empty"}), 400
                
        # messages = global_memory[user_id]
        messages = global_memory["user_57d57b4dab129529dffc6f6065edeb8a"]
        # 传入message + 传入promt
        add_episodic_memory(messages)
        # 解析返回打印
    except Exception as e:
        print(f"Export Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500 
    
@app.route('/api/del_episodic_memory', methods=['DELETE'])
def delete_episodic_memory():
    """Delete a user's episodic memory from global storage"""
    try:
        # 获取请求数据
        data = request.json
        if not data or 'user_id' not in data:
            return jsonify({"error": "user_id is required"}), 400

        user_id = data['user_id']
        
        # 检查用户是否存在
        if user_id not in global_memory:
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        # 执行删除操作
        del global_memory[user_id]
        
        # 返回成功响应
        return jsonify({
            "success": True,
            "message": f"Episodic memory for user {user_id} deleted"
        }), 200
        
    except Exception as e:
        print(f"Deletion Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def run_api(host='localhost', port=5000, debug=False):
    """Run the API server"""
    app.run(host=host, port=port, debug=debug, use_reloader=debug)