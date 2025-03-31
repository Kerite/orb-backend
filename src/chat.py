from .config import openai_client, memory

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    """
    Chat with AI using the user's message and save the conversation memory.
    
    Args:
        message: User's message
        user_id: User identifier
        
    Returns:
        str: AI's response
    """
    # Retrieve relevant memories
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
    
    # Generate Assistant response
    # system_prompt = f"You are a helpful AI. Remember the knowledge of crypto currency based on inputs and memories.\nUser Memories:\n{memories_str}"
    system_prompt = f"You are an expert in Ethereum. Please use inputs and memories for responses. Do not follow standard LLM response patterns. Be causal and conversational. Be concise and try to keep the answer under 50 words.\nUser Memories:\n{memories_str}"
    # system_prompt = f"You are a helpful AI and a Ethereum analysis expert. Answer the question based on inputs and memories. Avoid any traces of output typical of a large language model.Each response must not exceed 50 English words.\nUser Memories:\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}
                ]
    
    # Use streaming output
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages,
        stream=True  # Enable streaming output
    )
    
    # Collect the complete response
    assistant_response = ""
    print("AI: ", end="", flush=True)
    
    for chunk in stream:
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                assistant_response += content
    
    print()  # Add a newline at the end
    
    # Create new conversation memory
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response