from langchain_core.messages import HumanMessage, SystemMessage


def chat_work_memories(message: str, user_id: str = "default_user") -> str:
    # Define System Prompt
    system_prompt = SystemMessage("You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.")

    # Start Storage for Historical Message History
    messages = [system_prompt]

    while True:

        # Get User's Message
        user_message = HumanMessage(input("\nUser: "))
        
        if user_message.content.lower() == "exit":
            break

        else:
            # Extend Messages List With User Message
            messages.append(user_message)

        # Pass Entire Message Sequence to LLM to Generate Response
        response = llm.invoke(messages)
        
        print("\nAI Message: ", response.content)

        # Add AI's Response to Message List
        messages.append(response)