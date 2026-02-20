from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo",
                     max_history: int = 10) -> str:
    """
    Generate response using OpenAI with context and managed history.
    
    Args:
        openai_key: API key for OpenAI
        user_message: The latest query from the user
        context: Retrieved context for the query
        conversation_history: List of previous messages in the format {"role": str, "content": str}
        model: OpenAI model to use
        max_history: Maximum number of previous message turns to include (default 10)
    """

    # Define system prompt with context
    system_prompt = f"""You are a specialized NASA mission expert assistant. A conversation history will be provided followed by the retrieved context. 

DIRECTIVE:
1. Provide a concise, factual answer based ONLY on the retrieved context below. 
2. Use short inline citations like "[Source 1]" or "[Source 2]" after key claims to show exactly where the information came from.
3. Only use information provided in the following context. Do not add facts or information not present in the context.
4. If the information is not in the context, clearly state: "I'm sorry, I don't have that specific information in my mission records."
5. Avoid general greetings or filler text.

Context:
{context}
"""
    
    # Initialize message list with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # Prune history to include only necessary context (last max_history turns)
    # Each 'turn' is typically 2 messages (user + assistant)
    pruned_history = conversation_history[-max_history:] if max_history > 0 else []
    
    # Add chat history with role verification
    valid_roles = {"user", "assistant"}
    for msg in pruned_history:
        if msg.get("role") in valid_roles:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)
    
    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=700,
        temperature=0.7
    )
    
    return response.choices[0].message.content