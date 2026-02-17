from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt with context
    system_prompt = f"""You are a specialized NASA mission assistant. 
DIRECTIVE: Provide a concise, factual answer based ONLY on the retrieved context below. 
If the information is not in the context, clearly state: "I'm sorry, I don't have that specific information in my mission records."
Avoid general greetings or filler text to ensure high relevancy to the specific question asked.

Context:
{context}
"""
    
    # Assembly messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in conversation_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)
    
    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )
    
    return response.choices[0].message.content

    pass