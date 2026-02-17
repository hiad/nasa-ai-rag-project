import os
from dotenv import load_dotenv
from llm_client import generate_response

# 1. Load your API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in .env file.")
else:
    # 2. Run the test
    print("🚀 Sending request to OpenAI...")
    response = generate_response(
        openai_key=api_key, 
        user_message="What was Apollo 11?", 
        context="", 
        conversation_history=[]
    )

    print("\n--- Response ---")
    print(response)