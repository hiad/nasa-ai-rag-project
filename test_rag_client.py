import os
from dotenv import load_dotenv
from rag_client import discover_chroma_backends

load_dotenv()

backends = discover_chroma_backends()
print(backends)