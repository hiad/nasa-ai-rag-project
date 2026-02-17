import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    chroma_dirs = [d.name for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma_db")]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        try:
            # Initialize database client with directory path
            client = chromadb.PersistentClient(path=chroma_dir)
            
            # Retrieve list of available collections
            collections = client.list_collections()
            
            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key
                key = f"{chroma_dir}/{collection.name}"
                
                info = {
                    "directory": chroma_dir,
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir} - {collection.name}",
                    "document_count": collection.count()
                }
                backends[key] = info
        
        except Exception as e:
            # Create fallback entry for inaccessible directories
            key = f"{chroma_dir}/inaccessible"
            info = {
                "directory": chroma_dir,
                "collection_name": "inaccessible",
                "display_name": f"{chroma_dir} - inaccessible - {str(e)[:50]}",
                "document_count": 0
            }
            backends[key] = info
            continue
            
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    import os
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None, False, "OPENAI_API_KEY not found in environment"
            
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        )
        return collection, True, None
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    where_filter = None
    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}
    # TODO: Execute database query with the following parameters:
    # TODO: Pass search query in the required format
    # TODO: Set maximum number of results to return
    # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    # TODO: Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    context_parts = ["NASA Mission Data Context:\n"]
    for i, (document, metadata) in enumerate(zip(documents, metadatas)):
        mission = metadata.get("mission", "Unknown Mission").replace("_", " ").title()
        source = metadata.get("source", "Unknown Source")
        category = metadata.get("document_category", "Unknown Category").replace("_", " ").title()
        
        source_header = f"Source {i+1}: {mission} | {source} | {category}"
        context_parts.append(f"{source_header}\n{document}\n")
    
    return "\n".join(context_parts)