import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
import hashlib

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

def search_db(collection, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> List[Dict]:
    """Helper function to query ChromaDB and return a flat list of results"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )
    
    if not results or not results.get("documents"):
        return []
        
    formatted_results = []
    # results format: {"documents": [[...]], "metadatas": [[...]], "ids": [[...]], "distances": [[...]]}
    for i in range(len(results["documents"][0])):
        formatted_results.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "id": results["ids"][0][i],
            "similarity": 1.0 - (results["distances"][0][i] if "distances" in results else 0)
        })
    return formatted_results

def retrieve_and_deduplicate(collection, query: str, top_k: int = 5, where_filter: Optional[Dict] = None) -> List[Dict]:
    """Retrieve relevant documents and remove duplicates based on content hash"""
    # Get more results than needed to account for deduplication
    raw_results = search_db(collection, query, n_results=top_k * 2, where_filter=where_filter)
    
    # Calculate similarities... results.append((review, similarity)) 
    # (Similarity already calculated in search_db)
    
    # 1. Sort by similarity (highest first)
    # ChromaDB already returns results sorted by distance (similarity)
    # raw_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    seen_content = set()
    unique_snippets = []
    
    for snippet in raw_results:
        # Normalize content for robust hashing
        content_hash = hashlib.md5(snippet['content'].strip().encode('utf-8')).hexdigest()
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_snippets.append(snippet)
            
        if len(unique_snippets) >= top_k:
            break
            
    return unique_snippets

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None,
                      deduplicate: bool = True) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering and deduplication"""

    where_filter = None
    if mission_filter and mission_filter.lower() != "all":
        where_filter = {"mission": mission_filter}
    
    if deduplicate:
        snippets = retrieve_and_deduplicate(collection, query, top_k=n_results, where_filter=where_filter)
        # Reformat back to the expected structure for compatibility
        if not snippets:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
        
        return {
            "documents": [[s["content"] for s in snippets]],
            "metadatas": [[s["metadata"] for s in snippets]],
            "ids": [[s["id"] for s in snippets]]
        }
    else:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
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