# gcp_rag_api/rag_core.py

# Coded By: Developer B
# Branch: feature/query-endpoint

from typing import List, Dict, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .vector_db_client import ChromaDBClient

# --- Global Initialization ---
# Initialize the embedding model once to avoid reloading it on every API call.
# This ensures efficiency.
print("Initializing embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model initialized successfully.")


def retrieve_relevant_chunks(query: str, db_client: ChromaDBClient, n_results: int = 5) -> List[Dict[str, Any]]:
    """
    The core "Retrieval" function of RAG.
    1. Takes a user's text query.
    2. Generates an embedding for the query.
    3. Queries the vector database to find the most relevant text chunks.
    
    Args:
        query (str): The user's question.
        db_client (ChromaDBClient): An instance of our database client.
        n_results (int): The number of chunks to retrieve.

    Returns:
        List[Dict[str, Any]]: The retrieved context from the database.
    """
    # 1. Generate embedding for the user's query
    print(f"Generating embedding for query: '{query}'")
    query_embedding = embed_model.get_text_embedding(query)
    
    # 2. Query the vector database
    retrieved_results = db_client.query_collection(
        query_embedding=query_embedding,
        n_results=n_results
    )
    
    return retrieved_results