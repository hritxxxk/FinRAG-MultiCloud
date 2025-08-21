# gcp_rag_api/main.py (Updated)

# Coded By: Developer B
# Branch: feature/gcp-api-skeleton

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any

# Import our new ChromaDB client
from .vector_db_client import ChromaDBClient, COLLECTION_NAME

# --- App Initialization & Global Objects ---
# Create the FastAPI app instance
app = FastAPI(
    title="FinRAG GCP API",
    version="0.1.0",
    description="API for the RAG-powered Smart Financial Research Assistant."
)

# Create a single, global instance of our ChromaDB client.
# This object will be shared across all API requests.
db_client = ChromaDBClient()

# --- API Events ---
# Use a startup event to ensure the collection is ready when the API starts.
@app.on_event("startup")
def startup_event():
    """
    Actions to perform when the API server starts.
    """
    print("API is starting up...")
    db_client.get_or_create_collection(name=COLLECTION_NAME)

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def read_root():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {
        "message": "FinRAG GCP API is running!",
        "database_status": "connected",
        "collection_name": db_client.collection.name if db_client.collection else "N/A",
        "document_count": db_client.count()
    }

@app.post("/ingest", tags=["Data Ingestion"])
def ingest_data(data: List[Dict[str, Any]]):
    """
    Receives processed data and stores it in the ChromaDB vector store.
    
    The expected format for each item in the list is a dictionary with keys:
    'id', 'text', 'embedding', and 'metadata'.
    """
    if not data:
        raise HTTPException(status_code=400, detail="No data provided for ingestion.")

    try:
        # Use our client to add the data to the database.
        db_client.add_nodes(data)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {len(data)} nodes into collection '{db_client.collection.name}'.",
            "current_document_count": db_client.count()
        }
    except Exception as e:
        # If anything goes wrong during ingestion, return a detailed error.
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during data ingestion: {e}")
    
    # gcp_rag_api/main.py (add these imports and the new endpoint)

# ... (keep existing imports)
from fastapi import FastAPI, HTTPException, Query
# ...

# Import the new function from rag_core
from .rag_core import retrieve_relevant_chunks

# ... (keep all the existing app, db_client, and endpoint code) ...


# --- NEW QUERY ENDPOINT ---
@app.get("/query", tags=["RAG Query"])
def perform_query(
    query: str = Query(..., min_length=3, description="The user's question to the RAG system."),
    top_k: int = Query(3, ge=1, le=10, description="The number of relevant chunks to retrieve.")
):
    """
    Receives a user query, retrieves relevant context from the vector DB,
    and returns it. This is the "Retrieval" part of RAG.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        print(f"Performing query for: '{query}' with top_k={top_k}")
        
        # Use our core retrieval function
        retrieved_context = retrieve_relevant_chunks(
            query=query, 
            db_client=db_client, 
            n_results=top_k
        )
        
        return {
            "status": "success",
            "query": query,
            "retrieved_context": retrieved_context
        }
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during query: {e}")