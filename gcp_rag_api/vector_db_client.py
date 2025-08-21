# gcp_rag_api/vector_db_client.py

# Coded By: Developer B
# Branch: feature/gcp-api-skeleton

import chromadb
from chromadb.types import Collection
from typing import List, Dict, Any

# Define a path for our local, file-based ChromaDB instance.
# This will create a 'chroma_db' folder in the project's root directory.
DB_PATH = "./chroma_db"
COLLECTION_NAME = "financial_documents"

class ChromaDBClient:
    """
    A client to manage interactions with a local ChromaDB vector database.
    """
    def __init__(self, path: str = DB_PATH):
        # Initialize the ChromaDB client. 'PersistentClient' saves the data to disk.
        self.client = chromadb.PersistentClient(path=path)
        self.collection = None
        print(f"ChromaDB client initialized. Database path: {path}")

    def get_or_create_collection(self, name: str = COLLECTION_NAME) -> Collection:
        """
        Retrieves an existing collection or creates a new one if it doesn't exist.
        This is an idempotent operation, which is safe to call multiple times.
        """
        print(f"Attempting to get or create collection: '{name}'")
        self.collection = self.client.get_or_create_collection(name=name)
        print(f"Collection '{self.collection.name}' is ready.")
        return self.collection

    def add_nodes(self, nodes_data: List[Dict[str, Any]]) -> None:
        """
        Adds processed nodes (chunks with embeddings) to the ChromaDB collection.

        Args:
            nodes_data: A list of dictionaries, where each dictionary represents
                        a node and is expected to have 'id', 'text', 'embedding',
                        and 'metadata' keys.
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call get_or_create_collection() first.")

        if not nodes_data:
            print("No nodes to add.")
            return

        # ChromaDB expects separate lists for ids, documents, metadatas, and embeddings.
        # We need to transform the data from Developer A's format into this format.
        ids = [node['id'] for node in nodes_data]
        documents = [node['text'] for node in nodes_data]
        embeddings = [node['embedding'] for node in nodes_data]
        metadatas = [node['metadata'] for node in nodes_data]

        print(f"Adding {len(ids)} documents to collection '{self.collection.name}'...")
        
        # The core operation to add data to the vector store.
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("Successfully added documents to ChromaDB.")

    def count(self) -> int:
        """Returns the number of items in the collection."""
        if not self.collection:
            return 0
        return self.collection.count()
    


    def query_collection(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Queries the collection to find the most similar documents to a given embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the user's query.
            n_results (int): The number of top similar documents to return.

        Returns:
            List[Dict[str, Any]]: A list of the most relevant document chunks.
        """
        if not self.collection:
            raise ValueError("Collection not initialized.")

        print(f"Querying collection '{self.collection.name}' for {n_results} results...")
        
        results = self.collection.query(
            query_embeddings=[query_embedding], # Note: ChromaDB expects a list of embeddings
            n_results=n_results
        )
        
        return results