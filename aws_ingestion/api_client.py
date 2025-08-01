# aws_ingestion/api_client.py

# Coded By: Developer A (You)
# Branch: feature/api-client

import requests
from typing import List
from llama_index.core.schema import BaseNode

def send_nodes_to_api(nodes: List[BaseNode], api_url: str):
    """
    Formats node data and sends it to the GCP RAG API's /ingest endpoint.

    Args:
        nodes (List[BaseNode]): The list of nodes with text, metadata, and embeddings.
        api_url (str): The URL of the ingestion API endpoint.

    Returns:
        bool: True if the request was successful, False otherwise.
    """
    print(f"\n--- Preparing to Send {len(nodes)} Nodes to API ---")

    # 1. Format the data according to the API contract
    payload = []
    for node in nodes:
        if node.embedding is None:
            print(f"Warning: Skipping node {node.node_id} because it has no embedding.")
            continue
            
        payload.append({
            "id": node.node_id,
            "text": node.get_content(),
            "embedding": node.embedding,
            "metadata": node.metadata
        })
    
    if not payload:
        print("Error: No nodes with embeddings to send.")
        return False

    print(f"Formatted {len(payload)} nodes. Sending data to {api_url}...")

    # 2. Send the data via an HTTP POST request
    try:
        response = requests.post(api_url, json=payload, timeout=60) # timeout in seconds
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status() 

        print(f"Successfully sent data. API Response: {response.json()}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to send data to API. {e}")
        return False