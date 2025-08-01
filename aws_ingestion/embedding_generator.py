# aws_ingestion/embedding_generator.py

# Coded By: Developer A (You)
# Branch: feature/local-ingestion-modules

from typing import List
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def generate_embeddings_for_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """
    Generates embeddings for a list of nodes using a local HuggingFace model.

    This function takes the text from each node, converts it into a vector
    embedding, and attaches that embedding back to the node object.

    Args:
        nodes (List[BaseNode]): The list of nodes to be embedded.

    Returns:
        List[BaseNode]: The same list of nodes, now enriched with embeddings.
    """
    print(f"\n--- Starting Embedding Generation ---")
    
    # Initialize the embedding model.
    # "sentence-transformers/all-MiniLM-L6-v2" is a popular, efficient model.
    # The first time this runs, it will download the model from Hugging Face and
    # cache it locally. This may take a few minutes and requires internet.
    # Subsequent runs will be much faster.
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Embedding model '{embed_model.model_name}' initialized.")
    
    # Iterate through the nodes and generate embeddings.
    # The model will process the text of each node.
    for node in nodes:
        # The embedding is a list of floating-point numbers (a vector).
        node.embedding = embed_model.get_text_embedding(node.get_content())

    print(f"Successfully generated embeddings for {len(nodes)} nodes.")
    
    return nodes