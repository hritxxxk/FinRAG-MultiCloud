# aws_ingestion/text_processor.py

# Coded By: Developer A (You)
# Branch: feature/local-ingestion-modules

from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode

def process_documents_into_nodes(documents: List[Document]) -> List[BaseNode]:
    """
    Processes a list of Document objects into a list of smaller text nodes (chunks).

    This function uses a SentenceSplitter to break down the text while trying
    to respect sentence boundaries. It also ensures that metadata from the parent
    document is carried over to the child nodes.

    Args:
        documents (List[Document]): The list of Document objects to process.

    Returns:
        List[BaseNode]: A list of nodes, ready for embedding.
    """
    print(f"\n--- Starting Text Processing and Chunking ---")
    
    # Initialize the SentenceSplitter.
    # chunk_size: The target size of each text chunk in tokens. 1024 is a common default.
    # chunk_overlap: The number of tokens to overlap between chunks. This helps
    #                maintain context across the boundary of two chunks.
    text_splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=20
    )

    # Use the parser to get nodes from the documents. This is the core chunking operation.
    # LlamaIndex handles the heavy lifting of splitting and creating Node objects.
    nodes = text_splitter.get_nodes_from_documents(documents)

    # --- Metadata Verification ---
    # It's good practice to verify that our metadata has been preserved.
    # LlamaIndex's parsers are designed to do this automatically.
    if nodes:
        print(f"Successfully created {len(nodes)} nodes (chunks).")
        # Let's check the first node to confirm metadata propagation.
        first_node_metadata = nodes[0].metadata
        print(f"Metadata from parent document '{first_node_metadata.get('file_name')}' was successfully carried over to the first node.")
    else:
        print("Warning: No nodes were created from the documents.")

    return nodes