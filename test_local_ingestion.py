# test_local_ingestion.py (Final Version for this feature)

# Coded By: Developer A (You)
# Branch: feature/local-ingestion-modules

import sys
from pathlib import Path

# This is a common pattern to make sure the script can find your modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from aws_ingestion.document_loader import load_financial_document
from aws_ingestion.text_processor import process_documents_into_nodes
from aws_ingestion.embedding_generator import generate_embeddings_for_nodes # <-- IMPORT THE NEW FUNCTION

def main():
    """
    Tests the full local ingestion pipeline: Load -> Process -> Embed.
    """
    print("--- Starting Full Local Ingestion Pipeline Test ---")

    sample_pdf_path = Path("data/raw/sample-report.pdf")

    try:
        # --- Step 1: Document Loading ---
        documents = load_financial_document(sample_pdf_path)
        if not documents:
            print("Stopping: No documents were loaded.")
            return

        # --- Step 2: Text Processing (Chunking) ---
        nodes = process_documents_into_nodes(documents)
        if not nodes:
            print("Stopping: No nodes were created from documents.")
            return

        # --- Step 3: Embedding Generation ---
        nodes_with_embeddings = generate_embeddings_for_nodes(nodes)
        if not nodes_with_embeddings:
            print("Stopping: Embedding generation failed.")
            return

        print("\n--- Final Verification ---")
        print(f"Total nodes processed: {len(nodes_with_embeddings)}")

        if nodes_with_embeddings:
            first_node = nodes_with_embeddings[0]
            print(f"Verification for the first node:")
            print(f"  - Has text: {len(first_node.text) > 0}")
            print(f"  - Has metadata: {'file_name' in first_node.metadata}")
            
            # The most important check for this step
            print(f"  - Has embedding: {first_node.embedding is not None}")
            if first_node.embedding:
                # The all-MiniLM-L6-v2 model produces embeddings of size 384
                print(f"  - Embedding dimension: {len(first_node.embedding)}")
                if len(first_node.embedding) == 384:
                    print("  - Verification successful: Embedding dimension is correct (384).")
                else:
                    print(f"  - Verification FAILED: Embedding dimension is {len(first_node.embedding)}, expected 384.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the file 'sample-report.pdf' exists in the 'data/raw/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- Full Local Ingestion Pipeline Test Finished ---")

if __name__ == "__main__":
    main()