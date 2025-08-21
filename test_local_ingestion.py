# test_local_ingestion.py (Updated to include API call)

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from aws_ingestion.document_loader import load_financial_document
from aws_ingestion.text_processor import process_documents_into_nodes
from aws_ingestion.embedding_generator import generate_embeddings_for_nodes
from aws_ingestion.api_client import send_nodes_to_api # <-- IMPORT THE NEW FUNCTION

def main():
    """
    Tests the full local ingestion pipeline and sends data to the local RAG API.
    """
    print("--- Starting Full Local Ingestion & API Send Test ---")

    # The URL of your collaborator's locally running API
    GCP_API_URL = "http://127.0.0.1:8000/ingest"
    sample_pdf_path = Path("data/raw/sample-report.pdf")

    try:
        # Step 1: Load
        documents = load_financial_document(sample_pdf_path)
        # Step 2: Chunk
        nodes = process_documents_into_nodes(documents)
        # Step 3: Embed
        nodes_with_embeddings = generate_embeddings_for_nodes(nodes)

        if not nodes_with_embeddings:
            print("Stopping: Pipeline did not produce nodes with embeddings.")
            return

        # --- Step 4: Send Data to Local API ---
        success = send_nodes_to_api(nodes_with_embeddings, GCP_API_URL)

        if success:
            print("\nVerification successful: Data was sent to the API.")
        else:
            print("\nVerification FAILED: Data could not be sent. Is the GCP API server running?")

    except Exception as e:
        print(f"An unexpected error occurred in the pipeline: {e}")

    print("\n--- Full Local Ingestion & API Send Test Finished ---")

if __name__ == "__main__":
    main()