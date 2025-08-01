# aws_ingestion/document_loader.py

from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_index.readers.file import PDFReader

def load_financial_document(file_path: Path) -> List[Document]:
    """
    Loads a financial document from a given PDF file path and enriches it
    with metadata.

    Args:
        file_path (Path): The path to the PDF file.

    Returns:
        List[Document]: A list of Document objects, where each object
                        represents a page or the entire document content.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Initialize the PDF reader. PyPDFReader is a robust choice from LlamaIndex.
    loader = PDFReader()

    # The load_data method reads the PDF and returns a list of Document objects.
    # By default, PyPDFReader often creates one Document per page.
    docs = loader.load_data(file=file_path)

    # --- Metadata Enrichment ---
    # This is a critical step for our RAG system. We add the source file name
    # to the metadata of each document chunk. This helps in tracing the source
    # of information during retrieval.
    for doc in docs:
        doc.metadata["file_name"] = file_path.name
        # The loader already adds 'page_label' to metadata, which is great.

    print(f"Successfully loaded {len(docs)} documents/pages from {file_path.name}")
    
    return docs