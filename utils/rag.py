"""
RAG (Retrieval-Augmented Generation) module.
Handles PDF loading, text chunking, vector storage, and document retrieval.
"""

import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
try:
    # LangChain >= 0.2+ moved splitters to a dedicated package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    # Fallback for older LangChain versions.
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K
from models.embeddings import EmbeddingFunction


def load_pdf(pdf_file) -> list:
    """
    Load and parse a PDF file.

    Args:
        pdf_file: Uploaded PDF file object (from Streamlit)

    Returns:
        List of document objects containing page content and metadata
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        # Load the PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Clean up temporary file
        os.unlink(tmp_path)

        if not documents:
            raise ValueError("No content found in the PDF file.")

        return documents

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")


def chunk_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of document objects
        chunk_size: Size of each chunk (default from config)
        chunk_overlap: Overlap between chunks (default from config)

    Returns:
        List of chunked document objects
    """
    try:
        chunk_size = chunk_size or CHUNK_SIZE
        chunk_overlap = chunk_overlap or CHUNK_OVERLAP

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No chunks created from documents.")

        return chunks

    except Exception as e:
        raise RuntimeError(f"Failed to chunk documents: {str(e)}")


def create_vector_store(documents: list) -> FAISS:
    """
    Create a FAISS vector store from documents.

    Args:
        documents: List of document objects (can be raw or chunked)

    Returns:
        FAISS vector store instance
    """
    try:
        # Chunk documents if they haven't been chunked yet
        chunks = chunk_documents(documents)

        # Create embedding function
        embedding_function = EmbeddingFunction()

        # Create vector store
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_function
        )

        return vector_store

    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {str(e)}")


def search_documents(
    vector_store: FAISS,
    query: str,
    k: int = None
) -> list[dict]:
    """
    Search the vector store for relevant documents.

    Args:
        vector_store: FAISS vector store instance
        query: Search query string
        k: Number of results to return (default from config)

    Returns:
        List of dictionaries containing content and metadata
    """
    try:
        k = k or RETRIEVAL_K

        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(query, k=k)

        processed_results = []
        for doc, score in results:
            processed_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "source": doc.metadata.get("source", "Unknown")
            })

        return processed_results

    except Exception as e:
        raise RuntimeError(f"Failed to search documents: {str(e)}")


def format_context(search_results: list[dict]) -> str:
    """
    Format search results into a context string for the LLM.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Formatted context string
    """
    if not search_results:
        return ""

    context_parts = []
    for i, result in enumerate(search_results, 1):
        page_info = ""
        if "page" in result.get("metadata", {}):
            page_info = f" (Page {result['metadata']['page'] + 1})"

        context_parts.append(
            f"[Source {i}{page_info}]:\n{result['content']}"
        )

    return "\n\n---\n\n".join(context_parts)


def has_relevant_context(search_results: list[dict], threshold: float = 1.5) -> bool:
    """
    Determine if the search results contain relevant context.
    Lower scores indicate better matches in FAISS.

    Args:
        search_results: List of search result dictionaries
        threshold: Maximum score to consider relevant (lower is better)

    Returns:
        True if results are relevant, False otherwise
    """
    if not search_results:
        return False

    # Check if at least one result has a good score
    best_score = min(result["score"] for result in search_results)
    return best_score < threshold


def process_pdf_for_rag(pdf_file) -> tuple[FAISS, int]:
    """
    Complete pipeline to process a PDF file for RAG.

    Args:
        pdf_file: Uploaded PDF file object

    Returns:
        Tuple of (vector_store, number_of_chunks)
    """
    try:
        # Load PDF
        documents = load_pdf(pdf_file)

        # Create vector store (includes chunking)
        vector_store = create_vector_store(documents)

        # Get chunk count
        chunks = chunk_documents(documents)
        chunk_count = len(chunks)

        return vector_store, chunk_count

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {str(e)}")
