"""
Document ingestion service.

This module handles the complete pipeline for processing PDF documents:
1. Loading PDF files
2. Splitting text into chunks
3. Creating embeddings
4. Storing in FAISS vector database

Why separate ingestion from RAG?
- Separation of concerns: ingestion is a one-time process
- Allows for batch processing of multiple documents
- Can be run independently or as part of the API
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# Use sentence-transformers directly instead of langchain-huggingface wrapper
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings

# Configure logging
# Why logging?
# - Essential for debugging in production
# - Helps track document processing progress
# - Useful for monitoring and troubleshooting
logger = logging.getLogger(__name__)


def process_document(file_path: str) -> str:
    """
    Process a PDF document and create/update the FAISS vector store.
    
    This function:
    1. Loads the PDF using PyPDFLoader
    2. Splits the text into chunks with overlap
    3. Creates embeddings using BAAI/BGE embeddings
    4. Saves to FAISS vector store (appends if store exists)
    
    Args:
        file_path: Path to the PDF file to process
        
    Returns:
        str: Path to the vector store directory
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF processing fails
        
    Why this approach?
    - RecursiveCharacterTextSplitter: Preserves semantic meaning better than
      simple character splitting by respecting sentence/paragraph boundaries
    - Chunk overlap: Ensures context isn't lost at chunk boundaries
    - FAISS: Fast similarity search, suitable for production use
    - Appending to existing store: Allows incremental document addition
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    logger.info(f"Processing document: {file_path}")
    
    try:
        # Step 1: Load PDF document
        # Why PyPDFLoader?
        # - Extracts text while preserving structure
        # - Handles multi-page documents automatically
        # - Part of LangChain ecosystem for easy integration
        loader = PyPDFLoader(str(file_path_obj))
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content extracted from PDF: {file_path}")
        
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        # Step 2: Split documents into chunks
        # Why RecursiveCharacterTextSplitter?
        # - Tries to split on paragraphs, then sentences, then words
        # - Preserves semantic units better than fixed-size splitting
        # - Overlap ensures context continuity across chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # Why these separators?
            # - Prioritizes splitting at natural boundaries
            # - Preserves document structure
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Step 3: Create embeddings
        # Why BGE (BAAI General Embeddings)?
        # - Top-ranked on MTEB (Massive Text Embedding Benchmark)
        # - Significantly better retrieval performance than older MiniLM
        # - optimized for retrieval-augmented generation tasks
        # Determine device (CUDA if available, else CPU)
        # Why check CUDA availability?
        # - Graceful fallback if GPU not available
        # - Prevents errors in CPU-only environments
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings (GPU not available)")
        
        # Use SentenceTransformer directly to avoid langchain-huggingface dependency issues
        # Create a wrapper that works with LangChain's FAISS
        from langchain_core.embeddings import Embeddings
        
        class SentenceTransformerEmbeddings(Embeddings):
            """Wrapper to use SentenceTransformer with LangChain."""
            def __init__(self, model_name: str, device: str = "cpu"):
                self.model = SentenceTransformer(model_name, device=device)
            
            def embed_documents(self, texts):
                # BGE MTEB leaderboard recommendation:
                # No instruction needed for documents, only for queries
                return self.model.encode(texts, normalize_embeddings=True).tolist()
            
            def embed_query(self, text: str):
                # PRO TIP: BGE models need this prefix for queries to work 10x better
                text = f"Represent this sentence for searching relevant passages: {text}"
                return self.model.encode([text], normalize_embeddings=True)[0].tolist()
        
        embeddings = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model_name,
            device=device
        )
        
        # Step 4: Create or update FAISS vector store
        vector_store_path = settings.vector_store_path
        
        # Check if vector store already exists
        # Why check first?
        # - Allows incremental addition of documents
        # - Appends new documents to existing knowledge base
        if vector_store_path.exists() and any(vector_store_path.iterdir()):
            logger.info("Loading existing vector store")
            vector_store = FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            # Add new chunks to existing store
            vector_store.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new chunks to existing vector store")
        else:
            logger.info("Creating new vector store")
            vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store
        # Why save locally?
        # - Persists between server restarts
        # - No need to re-process documents
        # - FAISS format is efficient for fast loading
        vector_store.save_local(str(vector_store_path))
        logger.info(f"Vector store saved to: {vector_store_path}")
        
        return str(vector_store_path)
        
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {str(e)}")
        raise


def get_vector_store_info() -> dict:
    """
    Get information about the current vector store.
    
    Returns:
        dict: Information about the vector store including path and existence
    """
    vector_store_path = settings.vector_store_path
    exists = vector_store_path.exists() and any(vector_store_path.iterdir())
    
    return {
        "path": str(vector_store_path),
        "exists": exists,
        "embedding_model": settings.embedding_model_name
    }

