"""
FastAPI application entry point.

This module defines the REST API endpoints for:
1. Uploading and processing PDF documents
2. Querying the RAG system with questions

Why FastAPI?
- Modern, fast Python web framework
- Automatic API documentation (Swagger/OpenAPI)
- Built-in validation and type checking
- Excellent async support
"""

import logging
import shutil
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.core.config import settings
from app.services import ingest, rag

# Configure logging
# Why configure logging here?
# - Sets up logging for the entire application
# - Ensures consistent log format
# - Useful for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
# Why these metadata?
# - Provides API documentation
# - Useful for portfolio demonstration
app = FastAPI(
    title="Financial Research RAG Assistant",
    description="A production-ready RAG system for querying financial documents",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc alternative
)


# Request/Response Models
# Why Pydantic models?
# - Automatic validation
# - Type safety
# - Auto-generated API documentation
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What was the company's revenue in the last fiscal year?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list
    num_sources: int


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    message: str
    vector_store_path: str
    status: str


# Health check endpoint
# Why health check?
# - Useful for monitoring and deployment
# - Verifies system is running
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Financial Research RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns the status of the vector store and RAG engine.
    """
    vector_store_info = ingest.get_vector_store_info()
    return {
        "status": "healthy",
        "vector_store": vector_store_info,
        "rag_engine_initialized": rag.rag_engine.qa_chain is not None
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload and process a PDF document.
    
    This endpoint:
    1. Validates the uploaded file (must be PDF)
    2. Saves it temporarily
    3. Processes it through the ingestion pipeline
    4. Reloads the RAG engine with new documents
    
    Args:
        file: The PDF file to upload
        
    Returns:
        UploadResponse: Confirmation with vector store path
        
    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Validate file type
    # Why validate here?
    # - Early error detection
    # - Better user experience
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Validate file size
    # Why check size?
    # - Prevents memory issues
    # - Protects against DoS attacks
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum of {settings.max_file_size_mb}MB"
        )
    
    # Save uploaded file temporarily
    # Why temporary?
    # - Processes file and then cleans up
    # - Avoids cluttering upload directory
    upload_path = settings.upload_dir / file.filename
    
    try:
        logger.info(f"Uploading file: {file.filename}")
        
        # Save file
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {upload_path}")
        
        # Process document
        # Why try-except?
        # - Handles processing errors gracefully
        # - Cleans up file even if processing fails
        try:
            vector_store_path = ingest.process_document(str(upload_path))
            logger.info(f"Document processed successfully: {vector_store_path}")
            
            # Reload RAG engine with new documents
            # Why reload?
            # - Makes new documents immediately available for queries
            # - Updates the knowledge base without restarting server
            rag.rag_engine.reload()
            
            return UploadResponse(
                message=f"Document '{file.filename}' processed successfully",
                vector_store_path=vector_store_path,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )
        
        finally:
            # Clean up uploaded file
            # Why cleanup?
            # - Saves disk space
            # - Only vector store is needed after processing
            if upload_path.exists():
                upload_path.unlink()
                logger.info(f"Temporary file deleted: {upload_path}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the RAG system with a question.
    
    This endpoint:
    1. Validates the question
    2. Uses the RAG engine to retrieve relevant context
    3. Generates an answer using the LLM
    4. Returns answer with source documents
    
    Args:
        request: QueryRequest containing the question
        
    Returns:
        QueryResponse: Answer with source documents
        
    Raises:
        HTTPException: If RAG engine not initialized or query fails
    """
    if rag.rag_engine.qa_chain is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents uploaded yet. Please upload a PDF first."
        )
    
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        result = rag.rag_engine.ask_question(request.question)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"]
        )
    
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


# Startup event
# Why startup event?
# - Initializes RAG engine when server starts
# - Ensures system is ready before accepting requests
@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on application startup."""
    logger.info("Starting Financial Research RAG Assistant...")
    
    # Check GPU availability and display information
    # Why check GPU on startup?
    # - Confirms RTX 4050 is detected and active
    # - Provides memory information for debugging
    # - Helps verify GPU configuration is correct
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✅ GPU Detected: {device_name}")
        logger.info(f"✅ GPU Memory: {device_memory:.2f} GB")
        logger.info(f"✅ CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA not available. GPU acceleration disabled.")
        logger.warning("⚠️  System will fall back to CPU (slower performance).")
    
    logger.info(f"Vector store path: {settings.vector_store_path}")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info(f"LLM model: Qwen/Qwen2.5-1.5B-Instruct (4-bit quantized)")
    
    # Try to initialize RAG engine if vector store exists
    if ingest.get_vector_store_info()["exists"]:
        logger.info("Vector store found. Initializing RAG engine...")
        try:
            rag.rag_engine._initialize()
        except Exception as e:
            logger.warning(f"Could not initialize RAG engine: {str(e)}")
            logger.warning("Please upload documents to initialize the system.")
    else:
        logger.info("No vector store found. Please upload documents first.")


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # Why uvicorn?
    # - ASGI server for FastAPI
    # - Supports async operations
    # - Production-ready with proper configuration
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (development)
    )

