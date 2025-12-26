"""
Configuration management.

This module centralizes all configuration settings.
Simplified to standard Python class to avoid Pydantic v2/Windows path conflicts.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    Application settings.
    """
    
    def __init__(self):
        # HuggingFace API Configuration
        self.huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        
        # Embedding Model Configuration
        self.embedding_model_name = "BAAI/bge-small-en-v1.5"
        
        # Vector Store Configuration
        self.vector_store_path = Path("./vector_store")
        
        # Model Configuration
        self.model_path = Path("./models/Phi-3-mini-4k-instruct-q4.gguf")
        self.model_repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
        self.model_filename = "Phi-3-mini-4k-instruct-q4.gguf"
        
        # Text Splitting Configuration
        # Text Splitting Configuration
        # Increased to 1200 to capture full paragraphs for "Risk Factors" / "Management Discussion"
        self.chunk_size = 1200
        self.chunk_overlap = 200
        
        # File Upload Configuration
        self.upload_dir = Path("./uploads")
        self.max_file_size_mb = 50
        
        # Create directories
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.huggingfacehub_api_token:
            print("WARNING: HUGGINGFACEHUB_API_TOKEN is not set.")

settings = Settings()
