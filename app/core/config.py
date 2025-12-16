"""
Configuration management using Pydantic Settings.

This module centralizes all configuration settings, ensuring type safety
and easy access to environment variables. Using Pydantic Settings provides
automatic validation and allows for default values.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Why Pydantic Settings?
    - Automatic type validation
    - Environment variable parsing
    - Default value support
    - Type hints for IDE support
    """
    
    # HuggingFace API Configuration
    # Note: Token is required for downloading models from HuggingFace Hub
    # Get a free token at: https://huggingface.co/settings/tokens
    huggingfacehub_api_token: str = ""
    huggingface_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Embedding Model Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Vector Store Configuration
    vector_store_path: Path = Path("./vector_store")
    
    # Text Splitting Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # File Upload Configuration
    upload_dir: Path = Path("./uploads")
    max_file_size_mb: int = 50
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow reading from .env file with uppercase or lowercase keys
        env_prefix="",
        # Allow extra fields to be ignored (for backward compatibility)
        extra="ignore",
    )
    
    def __init__(self, **kwargs):
        """
        Initialize settings and create necessary directories.
        
        Why create directories here?
        - Ensures upload and vector store directories exist before use
        - Prevents runtime errors from missing directories
        """
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate HuggingFace token
        # Why check here?
        # - Provides helpful error message if token is missing
        # - Models require authentication to download from HuggingFace Hub
        if not self.huggingfacehub_api_token or self.huggingfacehub_api_token.strip() == "":
            import warnings
            warnings.warn(
                "HUGGINGFACEHUB_API_TOKEN is not set. "
                "You may encounter errors when downloading models. "
                "Get a free token at: https://huggingface.co/settings/tokens "
                "and add it to your .env file."
            )


# Global settings instance
# Why a global instance?
# - Single source of truth for configuration
# - Easy to import and use across the application
# - Can be overridden in tests if needed
settings = Settings()

