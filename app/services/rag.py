"""
RAG (Retrieval-Augmented Generation) Engine.

This module implements the RAG pipeline that:
1. Loads the FAISS vector store
2. Retrieves relevant document chunks based on user queries
3. Uses an LLM to generate answers based on retrieved context

Why RAG?
- Combines the benefits of information retrieval and generation
- Provides answers grounded in the uploaded documents
- Reduces hallucination by providing source context
- More efficient than fine-tuning for domain-specific knowledge

GPU Optimization:
- Uses 4-bit quantization (BitsAndBytes) to fit models in 6GB VRAM
- Qwen2.5-1.5B-Instruct: Small, powerful model optimized for constrained GPUs
- Automatic device mapping ensures safe memory usage
"""

import logging
from typing import Dict, Optional

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
# Use sentence-transformers directly instead of langchain-huggingface wrapper
# This avoids dependency conflicts with older huggingface-hub
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for question-answering over financial documents.
    
    This class encapsulates the complete RAG pipeline:
    - Vector store loading and management
    - LLM initialization
    - RetrievalQA chain setup
    - Question answering with source attribution
    
    Why a class instead of functions?
    - Maintains state (vector store, LLM, chain)
    - Avoids reloading expensive resources on each query
    - Cleaner API for the FastAPI endpoints
    """
    
    def __init__(self):
        """
        Initialize the RAG engine.
        
        Loads the vector store and initializes the LLM and QA chain.
        This is called once when the application starts (singleton pattern).
        """
        self.vector_store: Optional[FAISS] = None
        self.qa_chain = None  # Will be a LangChain LCEL chain
        self.llm = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Initialize the vector store, embeddings, LLM, and QA chain.
        
        Why separate initialization?
        - Allows for lazy loading if needed
        - Can be called again to reload after new documents are added
        - Clear separation of setup logic
        """
        try:
            # Load embeddings (must match the ones used during ingestion)
            # Why reload embeddings?
            # - Must use the same model and configuration as ingestion
            # - Ensures compatibility with stored vectors
            # Determine device (CUDA if available, else CPU)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logger.info(f"Using GPU for embeddings: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("Using CPU for embeddings (GPU not available)")
            
            # Use SentenceTransformer directly to avoid langchain-huggingface dependency issues
            # Create a wrapper that works with LangChain's FAISS
            from langchain_core.embeddings import Embeddings
            
            class SentenceTransformerEmbeddings(Embeddings):
                """Wrapper to use SentenceTransformer with LangChain."""
                def __init__(self, model_name: str, device: str = "cpu"):
                    self.model = SentenceTransformer(model_name, device=device)
                
                def embed_documents(self, texts):
                    return self.model.encode(texts, normalize_embeddings=True).tolist()
                
                def embed_query(self, text: str):
                    return self.model.encode([text], normalize_embeddings=True)[0].tolist()
            
            embeddings = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model_name,
                device=device
            )
            
            # Load vector store
            # Why check existence?
            # - Prevents errors if no documents have been uploaded yet
            # - Provides better error messages to users
            vector_store_path = settings.vector_store_path
            if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
                logger.warning("Vector store not found. Upload documents first.")
                return
            
            logger.info(f"Loading vector store from: {vector_store_path}")
            self.vector_store = FAISS.load_local(
                str(vector_store_path),
                embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
            
            # Initialize LLM for GPU
            # Note: PyTorch 1.13.1 is not compatible with bitsandbytes quantization
            # GPT-2 (548MB) is small enough to fit in 6GB VRAM without quantization
            logger.info("Initializing LLM for GPU (without quantization - PyTorch 1.13.1 compatibility)...")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Will attempt to use CPU (slower).")
                device_map = "cpu"
                device_str = "cpu"
            else:
                device = torch.cuda.current_device()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
                device_map = "cuda"
                device_str = f"cuda:{device}"
            
            # Skip quantization - bitsandbytes doesn't work with PyTorch 1.13.1
            # GPT-2 is small enough (548MB) to fit without quantization
            use_quantization = False
            
            # Model ID optimized for 6GB VRAM and compatible with transformers 4.30.2
            # Qwen2.5 requires transformers 4.37+, so using alternative compatible model
            # Using microsoft/DialoGPT-medium as it's compatible with transformers 4.30.2
            # Alternative: Use a smaller GPT-2 based model
            model_id = "gpt2"  # Compatible with transformers 4.30.2, small enough for 6GB VRAM
            logger.info(f"Loading model: {model_id} (compatible with PyTorch 1.13.1)")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Load model without quantization (PyTorch 1.13.1 compatibility)
            # GPT-2 is small enough to fit in 6GB VRAM
            logger.info("Loading model without quantization (PyTorch 1.13.1 compatible)")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for better compatibility with PyTorch 1.13.1
                low_cpu_mem_usage=True
            )
            # Move model to device manually
            model = model.to(device_str)
            
            # Create pipeline
            # Why pipeline?
            # - Simplifies text generation
            # - Handles tokenization and generation in one interface
            # - Compatible with LangChain's HuggingFacePipeline
            # Set pad token for GPT-2 (it doesn't have one by default)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine device index for pipeline
            pipeline_device = 0 if device_str.startswith("cuda") else -1
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=pipeline_device,  # 0 for CUDA device, -1 for CPU
                max_length=1024,  # Increased to handle longer contexts (RAG can have large prompts)
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Wrap in LangChain pipeline
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info("LLM initialized successfully (without quantization - PyTorch 1.13.1 compatible)")
            
            # Create retriever
            # Why these retrieval parameters?
            # - search_kwargs: Controls how many chunks to retrieve
            # - k=4: Balance between context and token usage
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Create custom prompt template
            # Why custom prompt?
            # - Provides context to the LLM about the task
            # - Instructs it to use only provided context
            # - Improves answer quality and reduces hallucination
            # Prompt template optimized for GPT-2 style models
            prompt_template = """Based on the following financial document context, answer the question.

Context: {context}

Question: {question}

Answer based on the context above:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create RAG chain using LangChain 1.x LCEL pattern
            # Why LCEL?
            # - Modern LangChain 1.x approach
            # - More flexible and composable
            # - Better performance and error handling
            
            def format_docs(docs):
                """Format retrieved documents into a single string."""
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Create the chain: retrieve -> format -> prompt -> llm -> parse
            self.qa_chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG engine: {str(e)}")
            raise
    
    def ask_question(self, query: str) -> Dict[str, any]:
        """
        Answer a question using the RAG pipeline.
        
        This method:
        1. Retrieves relevant document chunks
        2. Generates an answer using the LLM
        3. Returns the answer with source documents
        
        Args:
            query: The user's question
            
        Returns:
            dict: Contains 'answer', 'source_documents', and metadata
            
        Raises:
            ValueError: If vector store is not initialized
            Exception: If query processing fails
        """
        if self.qa_chain is None:
            raise ValueError(
                "RAG engine not initialized. Please upload documents first."
            )
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Execute the QA chain
            # Why try-except?
            # - Handles API errors gracefully
            # - Provides meaningful error messages
            answer = self.qa_chain.invoke(query)
            
            # Retrieve source documents separately for attribution
            # Why retrieve separately?
            # - LCEL chain doesn't return sources by default
            # - We need to get them for citation purposes
            source_documents = self.retriever.get_relevant_documents(query)
            
            # Format source documents for response
            # Why format sources?
            # - Provides transparency about answer sources
            # - Allows users to verify information
            # - Important for financial documents (citations)
            sources = []
            for doc in source_documents:
                sources.append({
                    "content": doc.page_content[:200] + "...",  # Preview
                    "metadata": doc.metadata
                })
            
            response = {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
            
            logger.info(f"Query processed successfully. Found {len(sources)} sources.")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def reload(self) -> None:
        """
        Reload the vector store and QA chain.
        
        Useful after new documents are uploaded.
        Why this method?
        - Allows updating the knowledge base without restarting server
        - Called automatically after document upload
        """
        logger.info("Reloading RAG engine...")
        self._initialize()


# Global RAG engine instance
# Why singleton?
# - Avoids reloading expensive resources on each request
# - Maintains state across API calls
# - More efficient than creating new instance per query
rag_engine = RAGEngine()

