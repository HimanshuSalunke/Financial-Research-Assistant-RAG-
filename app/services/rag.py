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

GPU Optimization (Best in the World for local RAG):
- Uses Llama.cpp with GGUF quantization (Q4_K_M)
- Microsoft Phi-3-mini: State-of-the-art 3.8B model that rivals GPT-3.5
- Bypasses PyTorch memory overhead by running directly on CUDA runtime
"""

import logging
from typing import Dict, Optional

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
# Use sentence-transformers directly instead of langchain-huggingface wrapper
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG Engine for question-answering over financial documents.
    """
    
    def __init__(self):
        """
        Initialize the RAG engine.
        """
        self.vector_store: Optional[FAISS] = None
        self.qa_chain = None
        self.llm = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Initialize the vector store, embeddings, LLM (LlamaCpp), and QA chain.
        """
        try:
            # Load embeddings (using BGE optimized prefix)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logger.info(f"Using GPU for embeddings (SentenceTransformers): {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("Using CPU for embeddings (GPU not available)")
            
            # Use SentenceTransformer directly to avoid langchain-huggingface dependency issues
            from langchain_core.embeddings import Embeddings
            
            class SentenceTransformerEmbeddings(Embeddings):
                """Wrapper to use SentenceTransformer with LangChain."""
                def __init__(self, model_name: str, device: str = "cpu"):
                    self.model = SentenceTransformer(model_name, device=device)
                
                def embed_documents(self, texts):
                    return self.model.encode(texts, normalize_embeddings=True).tolist()
                
                def embed_query(self, text: str):
                    # BGE instruction for better retrieval
                    text = f"Represent this sentence for searching relevant passages: {text}"
                    return self.model.encode([text], normalize_embeddings=True)[0].tolist()
            
            embeddings = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model_name,
                device=device
            )
            
            # Load vector store
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
            
            # Initialize LlamaCpp (GGUF Model)
            logger.info(f"Initializing LlamaCpp with model: {settings.model_path}")
            
            if not settings.model_path.exists():
                logger.error(f"Model file not found at {settings.model_path}. Please run scripts/download_models.py")
                return

            # Check for GPU offloading
            n_gpu_layers = -1 if device == "cuda" else 0
            if n_gpu_layers == -1:
                logger.info("Offloading ALL layers to GPU (CUDA)")
            
            self.llm = LlamaCpp(
                model_path=str(settings.model_path),
                n_gpu_layers=n_gpu_layers,
                n_batch=512,
                n_ctx=4096,  # Phi-3 supports 4k context
                f16_kv=True,  # Use FP16 for KV cache
                verbose=False,
                temperature=0.1,  # Low temperature for factual RAG
                max_tokens=1024,
                stop=["<|end|>", "<|user|>", "<|assistant|>"]
            )
            
            logger.info("LLM initialized successfully via LlamaCpp")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Phi-3 Instruct Prompt Template
            # Best practice: strict system instructions + context wrapper
            prompt_template = """<|system|>
You are a financial research assistant. Answer the question specifically using ONLY the provided context.
If the answer is not in the context, say "I cannot find the answer in the documents."
Context:
{context}
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
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
        """Answer a question using the RAG pipeline."""
        if self.qa_chain is None:
            raise ValueError("RAG engine not initialized. Please upload documents first.")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            answer = self.qa_chain.invoke(query)
            
            # Cleanup unwanted generation artifacts if any
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
            
            source_documents = self.retriever.get_relevant_documents(query)
            
            sources = []
            for doc in source_documents:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def reload(self) -> None:
        """Reload the vector store and QA chain."""
        logger.info("Reloading RAG engine...")
        self._initialize()


rag_engine = RAGEngine()

