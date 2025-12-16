# Financial Research RAG Assistant

A production-ready system for document-based question answering using Retrieval-Augmented Generation (RAG). This project demonstrates expertise in building RAG pipelines, working with vector databases, and deploying AI models with GPU acceleration.

## 🏗️ Architecture

### RAG Pipeline Overview

```
PDF Document → Text Extraction → Chunking → Embeddings (GPU) → FAISS Vector Store
                                                              ↓
User Query → Embedding → Similarity Search → Retrieved Context → LLM (GPU) → Answer
```

### Key Components

1. **Document Ingestion** (`app/services/ingest.py`)
   - Loads PDF documents using PyPDF
   - Splits text into semantic chunks with overlap (500 chars, 50 overlap)
   - Creates embeddings using HuggingFace sentence transformers (GPU accelerated)
   - Stores vectors in FAISS for fast similarity search

2. **RAG Engine** (`app/services/rag.py`)
   - Loads FAISS vector store
   - Retrieves relevant document chunks for queries
   - Uses GPT-2 LLM (GPU accelerated) to generate answers
   - Returns answers with source attribution

3. **REST API** (`app/main.py`)
   - FastAPI endpoints for document upload and querying
   - Automatic API documentation (Swagger UI)
   - Error handling and validation

4. **Streamlit Frontend** (`frontend.py`)
   - User-friendly web interface
   - Document upload and chat interface
   - Source document visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended) - tested on RTX 4050 (6GB VRAM)
- HuggingFace account (free) - for downloading models

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA** (if using GPU):
   ```bash
   # For CUDA 11.7 (RTX 4050 compatible)
   pip install --user torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   - Create `.env` file in project root:
     ```
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```
   - Get a free token at: https://huggingface.co/settings/tokens
   - Note: Token is optional for public models, but recommended

### Running the Application

#### Option 1: Using the Streamlit Frontend (Recommended)

**Terminal 1 - Start Backend:**
```bash
python -m app.main
```

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend.py
```

Then open http://localhost:8501 in your browser.

#### Option 2: Using the API Directly

Start the FastAPI server:
```bash
python -m app.main
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 API Usage

### 1. Upload a PDF Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/document.pdf"
```

**Response:**
```json
{
  "message": "Document 'document.pdf' processed successfully",
  "vector_store_path": "./vector_store",
  "status": "success"
}
```

### 2. Query the Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What was the company's revenue in the last fiscal year?\"}"
```

**Response:**
```json
{
  "answer": "Based on the financial documents...",
  "sources": [
    {
      "content": "Revenue for fiscal year 2023 was...",
      "metadata": {
        "source": "document.pdf",
        "page": 81
      }
    }
  ],
  "num_sources": 4
}
```

### 3. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## 🛠️ Configuration

Configuration is managed through environment variables (create `.env` file):

- `HUGGINGFACEHUB_API_TOKEN`: Your HuggingFace API token (optional but recommended)
- `EMBEDDING_MODEL_NAME`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `VECTOR_STORE_PATH`: Path to store FAISS index (default: `./vector_store`)
- `CHUNK_SIZE`: Text chunk size (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `MAX_FILE_SIZE_MB`: Maximum upload size (default: 50)

## 📁 Project Structure

```
Financial Research Assistant (RAG)/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py           # Configuration management
│   │
│   └── services/
│       ├── __init__.py
│       ├── ingest.py            # Document processing pipeline
│       └── rag.py               # RAG engine implementation
│
├── frontend.py                  # Streamlit frontend
├── vector_store/               # FAISS vector database (auto-created)
├── uploads/                    # Temporary upload directory (auto-created)
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔍 Key Features

### Production-Ready Design

- **Error Handling**: Comprehensive error handling with meaningful messages
- **Type Hints**: Full type annotations for better code quality
- **Logging**: Structured logging throughout the application
- **Validation**: Pydantic models for request/response validation
- **Documentation**: Inline comments explaining design decisions

### Technical Highlights

- **GPU Acceleration**: Both embeddings and LLM run on GPU (CUDA)
- **Modular Architecture**: Separation of concerns (ingestion, RAG, API)
- **Incremental Updates**: Can add documents without reprocessing existing ones
- **Source Attribution**: Returns source documents with page numbers for verification
- **Efficient Storage**: FAISS for fast similarity search
- **API Documentation**: Auto-generated Swagger/OpenAPI docs
- **Web Interface**: Streamlit frontend for easy interaction

### Models Used

- **LLM**: GPT-2 (548MB) - Compatible with PyTorch 1.13.1, runs on GPU
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (90MB) - GPU accelerated
- **Vector Store**: FAISS (CPU version) - Fast similarity search

## 💻 System Requirements

### Minimum Requirements
- Python 3.9+
- 8GB RAM
- 2GB free disk space (for models)

### Recommended (GPU)
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 4050)
- PyTorch 1.13.1+cu117 (for CUDA 11.7)
- CUDA Toolkit 11.7+

### CPU Fallback
- System will automatically fall back to CPU if GPU is not available
- CPU mode is slower but fully functional

## 🧪 Testing the System

### Using Streamlit Frontend

1. Start backend: `python -m app.main`
2. Start frontend: `streamlit run frontend.py`
3. Upload a PDF document (e.g., SEC 10-K filing)
4. Ask questions about the document

### Using Swagger UI

1. Navigate to http://localhost:8000/docs
2. Use the interactive API documentation to test endpoints
3. Upload a PDF and query it directly from the browser

### Example Questions for Financial Documents

- "What was the company's total revenue for the fiscal year?"
- "What are the main risk factors mentioned in the 10-K?"
- "What is the company's debt-to-equity ratio?"
- "Who are the key executives?"
- "What does the MD&A say about future outlook?"

## 🔧 Troubleshooting

### GPU Not Detected
- Verify CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Check PyTorch CUDA version matches your CUDA installation
- System will automatically use CPU if GPU is unavailable

### "Vector store not found" error
- Upload at least one PDF document first using `/upload` endpoint or frontend

### Slow query responses
- Normal for first query (model loading)
- Ensure GPU is being used (check logs for "Using GPU" message)
- Large documents may take longer to process

### Import errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Verify all dependencies are installed correctly

### Model download issues
- Check internet connection
- Verify HuggingFace token is set (optional but recommended)
- Models download automatically on first use (~600MB total)

## 📝 Notes

- **First Run**: Models download automatically on first use (~600MB)
- **Model Caching**: Models are cached locally after first download
- **Memory Usage**: GPT-2 uses ~548MB VRAM (fits in 6GB GPU without quantization)
- **Document Processing**: Large PDFs (100+ pages) may take 1-2 minutes to process
- **Answer Quality**: GPT-2 is not instruction-tuned, so answers may require interpretation of context

## 🎯 Use Cases

- **Financial Analysis**: Query 10-K reports, earnings statements, and financial disclosures
- **Research Assistant**: Extract insights from research papers and reports
- **Document Q&A**: Build a knowledge base from PDF documents
- **Portfolio Project**: Demonstrate RAG pipeline implementation skills

## 🔄 Future Improvements

- Upgrade to instruction-tuned models (requires PyTorch 2.x+)
- Add support for other document types (Word, Excel, etc.)
- Implement advanced retrieval strategies (hybrid search, reranking)
- Add support for conversational memory
- Deploy with Docker for easy distribution

## 📄 License

This project is a portfolio piece for demonstration purposes.

---

**Built with**: Python, FastAPI, Streamlit, LangChain, HuggingFace, FAISS, PyTorch
