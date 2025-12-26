"""
Streamlit Frontend for Financial Research RAG Assistant

This frontend provides a user-friendly interface for:
1. Uploading and processing PDF documents
2. Querying the RAG system with questions
3. Viewing answers with source document citations

Why Streamlit?
- Rapid development of interactive web apps
- Built-in components for file uploads and chat interfaces
- No HTML/CSS/JavaScript required
- Perfect for AI/ML applications
"""

import streamlit as st
import requests
from typing import Optional, Dict, List

# Page configuration
# Why these settings?
# - Wide layout: Better for chat interface
# - Page title: Shows in browser tab
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


def check_api_health() -> bool:
    """
    Check if the backend API is available.
    
    Returns:
        bool: True if API is online, False otherwise
    """
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def upload_document(file) -> Dict:
    """
    Upload and process a PDF document.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        dict: Response from API with status and message
    """
    try:
        files = {"file": (file.name, file.read(), "application/pdf")}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=300)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to backend API. Please ensure the server is running."
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The document may be too large or processing is taking too long."
        }
    except requests.exceptions.HTTPError as e:
        error_msg = "Unknown error occurred."
        try:
            error_data = e.response.json()
            error_msg = error_data.get("detail", error_msg)
        except:
            error_msg = str(e)
        return {"success": False, "error": error_msg}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def query_documents(question: str) -> Dict:
    """
    Send a question to the RAG system.
    
    Args:
        question: User's question string
        
    Returns:
        dict: Response from API with answer and sources
    """
    try:
        payload = {"question": question}
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to backend API. Please ensure the server is running."
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out. The query is taking too long to process."
        }
    except requests.exceptions.HTTPError as e:
        error_msg = "Unknown error occurred."
        try:
            error_data = e.response.json()
            error_msg = error_data.get("detail", error_msg)
        except:
            error_msg = str(e)
        return {"success": False, "error": error_msg}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# Initialize session state
# Why session state?
# - Persists chat history across interactions
# - Maintains state during Streamlit reruns
# - Stores conversation context
if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# Main Title
st.title("📊 Financial Research Assistant (GPU Powered)")
st.markdown("---")

# Sidebar for document upload
with st.sidebar:
    st.header("📄 Document Upload")
    st.markdown("Upload financial documents (PDF) to build your knowledge base.")
    
    # Check API health
    api_online = check_api_health()
    if api_online:
        st.success("✅ Backend API Online")
    else:
        st.error("❌ Backend API Offline")
        st.warning("Please start the backend server:\n```bash\npython -m app.main\n```")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a financial document (e.g., 10-K report, earnings statement)"
    )
    
    # Process document button
    if uploaded_file is not None:
        st.info(f"📎 Selected: **{uploaded_file.name}**")
        
        if st.button("🔄 Process Document", type="primary", use_container_width=True):
            if not api_online:
                st.error("Cannot process document: Backend API is offline.")
            else:
                with st.spinner("Processing document... This may take a minute."):
                    result = upload_document(uploaded_file)
                    
                    if result["success"]:
                        st.success("✅ Document processed successfully!")
                        st.session_state.document_processed = True
                        # Show processing details
                        if "data" in result:
                            data = result["data"]
                            st.json(data)
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    st.markdown("---")
    
    # Status information
    st.subheader("ℹ️ Status")
    if st.session_state.document_processed:
        st.success("Document ready for queries")
    else:
        st.info("Upload and process a document to start")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
st.header("💬 Ask Questions")

# Display chat history
# Why display history?
# - Maintains conversation context
# - Better user experience
# - Shows previous Q&A pairs
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 Source Documents ({len(message['sources'])} sources)", expanded=False):
                for idx, source in enumerate(message["sources"], 1):
                    # Extract page number from metadata if available
                    page = source.get("metadata", {}).get("page", "N/A")
                    st.markdown(f"**Source {idx}** (Page {page})")
                    st.text(source.get("content", "")[:300] + "...")
                    st.markdown("---")

# Chat input
# Why chat input?
# - Native Streamlit component for chat interfaces
# - Better UX than regular text input
if prompt := st.chat_input("Ask a question about your financial documents..."):
    # Check if API is online
    if not api_online:
        st.error("❌ Backend API is offline. Please start the server first.")
    elif not st.session_state.document_processed:
        st.warning("⚠️ Please upload and process a document first before asking questions.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_documents(prompt)
                
                if result["success"]:
                    data = result["data"]
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("sources", [])
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources in expander
                    if sources:
                        with st.expander(f"📚 Source Documents ({len(sources)} sources)", expanded=False):
                            for idx, source in enumerate(sources, 1):
                                # Extract page number from metadata
                                metadata = source.get("metadata", {})
                                page = metadata.get("page", "N/A")
                                source_file = metadata.get("source", "Unknown")
                                
                                st.markdown(f"**Source {idx}**")
                                st.caption(f"📄 File: {source_file} | 📑 Page: {page}")
                                st.text(source.get("content", "")[:500])
                                st.markdown("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = result.get("error", "Unknown error occurred.")
                    st.error(f"❌ Error: {error_msg}")
                    st.info("Please check that the backend server is running and documents are processed.")

# Footer
st.markdown("---")
st.caption("Powered by Phi-3, BGE Embeddings, and LlamaCpp | GPU Accelerated with GGUF Quantization")

