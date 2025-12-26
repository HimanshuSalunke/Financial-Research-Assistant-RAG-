import os
import requests
from pathlib import Path
from tqdm import tqdm
from app.core.config import settings

def download_file(url: str, dest_path: Path):
    """Download a file with progress bar."""
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return

    print(f"Downloading {dest_path.name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def main():
    # Create models directory
    models_dir = settings.model_path.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download Phi-3 Mini 4K Instruct (Q4_K_M GGUF)
    # Direct download link from HuggingFace
    model_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true"
    download_file(model_url, settings.model_path)
    
    # 2. Download BGE Embedding Model
    print("\nTriggering download of BGE embedding model via SentenceTransformers...")
    from sentence_transformers import SentenceTransformer
    # This will cache it in ~/.cache/torch/sentence_transformers
    SentenceTransformer(settings.embedding_model_name)
    print("Embedding model downloaded successfully.")

if __name__ == "__main__":
    main()
