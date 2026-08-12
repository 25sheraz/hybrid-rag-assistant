import os
import sys

# Ensure stdout handles unicode encoding safely on Windows CP1252 consoles
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress Hugging Face online checks, warnings, and progress bars
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer

def main():
    query = "What flagship products does DevelMo build and what is CrowdIQ?"
    print(f"🔍 Query: '{query}'\n")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load model from local cache (local_files_only=True prevents HF Hub queries)
    embed_model = SentenceTransformer(model_name, local_files_only=True)
    query_vector = embed_model.encode(query, show_progress_bar=False).tolist()

    manager = QdrantManager()
    results = manager.search_similar(query_vector, limit=3)

    print(f"--- Top {len(results)} Relevant Chunks Retrieved from Qdrant ---")
    for i, res in enumerate(results, 1):
        print(f"\n[Result {i}] Score: {res['score']:.4f} | Source: {res['source']} (Page {res['page']})")
        print(f"Content:\n{res['content'][:250]}...")

if __name__ == "__main__":
    main()
