import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

from qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer, CrossEncoder

def main():
    query = "What flagship products does DevelMo build and what is CrowdIQ?"
    print(f"🔍 QUERY: '{query}'\n")

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embed_model = SentenceTransformer(embedding_model_name, local_files_only=True)
    dense_vector = embed_model.encode(query, show_progress_bar=False).tolist()

    # Load Cross-Encoder Reranker
    reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"🤖 Loading Reranker Model: '{reranker_model_name}'...")
    try:
        reranker_model = CrossEncoder(reranker_model_name)
    except Exception as e:
        print(f"⚠️ Could not load primary reranker model, falling back to 'cross-encoder/ms-marco-MiniLM-L-6-v2': {e}")
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    manager = QdrantManager()

    # 1. Test Dense Search Only
    print("\n--- 1. DENSE SEARCH (Semantic Context) ---")
    dense_results = manager.search_dense(dense_vector, limit=5)
    for i, res in enumerate(dense_results, 1):
        print(f"[Dense {i}] Score: {res['score']:.4f} | Source: {res['source']} (Page {res['page']})")
        print(f"Content:\n{res['content'][:400]}\n{'-'*60}\n")

    # 2. Test BM25 Search Only
    print("--- 2. BM25 SEARCH (Exact Keyword Precision) ---")
    bm25_results = manager.search_bm25(query, limit=5)
    for i, res in enumerate(bm25_results, 1):
        print(f"[BM25 {i}] Score: {res['score']:.4f} | Source: {res['source']} (Page {res['page']})")
        print(f"Content:\n{res['content'][:400]}\n{'-'*60}\n")

    # 3. Test Hybrid RRF Fusion
    print("--- 3. HYBRID RRF FUSION (Fused Candidates) ---")
    hybrid_results = manager.search_hybrid(
        query_text=query,
        query_dense_vector=dense_vector,
        top_k_dense=20,
        top_k_bm25=20,
        rrf_k=60,
        top_k_final=5
    )
    for i, res in enumerate(hybrid_results, 1):
        print(f"[Hybrid RRF {i}] RRF Score: {res['rrf_score']:.6f} | Source: {res['source']} (Page {res['page']})")
        print(f"Content:\n{res['content'][:400]}\n{'-'*60}\n")

    # 4. Test Hybrid RRF + Cross-Encoder Reranking
    print("--- 4. HYBRID RRF + CROSS-ENCODER RERANKER (Final Top 5) ---")
    reranked_results = manager.search_hybrid(
        query_text=query,
        query_dense_vector=dense_vector,
        top_k_dense=20,
        top_k_bm25=20,
        rrf_k=60,
        top_k_rrf=20,
        top_k_final=5,
        reranker_model=reranker_model
    )
    for i, res in enumerate(reranked_results, 1):
        score_str = f"Rerank Score: {res['rerank_score']:.4f}" if "rerank_score" in res else f"RRF Score: {res['rrf_score']:.6f}"
        print(f"[Reranked {i}] {score_str} | Source: {res['source']} (Page {res['page']})")
        print(f"Content:\n{res['content'][:400]}\n{'-'*60}\n")

if __name__ == "__main__":
    main()
