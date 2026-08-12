import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure safe UTF-8 output on Windows consoles
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Suppress noisy HF warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_manager import QdrantManager
from rag_pipeline import RedisMemoryManager, QueryRewriter, ContextBuilder, LLMGenerator

BANNER = r"""
  ██████╗  ███████╗██╗   ██╗███████╗██╗     ███╗   ███╗██████╗ 
  ██╔══██╗ ██╔════╝██║   ██║██╔════╝██║     ████╗ ████║██╔══██╗
  ██║  ██║ █████╗  ██║   ██║█████╗  ██║     ██╔████╔██║██║  ██║
  ██║  ██║ ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██║     ██║╚██╔╝██║██║  ██║
  ██████╔╝ ███████╗ ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║██████╔╝
  ╚═════╝  ╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝╚═════╝ 
        🤖 Production Dual-Vector Hybrid RAG Assistant
"""

def main():
    parser = argparse.ArgumentParser(description="DevelMo RAG Interactive Assistant")
    parser.add_argument("--session", "-s", default="cli_user", help="Session ID for Redis conversation memory.")
    parser.add_argument("--sources", action="store_true", help="Always display retrieved context chunks.")
    args = parser.parse_args()

    print(BANNER)
    print("=" * 65)
    print("⚡ INITIALIZING DEVELMO HYBRID RAG PIPELINE...")
    print("=" * 65)

    # Load Embedding Model
    embed_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"🤖 [1/5] Loading Embedding Model: '{embed_name}'...")
    embed_model = SentenceTransformer(embed_name, local_files_only=True)

    # Load Reranker Model
    reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"🤖 [2/5] Loading Cross-Encoder Reranker: '{reranker_name}'...")
    try:
        reranker_model = CrossEncoder(reranker_name)
    except Exception as e:
        print(f"   Fallback to 'cross-encoder/ms-marco-MiniLM-L-6-v2': {e}")
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Connect to Services
    print("☁️ [3/5] Connecting to Qdrant Cloud Collection...")
    qdrant = QdrantManager()

    print(f"🧠 [4/5] Initializing Redis 5-Turn Memory (Session: '{args.session}')...")
    memory = RedisMemoryManager(session_id=args.session)

    print("🔄 [5/5] Initializing Query Rewriter & LLM Generator...")
    rewriter = QueryRewriter()
    context_builder = ContextBuilder()
    llm = LLMGenerator()

    show_sources = args.sources

    print("\n" + "=" * 65)
    print("✅ DEVELMO RAG SYSTEM IS READY!")
    print("💡 Commands: 'exit' to quit | '/clear' to reset memory | '/sources' to toggle chunks")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle Commands
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Thank you for using DevelMo AI Assistant!")
            break
        elif user_input.lower() == "/clear":
            memory.clear()
            print("🧹 Conversation memory cleared for this session.\n")
            continue
        elif user_input.lower() == "/sources":
            show_sources = not show_sources
            status = "ENABLED" if show_sources else "DISABLED"
            print(f"📌 Source chunk display is now {status}.\n")
            continue
        elif user_input.lower() == "/history":
            hist = memory.get_history()
            print(f"\n📜 REDIS HISTORY ({len(hist) // 2} turns):")
            for msg in hist:
                print(f"   [{msg['role'].upper()}]: {msg['content'][:100]}...")
            print()
            continue

        # Pipeline Processing
        history = memory.get_history()
        
        # 1. Query Rewriting
        rewritten_query = rewriter.rewrite(user_input, history)

        # 2. Vector Encoding
        dense_vec = embed_model.encode(rewritten_query, show_progress_bar=False).tolist()

        # 3. Hybrid Search (Dense + BM25 + RRF + Cross-Encoder Reranking)
        top_chunks = qdrant.search_hybrid(
            query_text=rewritten_query,
            query_dense_vector=dense_vec,
            top_k_dense=int(os.getenv("TOP_K_DENSE", "20")),
            top_k_bm25=int(os.getenv("TOP_K_BM25", "20")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            top_k_rrf=int(os.getenv("TOP_K_RRF", "20")),
            top_k_final=int(os.getenv("TOP_K_FINAL", "5")),
            reranker_model=reranker_model
        )

        # Optional Source Chunk Display
        if show_sources and top_chunks:
            print("\n📌 [RETRIEVED CONTEXT CHUNKS]")
            for idx, chunk in enumerate(top_chunks[:3], 1):
                score_val = chunk.get('rerank_score', chunk.get('rrf_score', 0))
                print(f"   [{idx}] Score: {score_val:.4f} | Source: {chunk['source']} (Page {chunk['page']})")
                print(f"       {chunk['content'][:180].replace(chr(10), ' ')}...")
            print()

        # 4. Context Payload Assembly
        payload = context_builder.build_prompt_payload(user_input, top_chunks, history)

        # 5. LLM Response Generation
        print("🤖 DevelMo AI: ", end="", flush=True)
        response = llm.generate(payload)
        print(response + "\n")

        # 6. Save Turn to Redis Memory
        memory.add_turn(user_input, response)

if __name__ == "__main__":
    main()
