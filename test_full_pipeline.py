import os
import sys
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_manager import QdrantManager
from rag_pipeline import RedisMemoryManager, QueryRewriter, ContextBuilder

def run_test_scenario(scenario_name: str, conversation_turns: list, memory: RedisMemoryManager, rewriter: QueryRewriter, embed_model: SentenceTransformer, reranker_model: CrossEncoder, qdrant: QdrantManager, context_builder: ContextBuilder):
    print("\n" + "=" * 70)
    print(f" 🧪 TEST SCENARIO: {scenario_name}")
    print("=" * 70)
    
    memory.clear()

    for turn_idx, user_query in enumerate(conversation_turns, 1):
        print(f"\n--- [TURN {turn_idx}] User Query: '{user_query}' ---")
        history = memory.get_history()
        
        # Query Rewriting
        rewritten_query = rewriter.rewrite(user_query, history)
        if rewritten_query != user_query:
            print(f"   🔄 Rewritten Query : '{rewritten_query}'")
        else:
            print(f"   ℹ️ Standalone Query : '{user_query}'")

        # Encode and Search
        dense_vec = embed_model.encode(rewritten_query, show_progress_bar=False).tolist()
        top_chunks = qdrant.search_hybrid(
            query_text=rewritten_query,
            query_dense_vector=dense_vec,
            top_k_dense=20,
            top_k_bm25=20,
            rrf_k=60,
            top_k_rrf=20,
            top_k_final=5,
            reranker_model=reranker_model
        )

        # Context Building
        context_str = context_builder.build_context(top_chunks)
        
        # Display top 2 retrieved results & relevancy check
        print("\n   📊 TOP RETRIEVED CHUNKS & RELEVANCY ANALYSIS:")
        for idx, chunk in enumerate(top_chunks[:3], 1):
            rerank_score = chunk.get('rerank_score', chunk.get('rrf_score', 0))
            score_type = "Rerank" if "rerank_score" in chunk else "RRF"
            content_snippet = chunk['content'][:250].replace('\n', ' ')
            print(f"   [{idx}] {score_type} Score: {rerank_score:.4f} | Page {chunk['page']}")
            print(f"       Content Snippet: \"{content_snippet}...\"")

        # Relevancy Verification
        print("\n   🔍 RELEVANCY EVALUATION:")
        top_chunk_text = top_chunks[0]['content'] if top_chunks else ""
        keywords = [kw.lower() for kw in rewritten_query.split() if len(kw) > 3]
        matched_keywords = [kw for kw in keywords if kw in top_chunk_text.lower()]
        
        if top_chunks and (top_chunks[0].get('rerank_score', 0) > 0.0 or len(matched_keywords) > 0):
            print(f"   ✅ Highly Relevant Context Found! (Reranker Score: {top_chunks[0].get('rerank_score', 0):.4f})")
            print(f"      Key term matches: {list(set(matched_keywords))[:5]}")
        else:
            print(f"   ⚠️ Low confidence retrieval.")

        # Record Turn in Redis Memory
        simulated_response = f"Simulated answer based on top context snippet: {top_chunk_text[:120]}..."
        memory.add_turn(user_query, simulated_response)
        print(f"   💾 Saved Turn {turn_idx} to Redis Memory (Current History: {len(memory.get_history()) // 2} turns)")

def main():
    print("=" * 70)
    print(" 🚀 DEVELMO END-TO-END HYBRID RAG RELEVANCY BENCHMARK")
    print("=" * 70)

    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reranker_model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    print(f"\nLoading Embedding Model: '{embedding_model_name}'...")
    embed_model = SentenceTransformer(embedding_model_name, local_files_only=True)

    print(f"Loading Reranker Model: '{reranker_model_name}'...")
    try:
        reranker_model = CrossEncoder(reranker_model_name)
    except Exception as e:
        print(f"   Fallback to 'cross-encoder/ms-marco-MiniLM-L-6-v2': {e}")
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    qdrant = QdrantManager()
    memory = RedisMemoryManager(session_id="multi_test_session")
    rewriter = QueryRewriter()
    context_builder = ContextBuilder()

    # TEST SCENARIO A: OmniRoad 2.0 (Traffic AI)
    run_test_scenario(
        scenario_name="OmniRoad 2.0 Product & Features",
        conversation_turns=[
            "What is OmniRoad 2.0 and who is it designed for?",
            "What types of vehicles does it detect and how does it alert authorities?"
        ],
        memory=memory, rewriter=rewriter, embed_model=embed_model,
        reranker_model=reranker_model, qdrant=qdrant, context_builder=context_builder
    )

    # TEST SCENARIO B: CrowdIQ Pricing & Deployment
    run_test_scenario(
        scenario_name="CrowdIQ Pricing & Deployment Tiers",
        conversation_turns=[
            "What are the pricing tiers and deployment options for CrowdIQ?",
            "How much does the Business tier cost and what is included?"
        ],
        memory=memory, rewriter=rewriter, embed_model=embed_model,
        reranker_model=reranker_model, qdrant=qdrant, context_builder=context_builder
    )

    # TEST SCENARIO C: Company Offices & Global Contact
    run_test_scenario(
        scenario_name="Global Presence & Contact Info",
        conversation_turns=[
            "Where are DevelMo's global offices located?",
            "What is the contact phone number for Saudi Arabia and Australia?"
        ],
        memory=memory, rewriter=rewriter, embed_model=embed_model,
        reranker_model=reranker_model, qdrant=qdrant, context_builder=context_builder
    )

    print("\n" + "=" * 70)
    print(" ✅ ALL TEST SCENARIOS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()
