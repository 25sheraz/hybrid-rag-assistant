import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_manager import QdrantManager
from rag_pipeline import RedisMemoryManager, QueryRewriter, ContextBuilder, LLMGenerator
from routers.documents import router as documents_router

app = FastAPI(title="DevelMo RAG API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)


# ---------------------------------------------------------
# INITIALIZE RAG PIPELINE
# ---------------------------------------------------------
print("⚡ Initializing DevelMo RAG Pipeline Models...")
embed_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embed_model = SentenceTransformer(embed_name, local_files_only=True)

reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
try:
    reranker_model = CrossEncoder(reranker_name)
except Exception:
    reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

qdrant = QdrantManager()
rewriter = QueryRewriter()
context_builder = ContextBuilder()
llm = LLMGenerator()

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "web_default_session"

class ChatResponse(BaseModel):
    query: str
    rewritten_query: str
    response: str
    sources: list

@app.get("/")
def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "DevelMo RAG Server Online", "docs": "/docs"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    memory = RedisMemoryManager(session_id=req.session_id)
    history = memory.get_history()

    # 1. Query Rewriting
    rewritten = rewriter.rewrite(req.query, history)

    # 2. Vector Encoding
    dense_vec = embed_model.encode(rewritten, show_progress_bar=False).tolist()

    # 3. Hybrid Search + Reranker
    top_chunks = qdrant.search_hybrid(
        query_text=rewritten,
        query_dense_vector=dense_vec,
        top_k_dense=int(os.getenv("TOP_K_DENSE", "20")),
        top_k_bm25=int(os.getenv("TOP_K_BM25", "20")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        top_k_rrf=int(os.getenv("TOP_K_RRF", "20")),
        top_k_final=int(os.getenv("TOP_K_FINAL", "5")),
        reranker_model=reranker_model
    )

    # 4. Context Assembly
    payload = context_builder.build_prompt_payload(req.query, top_chunks, history)

    # 5. LLM Answer Generation
    answer = llm.generate(payload)

    # 6. Record Turn in Redis
    memory.add_turn(req.query, answer)

    sources = [
        {"source": c.get("source", "Knowledge Base"), "page": c.get("page", 1)}
        for c in top_chunks
    ]

    return ChatResponse(
        query=req.query,
        rewritten_query=rewritten,
        response=answer,
        sources=sources
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
