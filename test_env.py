import os
import sys

# Suppress noisy HuggingFace Hub warnings & progress logs
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from dotenv import load_dotenv

# Ensure stdout handles unicode encoding safely on Windows CP1252 consoles
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load .env file
load_dotenv()

print("=" * 50)
print(" TESTING ENVIRONMENT VARIABLES & CONNECTIONS")
print("=" * 50)

# 1. Print loaded values (masking secrets)
def mask(val, show_chars=4):
    if not val:
        return "[NOT SET]"
    if len(val) <= show_chars * 2:
        return "***"
    return f"{val[:show_chars]}...{val[-show_chars:]}"

print("\n--- [1] Environment Variables ---")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_coll = os.getenv("QDRANT_COLLECTION")
redis_url = os.getenv("REDIS_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
reranker_model_name = os.getenv("RERANKER_MODEL")
llm_key = os.getenv("LLM_API_KEY")
llm_base = os.getenv("LLM_BASE_URL")
llm_model = os.getenv("LLM_MODEL")

print(f"QDRANT_URL        : {qdrant_url or '[NOT SET]'}")
print(f"QDRANT_API_KEY    : {mask(qdrant_key)}")
print(f"QDRANT_COLLECTION : {qdrant_coll or '[NOT SET]'}")
print(f"REDIS_URL         : {mask(redis_url, 12)}")
print(f"EMBEDDING_MODEL   : {embedding_model_name or '[NOT SET]'}")
print(f"RERANKER_MODEL    : {reranker_model_name or '[NOT SET]'}")
print(f"LLM_BASE_URL      : {llm_base or '[NOT SET]'}")
print(f"LLM_MODEL         : {llm_model or '[NOT SET]'}")
print(f"LLM_API_KEY       : {mask(llm_key)}")

# 2. Test Qdrant Connection
print("\n--- [2] Testing Qdrant Connection ---")
if not qdrant_url or not qdrant_key:
    print("[SKIP] QDRANT_URL or QDRANT_API_KEY missing.")
else:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10)
        collections = client.get_collections()
        col_names = [c.name for c in collections.collections]
        print(f"[SUCCESS] Qdrant Connection OK!")
        print(f"   Existing Collections: {col_names}")
    except Exception as e:
        print(f"[FAIL] Qdrant Connection FAILED: {e}")

# 3. Test Redis Connection
print("\n--- [3] Testing Redis Connection ---")
if not redis_url:
    print("[SKIP] REDIS_URL missing.")
else:
    try:
        import redis
        r = redis.from_url(redis_url, socket_timeout=10)
        response = r.ping()
        if response:
            print("[SUCCESS] Redis Connection OK! (Ping -> PONG)")
        else:
            print("[FAIL] Redis Ping returned False.")
    except Exception as e:
        print(f"[FAIL] Redis Connection FAILED: {e}")

# 4. Test LLM API Connection
print("\n--- [4] Testing LLM API Connection ---")
if not llm_key:
    print("[SKIP] LLM_API_KEY missing.")
else:
    try:
        import httpx
        url = f"{llm_base.rstrip('/')}/models" if llm_base else "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {llm_key}"}
        res = httpx.get(url, headers=headers, timeout=10.0)
        if res.status_code == 200:
            print("[SUCCESS] LLM API Connection OK!")
        else:
            print(f"[WARNING] LLM API returned status {res.status_code}: {res.text[:150]}")
    except Exception as e:
        print(f"[FAIL] LLM API Connection FAILED: {e}")

# 5. Test Embedding Model
print("\n--- [5] Testing Embedding Model ---")
if not embedding_model_name:
    print("[SKIP] EMBEDDING_MODEL missing.")
else:
    try:
        print(f"Loading embedding model '{embedding_model_name}'...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(embedding_model_name)
        test_text = "Hello, testing embedding generation."
        vector = embed_model.encode(test_text)
        print(f"[SUCCESS] Embedding Model Loaded OK!")
        print(f"   Model Name     : {embedding_model_name}")
        print(f"   Vector Dimension: {len(vector)}")
        print(f"   Sample Vector  : [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ...]")
    except Exception as e:
        print(f"[FAIL] Embedding Model test FAILED: {e}")

print("\n" + "=" * 50)
print(" TEST COMPLETED")
print("=" * 50)
