import os
import sys
import argparse
from dotenv import load_dotenv

# Reconfigure stdout encoding for Windows compatibility
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

from services.document_service import DocumentIngestionService, DuplicateDocumentError
from qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer

# Suppress HF noisy warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ingest_file(file_path: str, service: DocumentIngestionService) -> int:
    """Processes a single file using DocumentIngestionService and uploads to Qdrant."""
    print(f"\n📄 Ingesting: {file_path}")
    try:
        doc_info = service.process_and_ingest_path(file_path)
        print(f"   ✅ Ingested successfully (ID: {doc_info.id}, Chunks: {doc_info.chunk_count}, Size: {doc_info.size} bytes).")
        return doc_info.chunk_count
    except DuplicateDocumentError as dup:
        print(f"   ℹ️ Document already ingested (ID: {dup.existing_document_id}). Skipping insertion.")
        return 0
    except Exception as e:
        print(f"   ❌ Ingestion failed for {file_path}: {e}")
        return 0

def ingest_directory(dir_path: str, service: DocumentIngestionService):
    """Ingests all supported documents (.pdf, .md, .txt) in a directory."""
    if not os.path.exists(dir_path):
        print(f"❌ Directory not found: {dir_path}")
        return

    supported_exts = {".pdf", ".md", ".txt", ".markdown"}
    total_files = 0
    total_chunks = 0

    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                file_path = os.path.join(root, f)
                chunks_count = ingest_file(file_path, service)
                total_files += 1
                total_chunks += chunks_count

    print(f"\n✅ Directory Ingestion Complete!")
    print(f"   Files Processed : {total_files}")
    print(f"   Total Chunks    : {total_chunks}")

def main():
    parser = argparse.ArgumentParser(description="Document RAG Ingestion Tool")
    parser.add_argument("--file", "-f", help="Path to a single PDF, Markdown, or Text file to ingest.")
    parser.add_argument("--dir", "-d", help="Path to a directory containing documents to ingest.")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection before ingesting.")
    
    args = parser.parse_args()

    # Default to KnowledgeRAG_Knowledge_Base.md if no arguments supplied
    if not args.file and not args.dir:
        default_file = "KnowledgeRAG_Knowledge_Base.md"
        if os.path.exists(default_file):
            print(f"ℹ️ No file/directory specified. Defaulting to '{default_file}'.")
            args.file = default_file
        else:
            parser.print_help()
            return

    # Initialize components
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(f"🤖 Loading Embedding Model from local cache: '{embedding_model_name}'...")
    embed_model = SentenceTransformer(embedding_model_name, local_files_only=True)

    manager = QdrantManager(force_recreate=args.recreate)
    service = DocumentIngestionService(embed_model=embed_model, qdrant_mgr=manager)

    if args.file:
        ingest_file(args.file, service)
    elif args.dir:
        ingest_directory(args.dir, service)

    # Print final collection status
    info = manager.get_info()
    print("\n" + "=" * 50)
    print(f"📊 QDRANT COLLECTION STATUS: '{info['name']}'")
    print(f"   Total Vector Points: {info['vectors_count']}")
    print(f"   Collection Status  : {info['status']}")
    print("=" * 50)

if __name__ == "__main__":
    main()