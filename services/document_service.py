import os
import json
import uuid
import hashlib
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.documents import DocumentInfo
from document_processor import DocumentProcessor
from qdrant_manager import QdrantManager
from sentence_transformers import SentenceTransformer

class DuplicateDocumentError(Exception):
    def __init__(self, message: str, existing_document_id: str):
        super().__init__(message)
        self.existing_document_id = existing_document_id

class DocumentMetadataStore:
    """JSON-backed persistent metadata repository for uploaded documents."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join("data", "documents_meta.json")
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _read_data(self) -> List[Dict[str, Any]]:
        self._ensure_db_exists()
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_data(self, data: List[Dict[str, Any]]):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        temp_path = f"{self.db_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.db_path)

    def get_all(self) -> List[DocumentInfo]:
        raw_data = self._read_data()
        return [DocumentInfo(**item) for item in raw_data]

    def get_by_id(self, doc_id: str) -> Optional[DocumentInfo]:
        for item in self._read_data():
            if item.get("id") == doc_id:
                return DocumentInfo(**item)
        return None

    def get_by_hash(self, file_hash: str) -> Optional[DocumentInfo]:
        for item in self._read_data():
            if item.get("file_hash") == file_hash:
                return DocumentInfo(**item)
        return None

    def add(self, doc: DocumentInfo):
        data = self._read_data()
        # Remove any existing record with same ID if replacing
        data = [item for item in data if item.get("id") != doc.id]
        data.append(doc.dict())
        self._write_data(data)

    def delete(self, doc_id: str) -> bool:
        data = self._read_data()
        initial_len = len(data)
        new_data = [item for item in data if item.get("id") != doc_id]
        if len(new_data) < initial_len:
            self._write_data(new_data)
            return True
        return False

class DocumentIngestionService:
    """Unified service for document ingestion, duplicate detection, Qdrant upsertion, and deletion."""

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    def __init__(self, embed_model: Optional[SentenceTransformer] = None, qdrant_mgr: Optional[QdrantManager] = None):
        self.upload_dir = os.getenv("DOCUMENT_UPLOAD_DIR", os.path.join("data", "uploads"))
        self.max_size_mb = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "25"))
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
        
        os.makedirs(self.upload_dir, exist_ok=True)
        
        self.processor = DocumentProcessor()
        self.meta_store = DocumentMetadataStore()
        
        self._embed_model = embed_model
        self._qdrant = qdrant_mgr

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            embed_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            print(f"🤖 DocumentIngestionService: Loading Embedding Model '{embed_name}'...")
            self._embed_model = SentenceTransformer(embed_name, local_files_only=True)
        return self._embed_model

    @property
    def qdrant(self) -> QdrantManager:
        if self._qdrant is None:
            self._qdrant = QdrantManager()
        return self._qdrant

    def _sanitize_filename(self, filename: str) -> str:
        # Prevent path traversal and strip non-safe characters
        clean_name = os.path.basename(filename)
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
        return clean_name or "document.txt"

    def _compute_hash(self, content_bytes: bytes) -> str:
        return hashlib.sha256(content_bytes).hexdigest()

    def process_and_ingest_bytes(self, content_bytes: bytes, filename: str) -> DocumentInfo:
        """Processes raw file bytes, validates format & size, embeds chunks, and stores in Qdrant."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}")

        if len(content_bytes) > self.max_size_bytes:
            raise ValueError(f"File size ({len(content_bytes) / (1024*1024):.2f} MB) exceeds maximum allowed limit of {self.max_size_mb} MB.")

        if not content_bytes or len(content_bytes.strip()) == 0:
            raise ValueError("Uploaded file is empty.")

        # 1. Duplicate detection via SHA-256 hash
        file_hash = self._compute_hash(content_bytes)
        existing_doc = self.meta_store.get_by_hash(file_hash)
        if existing_doc:
            raise DuplicateDocumentError("This document has already been uploaded.", existing_doc.id)

        # 2. Generate unique document_id & save file locally
        doc_uuid = uuid.uuid4().hex[:12]
        doc_id = f"doc_{doc_uuid}"
        sanitized_name = self._sanitize_filename(filename)
        saved_filename = f"{doc_id}_{sanitized_name}"
        abs_upload_dir = os.path.abspath(self.upload_dir)
        saved_path = os.path.abspath(os.path.join(self.upload_dir, saved_filename))


        # Path traversal security check
        if not saved_path.startswith(abs_upload_dir):
            raise ValueError("Invalid file path / path traversal detected.")


        with open(saved_path, "wb") as f:
            f.write(content_bytes)

        # 3. Chunking & Rollback safety
        try:
            chunks = self.processor.chunk_document(saved_path, document_id=doc_id)
            if not chunks:
                raise ValueError(f"No valid text could be extracted from document '{filename}'.")
            
            # Override chunk source metadata to original filename
            for c in chunks:
                c["metadata"]["source"] = sanitized_name
                c["metadata"]["document_id"] = doc_id
                c["metadata"]["doc_id"] = doc_id

            # 4. Dense Embeddings
            texts = [c["content"] for c in chunks]
            embeddings = self.embed_model.encode(texts, show_progress_bar=False, batch_size=32).tolist()

            # 5. Qdrant Dual-Vector Upsert
            self.qdrant.upsert_chunks(chunks, embeddings)

            # 6. Save metadata record
            doc_info = DocumentInfo(
                id=doc_id,
                filename=sanitized_name,
                file_type=ext.lstrip('.'),
                size=len(content_bytes),
                chunk_count=len(chunks),
                uploaded_at=datetime.utcnow().isoformat() + "Z",
                file_hash=file_hash
            )
            self.meta_store.add(doc_info)
            return doc_info

        except Exception as e:
            # ROLLBACK LOGIC: Clean up Qdrant points and local file on failure
            print(f"⚠️ Upload failed during processing ({e}). Rolling back changes for '{doc_id}'...")
            try:
                self.qdrant.delete_document_points(doc_id)
            except Exception:
                pass
            if os.path.exists(saved_path):
                try:
                    os.remove(saved_path)
                except Exception:
                    pass
            raise e

    def process_and_ingest_path(self, file_path: str) -> DocumentInfo:
        """Helper for local file path ingestion (CLI)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            content_bytes = f.read()

        return self.process_and_ingest_bytes(content_bytes, filename)

    def list_documents(self) -> List[DocumentInfo]:
        """Returns all registered document metadata records."""
        return self.meta_store.get_all()

    def delete_document(self, document_id: str) -> bool:
        """Deletes Qdrant points, local stored file, and metadata record for document_id or filename."""
        doc_info = self.meta_store.get_by_id(document_id)
        
        # 1. Delete Qdrant points matching document_id, doc_id, or source filename
        self.qdrant.delete_document_points(document_id)

        if doc_info:
            # 2. Delete local stored file
            filename = doc_info.filename
            saved_filename = f"{document_id}_{filename}"
            saved_path = os.path.join(self.upload_dir, saved_filename)
            if os.path.exists(saved_path):
                try:
                    os.remove(saved_path)
                except Exception as e:
                    print(f"⚠️ Could not delete local file '{saved_path}': {e}")

            # Also search upload_dir for any file starting with document_id
            try:
                for f in os.listdir(self.upload_dir):
                    if f.startswith(document_id):
                        p = os.path.join(self.upload_dir, f)
                        if os.path.exists(p):
                            os.remove(p)
            except Exception:
                pass

            # 3. Remove metadata record
            self.meta_store.delete(document_id)
            return True
        else:
            # If doc_info wasn't in metadata JSON, check if metadata deletion deleted anything by ID
            deleted_meta = self.meta_store.delete(document_id)
            return True

    def clear_all_documents(self) -> bool:
        """Wipes all Qdrant vector points, deletes local upload files, and resets metadata store."""
        # 1. Clear Qdrant collection points
        self.qdrant.clear_all_points()

        # 2. Clear local uploads directory
        if os.path.exists(self.upload_dir):
            for f in os.listdir(self.upload_dir):
                p = os.path.join(self.upload_dir, f)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        # 3. Reset metadata store
        self.meta_store._write_data([])
        return True

