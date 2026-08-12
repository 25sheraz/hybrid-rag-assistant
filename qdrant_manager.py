import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Suppress HF noisy warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("fastembed").setLevel(logging.ERROR)

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector
from fastembed import SparseTextEmbedding

load_dotenv()

class QdrantManager:
    """Production-grade Qdrant Manager supporting Dual Vector Indexing (Dense Semantic + BM25 Sparse)."""

    def __init__(self, collection_name: Optional[str] = None, vector_size: int = 384, force_recreate: bool = False):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "rag_documents")
        self.vector_size = vector_size

        if not self.url or not self.api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment or .env file.")

        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=20)
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
        
        self.init_collection(force_recreate=force_recreate)

    def init_collection(self, force_recreate: bool = False):
        """Ensures the target Qdrant collection exists with both Dense and Sparse BM25 vector configs."""
        collections = [col.name for col in self.client.get_collections().collections]

        if force_recreate and self.collection_name in collections:
            print(f"Recreating collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            print(f"Creating hybrid collection '{self.collection_name}' (Dense=384, Sparse=BM25)...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.vector_size, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                }
            )
            
            # Create payload indexes for metadata filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            print(f"Hybrid collection '{self.collection_name}' created successfully with payload indexes.")
        else:
            # Ensure payload index for document_id exists even if collection was previously created
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
            print(f"Collection '{self.collection_name}' already exists.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], dense_embeddings: List[List[float]], batch_size: int = 100):
        """
        Upserts document chunks with BOTH Dense and BM25 Sparse vectors directly into Qdrant Cloud.
        """
        if len(chunks) != len(dense_embeddings):
            raise ValueError("The number of chunks and dense embeddings must match.")

        texts = [c["content"] for c in chunks]
        print(f"   Computing Sparse BM25 embeddings for {len(texts)} chunks...")
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for idx, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
            doc_id_val = chunk['metadata'].get('document_id') or chunk['metadata'].get('doc_id', '')
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id_val}_{chunk['metadata'].get('chunk_index', idx)}"))
            payload = {
                "content": chunk["content"],
                "text": chunk["content"],
                "source": chunk["metadata"].get("source", "unknown"),
                "page": chunk["metadata"].get("page", 1),
                "chunk_index": chunk["metadata"].get("chunk_index", idx),
                "total_chunks": chunk["metadata"].get("total_chunks", 1),
                "doc_id": doc_id_val,
                "document_id": doc_id_val,
            }

            vec_struct = {
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            }
            points.append(PointStruct(id=point_id, vector=vec_struct, payload=payload))

        # Batch upsert points
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
        
        print(f"Successfully upserted {len(points)} dual-vector points into Qdrant Cloud collection '{self.collection_name}'.")

    def delete_document_points(self, document_id_or_source: str) -> bool:
        """
        Deletes all vector points (dense + sparse) associated with specified document_id, doc_id, or source filename.
        """
        try:
            # Delete points matching document_id, doc_id, or source filename
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        should=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id_or_source)
                            ),
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=document_id_or_source)
                            ),
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=document_id_or_source)
                            )
                        ]
                    )
                )
            )
            print(f"🗑️ Successfully deleted Qdrant points for '{document_id_or_source}'")
            return True
        except Exception as e:
            print(f"❌ Failed to delete Qdrant points for '{document_id_or_source}': {e}")
            raise e

    def clear_all_points(self) -> bool:
        """Recreates and clears all vector points from Qdrant collection."""
        try:
            print(f"🧹 Clearing all vector points from Qdrant collection '{self.collection_name}'...")
            self.init_collection(force_recreate=True)
            return True
        except Exception as e:
            print(f"❌ Failed to clear Qdrant collection '{self.collection_name}': {e}")
            raise e





    def search_dense(self, query_vector: List[float], limit: int = 20) -> List[Dict[str, Any]]:
        """Dense Semantic Search against Qdrant Cloud."""
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit
        ).points
        return self._format_results(search_results)

    def search_bm25(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """BM25 Keyword Search against Qdrant Cloud using Sparse Vectors."""
        sparse_vec = list(self.sparse_model.embed([query_text]))[0]
        qdrant_sparse = SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
        
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=qdrant_sparse,
            using="sparse",
            limit=limit
        ).points
        return self._format_results(search_results)

    def search_hybrid(
        self,
        query_text: str,
        query_dense_vector: List[float],
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        rrf_k: int = 60,
        top_k_rrf: int = 20,
        top_k_final: int = 5,
        reranker_model: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Production Hybrid Search combining Dense + BM25 via Reciprocal Rank Fusion (RRF),
        followed by an optional Cross-Encoder Reranker stage.
        """
        dense_hits = self.search_dense(query_dense_vector, limit=top_k_dense)
        bm25_hits = self.search_bm25(query_text, limit=top_k_bm25)

        # Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        # Process Dense Ranks
        for rank, hit in enumerate(dense_hits, 1):
            doc_id = hit["id"]
            doc_store[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Process BM25 Ranks
        for rank, hit in enumerate(bm25_hits, 1):
            doc_id = hit["id"]
            if doc_id not in doc_store:
                doc_store[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

        # Sort documents by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        rrf_candidates = []
        for doc_id in sorted_ids[:top_k_rrf]:
            item = doc_store[doc_id].copy()
            item["rrf_score"] = rrf_scores[doc_id]
            rrf_candidates.append(item)

        # Apply Cross-Encoder Reranking if model provided
        if reranker_model and rrf_candidates:
            pairs = [[query_text, doc["content"]] for doc in rrf_candidates]
            try:
                # Predict cross-encoder relevance scores
                rerank_scores = reranker_model.predict(pairs)
                for idx, doc in enumerate(rrf_candidates):
                    # Handle numpy float conversion if needed
                    doc["rerank_score"] = float(rerank_scores[idx])
                
                # Sort descending by Cross-Encoder score
                rrf_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                print(f"⚠️ Reranker model failed during prediction: {e}. Falling back to RRF ordering.")

        return rrf_candidates[:top_k_final]

    def _format_results(self, points) -> List[Dict[str, Any]]:
        results = []
        for point in points:
            text_content = point.payload.get("content") or point.payload.get("text") or ""
            results.append({
                "id": str(point.id),
                "score": point.score,
                "content": text_content,
                "source": point.payload.get("source"),
                "page": point.payload.get("page"),
                "chunk_index": point.payload.get("chunk_index"),
            })
        return results

    def get_info(self) -> Dict[str, Any]:
        """Returns collection info and vector counts."""
        info = self.client.get_collection(self.collection_name)
        count = getattr(info, 'points_count', None) or getattr(info, 'vectors_count', 0)
        return {
            "name": self.collection_name,
            "vectors_count": count,
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
        }

if __name__ == "__main__":
    manager = QdrantManager()
    info = manager.get_info()
    print(f"Production Qdrant Manager Initialized: {info}")
