from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional

from models.documents import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    ErrorResponse
)
from services.document_service import DocumentIngestionService, DuplicateDocumentError

router = APIRouter(prefix="/api/documents", tags=["Documents"])

# Lazy singleton for DocumentIngestionService
_ingestion_service: Optional[DocumentIngestionService] = None

def get_ingestion_service() -> DocumentIngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService()
    return _ingestion_service

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request / Duplicate Document / Invalid File"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentIngestionService = Depends(get_ingestion_service)
):
    """
    Uploads a PDF, Markdown, or Text document into the RAG Knowledge Base.
    Generates dense & BM25 sparse vector embeddings and upserts chunks into Qdrant Cloud.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload request.")

    try:
        content_bytes = await file.read()
        doc_info = service.process_and_ingest_bytes(content_bytes, file.filename)
        return DocumentUploadResponse(success=True, document=doc_info)

    except DuplicateDocumentError as dup_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(dup_err),
                "existing_document_id": dup_err.existing_document_id
            }
        )
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": str(val_err)
            }
        )
    except Exception as exc:
        print(f"❌ Internal error processing uploaded file '{file.filename}': {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": f"Failed to ingest document: {str(exc)}"
            }
        )

@router.get("", response_model=DocumentListResponse)
def list_documents(service: DocumentIngestionService = Depends(get_ingestion_service)):
    """
    Lists all uploaded documents with summary metadata (ID, filename, size, chunk count, upload date).
    """
    docs = service.list_documents()
    return DocumentListResponse(documents=docs)

@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(
    document_id: str,
    service: DocumentIngestionService = Depends(get_ingestion_service)
):
    """
    Deletes an uploaded document by document_id or filename.
    Removes Qdrant vector chunks (dense & sparse), local files, and metadata records.
    """
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID or filename '{document_id}' not found."
        )

    return DocumentDeleteResponse(
        success=True,
        message=f"Document '{document_id}' and all associated vector chunks were successfully deleted.",
        document_id=document_id
    )

@router.delete("", response_model=DocumentDeleteResponse)
def clear_all_documents(service: DocumentIngestionService = Depends(get_ingestion_service)):
    """
    Wipes all Qdrant vector points, deletes local upload files, and resets metadata store.
    """
    service.clear_all_documents()
    return DocumentDeleteResponse(
        success=True,
        message="All Qdrant vector points, stored files, and metadata records have been completely cleared.",
        document_id="all"
    )

