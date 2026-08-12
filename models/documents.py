from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentInfo(BaseModel):
    id: str = Field(..., description="Unique document ID (e.g. doc_12345678)")
    filename: str = Field(..., description="Original filename of the document")
    file_type: str = Field(..., description="File extension/type (pdf, txt, md)")
    size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of vector chunks generated")
    uploaded_at: str = Field(..., description="ISO formatted timestamp of upload")
    file_hash: Optional[str] = Field(None, description="SHA-256 hash of the file content")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "doc_191e9334f8b8",
                "filename": "TecClub_Technology_Website_Documentation__1_.pdf",
                "file_type": "pdf",
                "size": 37388,
                "chunk_count": 55,
                "uploaded_at": "2026-08-12T12:09:00Z",
                "file_hash": "50f509f612228d04bfb34b429170c2a7c0c790fc96a7001080b09bff0447dfc1"
            }
        }
    }


class DocumentUploadResponse(BaseModel):
    success: bool = True
    document: DocumentInfo

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class DocumentDeleteResponse(BaseModel):
    success: bool = True
    message: str
    document_id: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    existing_document_id: Optional[str] = None
