import os
import hashlib
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

load_dotenv()

class DocumentProcessor:
    """Handles loading PDFs, Markdown, and text files, and splitting them into structured text chunks."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", "600"))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "100"))
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

        headers_to_split_on = [
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )

    def load_pdf(self, file_path: str, document_id: str = None) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF file using PyMuPDF (fitz) or pypdf fallback."""
        pages = []
        filename = os.path.basename(file_path)
        doc_id = document_id or f"doc_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:12]}"

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if text:
                    pages.append({
                        "content": text,
                        "metadata": {
                            "source": filename,
                            "file_path": file_path,
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "doc_id": doc_id,
                            "document_id": doc_id,
                        }
                    })
            doc.close()
        except ImportError:
            # Fallback to pypdf
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    pages.append({
                        "content": text,
                        "metadata": {
                            "source": filename,
                            "file_path": file_path,
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "doc_id": doc_id,
                            "document_id": doc_id,
                        }
                    })
        return pages

    def load_markdown_or_text(self, file_path: str, document_id: str = None) -> List[Dict[str, Any]]:
        """Reads text from a Markdown (.md) or Text (.txt) file."""
        filename = os.path.basename(file_path)
        doc_id = document_id or f"doc_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:12]}"
        
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        
        return [{
            "content": content,
            "metadata": {
                "source": filename,
                "file_path": file_path,
                "page": 1,
                "total_pages": 1,
                "doc_id": doc_id,
                "document_id": doc_id,
            }
        }]

    def load_document(self, file_path: str, document_id: str = None) -> List[Dict[str, Any]]:
        """Determines file type and loads document into page-level segments."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(file_path, document_id=document_id)
        elif ext in [".md", ".txt", ".markdown"]:
            return self.load_markdown_or_text(file_path, document_id=document_id)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf, .md, and .txt are supported.")

    def chunk_document(self, file_path: str, document_id: str = None) -> List[Dict[str, Any]]:
        """Loads a document and splits its text into header-aware, overlap-bounded chunks with metadata."""
        docs = self.load_document(file_path, document_id=document_id)
        ext = os.path.splitext(file_path)[1].lower()
        chunks = []

        chunk_counter = 0
        for doc in docs:
            # Header-aware chunking for markdown files
            if ext in [".md", ".markdown"]:
                md_splits = self.md_splitter.split_text(doc["content"])
                for md_doc in md_splits:
                    section_content = md_doc.page_content.strip()
                    if not section_content:
                        continue
                    
                    # Sub-split long section if it exceeds chunk_size
                    if len(section_content) > self.chunk_size:
                        sub_splits = self.splitter.split_text(section_content)
                        header_context = " > ".join([v for k, v in md_doc.metadata.items() if k in ["H1", "H2", "H3"]])
                        for sub in sub_splits:
                            if not sub.strip():
                                continue
                            
                            # Prefix sub-chunk with section header context if needed
                            final_text = sub
                            if header_context and not sub.startswith("#"):
                                final_text = f"[{header_context}]\n{sub}"

                            chunk_meta = doc["metadata"].copy()
                            chunk_meta["chunk_index"] = chunk_counter
                            chunk_meta.update(md_doc.metadata)
                            chunks.append({
                                "content": final_text,
                                "metadata": chunk_meta
                            })
                            chunk_counter += 1
                    else:
                        chunk_meta = doc["metadata"].copy()
                        chunk_meta["chunk_index"] = chunk_counter
                        chunk_meta.update(md_doc.metadata)
                        chunks.append({
                            "content": section_content,
                            "metadata": chunk_meta
                        })
                        chunk_counter += 1
            else:
                # Standard splitting for pdf/txt
                text_splits = self.splitter.split_text(doc["content"])
                for split_text in text_splits:
                    if not split_text.strip():
                        continue
                    
                    chunk_meta = doc["metadata"].copy()
                    chunk_meta["chunk_index"] = chunk_counter
                    chunks.append({
                        "content": split_text,
                        "metadata": chunk_meta
                    })
                    chunk_counter += 1

        # Set total_chunks for each chunk metadata
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)

        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor()
    print("DocumentProcessor initialized (chunk_size=600, overlap=100).")
