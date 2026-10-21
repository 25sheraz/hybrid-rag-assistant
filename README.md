# Dual-Vector Hybrid RAG & Dynamic Document Management

![Knowledge RAG and Document Manager: upload PDFs, ask questions and get source-backed answers](docs/cover.jpg)

<p align="center">
  <strong>Production-grade Enterprise Retrieval-Augmented Generation (RAG) System with Dual-Vector Search, Cross-Encoder Reranking, Dynamic Document Management, Redis Session Memory, and Embedded React UI.</strong>
</p>

---

## 🚀 Overview

This system is an end-to-end, enterprise-ready AI Chat & Knowledge Retrieval System. It combines dense semantic embeddings with sparse BM25 keyword matching via **Reciprocal Rank Fusion (RRF)**, re-ranks candidate matches with high-precision Cross-Encoders, leverages **Redis Cloud** for multi-turn conversation memory with dynamic query reformulation, and provides a full-featured **Dynamic Document Management System**.

Whether deployed as a FastAPI backend, embedded into web applications via the included React components, or executed interactively in the terminal, it delivers fast, contextually aware responses grounded strictly in your uploaded knowledge base.

---

## ⚡ Key Features

- **🔎 Dual-Vector Hybrid Search**: Combines Dense Vector Search (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and Sparse BM25 Search inside **Qdrant**.
- **🔀 Reciprocal Rank Fusion (RRF)**: Merges dense vector proximity scores and sparse lexical ranking to capture both semantic intent and exact keyphrase matches.
- **🎯 Cross-Encoder Reranking**: Re-scores top RRF candidate document chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2` / `BAAI/bge-reranker-v2-m3` to eliminate irrelevant context before LLM generation.
- **🧠 Multi-Turn Redis Session Memory**: Tracks conversation turns across user sessions (`RedisMemoryManager`) with sliding window limits and TTL expiration, alongside automatic local in-memory fallback.
- **🔄 Intelligent Query Rewriting**: Reformulates follow-up user queries (`QueryRewriter`) into standalone search statements using past turn context.
- **🌐 FastAPI REST Server & Modular Architecture**: Clean, enterprise-grade project layout with modular routers (`routers/documents.py`), services (`services/document_service.py`), and Pydantic data schemas (`models/documents.py`).
- **📁 Dynamic Document Management**: Real-time PDF upload, listing, document details, and deletion API (`POST /api/documents/upload`, `GET /api/documents`, `DELETE /api/documents/{document_id}`) with SHA-256 duplicate detection and automatic Qdrant vector point cleanup.
- **⚛️ Interactive React UI Components**: Built-in React/TypeScript UI (`DocumentManager.tsx`, `DocumentUpload.tsx`, `DocumentList.tsx`) and floating chat widget (`KnowledgeRAGChatWidget.tsx`) with drag-and-drop file upload, document tab management, and real-time status indicators.
- **📄 Shared Ingestion Pipeline**: Unified ingestion logic serving both live web API uploads and `ingest.py` batch CLI processing.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    User([User / Web UI / CLI]) -->|1. Upload Doc or Submit Query| API[FastAPI Server / Routers]
    
    subgraph Document Management
        API -->|Upload PDF| DocService[Document Management Service]
        DocService -->|SHA-256 Check & Metadata| MetaStore[(JSON Metadata Store)]
        DocService -->|Extract Chunks & Vectorize| IngestPipeline[Ingestion Service]
        IngestPipeline -->|Upsert Chunks| Qdrant[(Qdrant Cloud)]
    end

    subgraph Hybrid RAG Query Pipeline
        API -->|Submit Chat Query| Memory[Redis Session Memory]
        Memory -->|Fetch History| Rewriter[Query Rewriter]
        Rewriter -->|2. Standalone Query| Dense[Dense Embedding Model]
        
        Dense -->|3. Dense Vector| Qdrant
        Rewriter -->|3. BM25 Keywords| Qdrant
        
        Qdrant -->|4. Candidates| RRF[Reciprocal Rank Fusion]
        RRF -->|5. Top Candidates| Reranker[Cross-Encoder Reranker]
        Reranker -->|6. Top Chunks| Prompt[Context Builder]
        
        Prompt -->|7. Grounded Context| LLM[LLM Generator / OpenAI API]
        LLM -->|8. Answer| API
        API -->|9. Store Turn| Memory
        API -->|10. Response + Sources| User
    end
```

---

## 📋 Prerequisites

- **Python**: `3.10+`
- **Node.js**: `18.0+` & `npm`
- **Qdrant Cloud** vector database instance
- **Redis Cloud** (or local Redis instance)
- **LLM API Key** (OpenAI compatible endpoint, e.g., GPT-4o-mini / GPT-5)

---

## 📥 Installation & Setup

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/25sheraz/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create and activate Python virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install Python backend dependencies
pip install -r requirements.txt

# Install React frontend dependencies
npm install
```

---

## ⚙️ Environment Configuration (`.env`)

Create a `.env` file in the root directory containing your service credentials and model configurations:

```ini
# Qdrant Vector DB Configuration
QDRANT_URL=https://your-qdrant-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_documents

# Redis Session Memory Configuration
REDIS_URL=redis://default:password@your-redis-host:port
MEMORY_TURNS=5
MEMORY_TTL_SECONDS=1800

# Embedding & Reranker Models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# LLM Endpoint Configuration
LLM_API_KEY=your_openai_or_compatible_api_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Search & Retrieval Parameters
TOP_K_DENSE=20
TOP_K_BM25=20
RRF_K=60
TOP_K_RRF=20
TOP_K_FINAL=5

# Document Chunking Settings
CHUNK_SIZE=600
CHUNK_OVERLAP=100
```

---

## 🚀 Usage Guide

### 1. Ingest Documents into Vector Store (Batch CLI)

Ingest single files or entire folders of PDF, Markdown (`.md`), or Text (`.txt`) documents into Qdrant:

```bash
# Ingest default knowledge base document
python ingest.py --file path/to/document.pdf

# Ingest an entire directory (and optionally recreate collection)
python ingest.py --dir path/to/docs --recreate
```

### 2. Launch FastAPI Server & Web App

Run the server to expose the REST API and serve the React interface:

```bash
# Start FastAPI backend server on port 8000
python server.py
```

Access the application:
- **Web Interface**: `http://localhost:8000`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

For frontend hot-reloading during development:
```bash
npm run dev
```

### 3. Interactive CLI Assistant

Interact with the RAG pipeline directly inside your terminal:

```bash
python cli.py --session my_terminal_user --sources
```
*CLI Commands:*
- `/clear` — Reset session conversation memory.
- `/sources` — Toggle output of retrieved document chunks.
- `exit` — Exit the assistant.

### 4. Run Verification Suite

```bash
# Verify environment setup and connections
python test_env.py

# Test hybrid search & reranking isolation
python test_hybrid_search.py

# Test dynamic document upload, indexing & deletion lifecycle
python test_document_management.py

# Run end-to-end multi-turn scenario tests
python test_full_pipeline.py
```

---

## 🔌 API Reference

### Document Management Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload PDF file, extract text, chunk, and index into Qdrant |
| `GET` | `/api/documents` | List all ingested documents with metadata, size, chunk counts |
| `GET` | `/api/documents/{document_id}` | Retrieve metadata and chunk details for a specific document |
| `DELETE` | `/api/documents/{document_id}` | Delete document file, metadata, and remove vectors from Qdrant |

### RAG Chat Endpoint

`POST /api/chat`

#### Request Body
```json
{
  "query": "What key features are supported in this document store?",
  "session_id": "web_user_123"
}
```

#### Response Body
```json
{
  "query": "What key features are supported in this document store?",
  "rewritten_query": "What key features and capabilities are supported in this document store?",
  "response": "The system provides custom AI capabilities including enterprise RAG systems, dynamic document management...",
  "sources": [
    {
      "source": "Sample_Knowledge_Base.pdf",
      "page": 1
    }
  ]
}
```

---

## 📁 Repository Structure

```
PDF-Uploader/
├── server.py                  # FastAPI REST API & static file server
├── cli.py                     # Terminal interactive assistant
├── ingest.py                  # Document batch ingestion CLI tool
├── document_processor.py      # PDF/MD/TXT reader & chunker
├── qdrant_manager.py          # Qdrant client, sparse/dense indexing & RRF search
├── rag_pipeline.py            # Redis memory, query rewriter, context builder & LLM generator
│
├── models/                    # Pydantic data schemas
│   └── documents.py           # Document upload & metadata request/response models
├── routers/                   # API route handlers
│   └── documents.py           # REST endpoints for upload, list, delete documents
├── services/                  # Business logic services
│   └── document_service.py    # Document management & vector cleanup operations
│
├── KnowledgeRAGChatWidget.tsx # Embedded React UI Chat Widget component
├── DocumentManager.tsx        # React Document Management interface
├── DocumentUpload.tsx         # Drag-and-drop document uploader component
├── DocumentList.tsx           # Document list view & delete modal component
│
├── index.html                 # Web app entry point
├── package.json               # Node.js dependencies & build scripts
├── tsconfig.json              # TypeScript configuration
│
├── test_env.py                # Environment & connectivity test
├── test_hybrid_search.py      # Search & RRF test script
├── test_document_management.py# Dynamic document management lifecycle test
└── test_full_pipeline.py      # Multi-turn end-to-end pipeline test
```

---

## 📄 License

This project is open-source and available under standard open-source terms. All rights reserved.

