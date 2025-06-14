# 🚀 RAG-based Financial Statement Q&A System

## 📋 Overview

A full-stack **Retrieval Augmented Generation (RAG)** application that enables intelligent Q&A on financial documents with **multi-currency** and **multi-year** support.

### 🎯 Key Features
- **🌍 Multi-Currency Support**: USD, EUR, KRW, IDR, JPY, and 29+ other currencies
- **📅 Multi-Year Analysis**: Dynamic year detection
- **🔍 Smart Document Processing**: Enhanced PDF parsing with financial data extraction
- **💬 Intelligent Chat Interface**: Real-time Q&A with source citations
- **📊 Financial Context Awareness**: Revenue, profit, growth, cash flow analysis
- **🎨 Modern UI/UX**: Beautiful Next.js frontend with loading states

### 🏗️ Architecture
```
Frontend (Next.js) ↔ Backend (FastAPI) ↔ Vector DB (ChromaDB) ↔ LLM (OpenAI)
```

---

## 🛠️ Tech Stack

### Backend
- **FastAPI**: High-performance Python web framework
- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: GPT-4 for generation + text-embedding-ada-002
- **pdfplumber**: Advanced PDF text extraction

### Frontend
- **Next.js 13+**: React framework with App Router
- **TypeScript**: Type-safe development
- **React Hooks**: State management
- **Axios**: HTTP client for API calls

---

## 📁 Project Structure

```
coding-test-2nd/
├── 🔧 backend/                    # FastAPI Backend
│   ├── main.py                   # FastAPI application entry point
│   ├── config.py                 # Configuration settings
│   ├── requirements.txt          # Python dependencies
│   ├── 📦 models/
│   │   └── schemas.py           # Pydantic data models
│   ├── 🔄 services/
│       ├── pdf_processor.py     # Multi-currency PDF processing
│       ├── vector_store.py      # ChromaDB integration
│       └── rag_pipeline.py      # Enhanced RAG pipeline         
├── 🎨 frontend/                   # Next.js Frontend
│   ├── pages/
│   │   ├── index.tsx            # Main chat interface
│   │   └── _app.tsx             # App configuration
│   ├── components/
│   │   ├── ChatInterface.tsx    # Chat UI component
│   │   └── FileUpload.tsx       # PDF upload component
│   ├── styles/
│   │   └── globals.css          # Global styles
│   ├── package.json             # Node.js dependencies
│   └── next.config.js           # Next.js configuration
├── 📄 data/                      # Sample documents
│   └── sample.pdf               # Financial statement sample
└── 📖 README.md                  # This documentation
```

---

## 🚀 Quick Start Guide

### 1️⃣ Prerequisites

**System Requirements:**
- **Python 3.8+** (recommended: 3.11)
- **Node.js 16+** (recommended: 18+)
- **Git** for version control
- **OpenAI API Key** ([Get here](https://platform.openai.com/api-keys))

**Check your versions:**
```bash
python --version    # Should be 3.8+
node --version      # Should be 16+
npm --version       # Should be 8+
```

### 2️⃣ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/kardi06/coding-test-2nd.git
cd coding-test-2nd

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3️⃣ Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create environment configuration
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "VECTOR_DB_PATH=./chroma_db" >> .env
echo "UPLOAD_DIR=./uploads" >> .env

# Create necessary directories
mkdir -p uploads chroma_db

# Start the backend server
python main.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 4️⃣ Frontend Setup

**Open a new terminal** and run:

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

**Expected output:**
```
ready - started server on 0.0.0.0:3000, url: http://localhost:3000
event - compiled client and server successfully in 2.3s (173 modules)
```

### 5️⃣ Access the Application

🎉 **Your application is now running!**

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

---

## 📚 API Documentation

### 🔄 Core Endpoints

#### **POST /api/upload**
Upload and process PDF documents

**Request:**
```bash
curl -X POST "http://localhost:8000/api/upload" \
     -F "file=@data/sample.pdf"
```

**Response:**
```json
{
  "message": "PDF processed successfully",
  "filename": "sample.pdf",
  "chunks_created": 45,
  "currencies_found": ["USD", "KRW", "EUR"],
  "years_found": ["2023", "2022", "2021"],
  "processing_time": 12.34
}
```

#### **POST /api/chat**
Ask questions about uploaded documents

**Request:**
```json
{
  "question": "What is the total revenue in USD for 2023?",
  "chat_history": []
}
```

**Response:**
```json
{
  "answer": "Based on the financial statement, the total revenue for 2023 was USD 15.2 billion...",
  "sources": [
    {
      "content": "Consolidated Revenue 2023: USD 15,200 million",
      "page": 3,
      "score": 0.85,
      "currencies_found": ["USD"],
      "years_found": ["2023"]
    }
  ],
  "question_type": "revenue",
  "data_availability": {"has_data": true}
}
```

#### **GET /api/documents**
List all processed documents with metadata

**Request:**
```bash
curl -X GET "http://localhost:8000/api/documents"
```

**Response:**
```json
{
  "documents": [
    {
      "filename": "AADI_01_2025.pdf",
      "upload_date": "2025-06-16T00:15:07.350599",
      "chunks_count": 45,
      "currencies_found": ["USD", "KRW", "EUR"],
      "years_found": ["2023", "2022", "2021"],
      "status": "processed"
    }
  ],
  "total_count": 1,
  "total_chunks": 45
}
```

#### **GET /api/chunks**
Retrieve document chunks with optional filtering (optional endpoint)

**Request:**
```bash
# Get all chunks
curl -X GET "http://localhost:8000/api/chunks"

# Get chunks with pagination
curl -X GET "http://localhost:8000/api/chunks?limit=10"

# Get chunks from specific page
curl -X GET "http://localhost:8000/api/chunks?page=3&limit=5"
```

**Query Parameters:**
- `page` (optional): Filter chunks by page number
- `limit` (optional): Limit number of chunks returned (default: 50)

**Response (when chunks can be retrieved):**
```json
{
  "chunks": [
    {
      "id": "chunk_001",
      "content": "Revenue Analysis for FY2023...",
      "full_content": "Complete chunk content here...",
      "page": 3,
      "metadata": {
        "page": 3,
        "content_type": "financial_summary",
        "has_financial_data": true,
        "currencies_found": "USD,KRW",
        "years_found": "2023"
      },
      "content_length": 1250
    }
  ],
  "total_count": 45,
  "filter": {
    "page": 3,
    "limit": 5
  }
}
```

**Response (when chunks cannot be retrieved directly):**
```json
{
  "chunks": [],
  "total_count": 1674,
  "message": "Chunks exist but cannot be retrieved directly. Total count available.",
  "filter": {
    "page": null,
    "limit": 50
  }
}
```

**Note:** Due to ChromaDB access limitations, chunks may not always be retrievable directly. The endpoint will return the total count and indicate when chunks exist but cannot be accessed.

---

## 💬 Sample Questions

The system can handle complex financial queries in multiple currencies and years:

```
✅ "What is the total revenue for 2025?"
✅ "What is the year-over-year operating profit growth rate?"
✅ "What are the main cost items?"
✅ "How is the cash flow situation?"
✅ "What is the debt ratio?"
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in the `backend/` directory:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional (with defaults)
VECTOR_DB_PATH=./vector_store
PDF_UPLOAD_PATH=../data
VECTOR_DB_TYPE=chromadb
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=1.0
ALLOWED_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
```

---


## 🧪 Testing

### Manual Testing

1. **Upload PDF**:
   ```bash
   curl -X POST "http://localhost:8000/api/upload" \
        -F "file=@data/sample.pdf"
   ```

2. **Test Chat**:
   ```bash
   curl -X POST "http://localhost:8000/api/chat" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is the total revenue for 2023?"}'
   ```

3. **Check Documents**:
   ```bash
   curl "http://localhost:8000/api/documents"
   ```
