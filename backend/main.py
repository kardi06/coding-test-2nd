from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models.schemas import ChatRequest, ChatResponse, DocumentsResponse, UploadResponse
from services.pdf_processor import PDFProcessor
from services.vector_store import VectorStoreService
from services.rag_pipeline import RAGPipeline
from config import settings
import logging
import time
import os

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-based Financial Statement Q&A System",
    description="AI-powered Q&A system for financial documents using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
# TODO: Initialize your services here
pdf_processor = None
vector_store = None
rag_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # TODO: Initialize your services
    global pdf_processor, vector_store, rag_pipeline
    pdf_processor = PDFProcessor()
    vector_store = VectorStoreService()
    rag_pipeline = RAGPipeline(vector_store=vector_store)
    logger.info("Starting RAG Q&A System...")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG-based Financial Statement Q&A System is running"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file"""
    # TODO: Implement PDF upload and processing
    # 1. Validate file type (PDF)
    # 2. Save uploaded file
    # 3. Process PDF and extract text
    # 4. Store documents in vector database
    # 5. Return processing results
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    file_location = os.path.join(settings.pdf_upload_path, file.filename)
    os.makedirs(settings.pdf_upload_path, exist_ok=True)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Process PDF
    documents = pdf_processor.process_pdf(file_location)
    
    # Store documents in vector database
    vector_store.add_documents(documents)
    
    return {"filename": file.filename, "chunk_count": len(documents), "status": "processed" }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process chat request and return AI response"""
    # TODO: Implement chat functionality
    # 1. Validate request
    # 2. Use RAG pipeline to generate answer
    # 3. Return response with sources
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Use RAG pipeline to generate answer
    try:
        result = rag_pipeline.generate_answer(request.question, request.chat_history)
        return result
    except Exception as e:
        # logger.error(f"Error generating answer: {e}")
        logger.error("Error in RAG pipeline")
        # return JSONResponse(content={"error": str(e)}, status_code=500)
        raise HTTPException(status_code=500, detail="Failed to generate answer")


@app.get("/api/documents")
async def get_documents():
    """Get list of processed documents"""
    # TODO: Implement document listing
    # - Return list of uploaded and processed documents
    try:
        # Get document count from vector store
        document_count = vector_store.get_document_count()

        # For now, we'll return basic info since we don't store document metadata separately
        documents_info = []

        if document_count > 0:
            # Get all uploaded files from upload directory
            upload_dir = settings.pdf_upload_path
            if os.path.exists(upload_dir):
                for filename in os.listdir(upload_dir):
                    if filename.lower().endswith(".pdf"):
                        file_path = os.path.join(upload_dir, filename)
                        file_stat = os.stat(file_path)

                        documents_info.append({
                            "filename": filename,
                            "uploaded_date": file_stat.st_mtime,
                            "chunks_count": "unknown",
                            "status": "processed"
                        })
        return {
            "documents": documents_info,
            "total_count": len(documents_info)
            "total_chunks": document_count
        }
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrive documents list")
                        

@app.get("/api/chunks")
async def get_chunks():
    """Get document chunks (optional endpoint)"""
    # TODO: Implement chunk listing
    # - Return document chunks with metadata
    try:
        # Get all chunks from vector store
        # Note: This is a workaround since ChromaDB doesn't have direct "get all" method
        
        chunks_info = []
        
        # Try to get chunks using vector store collection directly
        if hasattr(vector_store.vector_store, '_collection'):
            try:
                # Get data from ChromaDB collection
                collection_data = vector_store.vector_store._collection.get(
                    limit=limit,
                    include=['documents', 'metadatas', 'ids']
                )
                
                # Process the chunks
                if collection_data and 'documents' in collection_data:
                    documents = collection_data.get('documents', [])
                    metadatas = collection_data.get('metadatas', [])
                    ids = collection_data.get('ids', [])
                    
                    for i, (doc_id, content, metadata) in enumerate(zip(ids, documents, metadatas)):
                        # Filter by page if specified
                        chunk_page = metadata.get('page', 'unknown')
                        if page is not None and chunk_page != page:
                            continue
                            
                        chunks_info.append({
                            "id": doc_id,
                            "content": content[:200] + "..." if len(content) > 200 else content,  # Preview first 200 chars
                            "full_content": content,
                            "page": chunk_page,
                            "metadata": metadata,
                            "content_length": len(content)
                        })
                
            except Exception as e:
                logger.warning(f"Could not access ChromaDB collection directly: {e}")
                # Fallback: return minimal info
                document_count = vector_store.get_document_count()
                return {
                    "chunks": [],
                    "total_count": document_count,
                    "message": "Chunks exist but cannot be retrieved directly. Total count available.",
                    "filter": {"page": page, "limit": limit}
                }
        
        # Apply limit after filtering
        if limit and len(chunks_info) > limit:
            chunks_info = chunks_info[:limit]
        
        return {
            "chunks": chunks_info,
            "total_count": len(chunks_info),
            "filter": {
                "page": page,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.debug) 