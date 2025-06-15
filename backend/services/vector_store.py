from typing import List, Tuple, Dict, Any
from langchain.schema import Document
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import settings
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStoreService:
    def __init__(self):
        # TODO: Initialize vector store (ChromaDB, FAISS, etc.)
        # embedding using OPENAPI 
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model=settings.embedding_model
        )
        self.persist_path = settings.vector_db_path
        self.vector_store = Chroma(
            persist_directory=self.persist_path,
            embedding_function=self.embeddings,
            collection_name="documents"
        )
        
        # Initialize document metadata storage
        self.metadata_file = os.path.join(self.persist_path, "document_metadata.json")
        self._ensure_metadata_file()
    
    def _ensure_metadata_file(self):
        """Ensure metadata file exists"""
        os.makedirs(self.persist_path, exist_ok=True)
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from file"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save document metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def add_documents(self, documents: List[Document], filename: str = None) -> None:
        """Add documents to the vector store"""
        # TODO: Implement document addition to vector store
        # - Generate embeddings for documents
        # - Store documents with embeddings in vector database
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            
            # Update metadata if filename provided
            if filename:
                metadata = self._load_metadata()
                
                # Extract document info
                currencies_found = set()
                years_found = set()
                
                for doc in documents:
                    doc_currencies = doc.metadata.get('currencies_found', '')
                    if doc_currencies:
                        currencies_found.update([c.strip() for c in doc_currencies.split(',') if c.strip()])
                    
                    doc_years = doc.metadata.get('years_found', '')
                    if doc_years:
                        years_found.update([y.strip() for y in doc_years.split(',') if y.strip()])
                
                metadata[filename] = {
                    "upload_date": datetime.now().isoformat(),
                    "chunks_count": len(documents),
                    "currencies_found": sorted(list(currencies_found)),
                    "years_found": sorted(list(years_found)),
                    "status": "processed"
                }
                
                self._save_metadata(metadata)
            
            logger.info(f"Successfully added {len(documents)} documents for {filename}")
        except Exception as e:
            logger.error(f"Error to add documents: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        # TODO: Implement similarity search
        # - Generate embedding for query
        # - Search for similar documents in vector store
        # - Return documents with similarity scores
        k = k or settings.retrieval_k
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from vector store"""
        # TODO: Implement document deletion
        if not document_ids:
            logger.warning("No document IDs provided for deletion")
            return
        try:
            #check if vector store has delete method
            if hasattr(self.vector_store, "delete"):
                self.vector_store.delete(ids=document_ids)
            else:
                #Direct ChromaDB delete
                self.vector_store._collection.delete(ids=document_ids)
            
            logger.info(f"Successfully deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error to delete documents: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of documents in vector store"""
        # TODO: Return document count
        return self.vector_store._collection.count() 
    
    def get_documents_info(self) -> List[Dict[str, Any]]:
        """Get information about all processed documents"""
        metadata = self._load_metadata()
        documents_info = []
        
        for filename, info in metadata.items():
            documents_info.append({
                "filename": filename,
                "upload_date": info.get("upload_date"),
                "chunks_count": info.get("chunks_count", 0),
                "currencies_found": info.get("currencies_found", []),
                "years_found": info.get("years_found", []),
                "status": info.get("status", "unknown")
            })
        
        return documents_info 