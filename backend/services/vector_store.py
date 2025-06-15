from typing import List, Tuple
from langchain.schema import Document
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config import settings
import logging

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
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        # TODO: Implement document addition to vector store
        # - Generate embeddings for documents
        # - Store documents with embeddings in vector database
        try:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            logger.info(f"Successfully added {len(documents)} documents")
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