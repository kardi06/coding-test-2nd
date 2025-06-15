from typing import List, Dict, Any
from langchain.schema import Document
from services.vector_store import VectorStoreService
from config import settings
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, vector_store: VectorStoreService):
        # TODO: Initialize RAG pipeline components
        # - Vector store service
        # - LLM client
        # - Prompt templates
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=settings.llm_model, 
            openai_api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens
        )
    
    def generate_answer(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate answer using RAG pipeline"""
        # TODO: Implement RAG pipeline
        # 1. Retrieve relevant documents
        docs_with_scores = self.vector_store.similarity_search(question,k=settings.retrieval_k)
        docs = [d[0] for d in docs_with_scores if d[1] < settings.similarity_threshold]
        # 2. Generate context from retrieved documents
        context = "\n\n".join([f"(page {doc.metadata.get('page', '?')}): {doc.page_content}" for doc in docs])

        # build chat history
        chat_context = ""
        if chat_history and len(chat_history) > 0:
            # Take last 3 messages for context (to avoid token limit)
            recent_messages = chat_history[-3:]
            chat_lines = []
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    chat_lines.append(f"User: {content}")
                elif role == "assistant":
                    chat_lines.append(f"Assistant: {content}")
            if chat_lines:
                chat_context = f"Previous conversation:\n{chr(10).join(chat_lines)}\n\n"
        prompt = (
            "You are an expert assistant for financial documents.\n"
            f"{chat_context}"
            f"Context:\n{context}\n"
            f"Question: {question}\n"
            "Answer based on the document context above. If the question refers to previous conversation, "
            "use that context appropriately. Always cite page numbers when referencing document content:"
        )
        # 3. Generate answer using LLM
        answer = self.llm.invoke(prompt).content
        sources = [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "?"),
                "score": score
            }
            for doc, score in docs_with_scores
        ]
        # 4. Return answer with sources
        return {
            "answer": answer,
            "sources": sources
        }
        
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for the query"""
        # TODO: Implement document retrieval
        # - Search vector store for similar documents
        # - Filter by similarity threshold
        # - Return top-k documents
        pass
    
    def _generate_context(self, documents: List[Document]) -> str:
        """Generate context from retrieved documents"""
        # TODO: Generate context string from documents
        pass
    
    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using LLM"""
        # TODO: Implement LLM response generation
        # - Create prompt with question and context
        # - Call LLM API
        # - Return generated response
        pass 