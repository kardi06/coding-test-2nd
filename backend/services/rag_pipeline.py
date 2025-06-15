from typing import List, Dict, Any, Tuple
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
        try:
            # 1. Retrieve documents using helper
            docs_with_scores = self._retrieve_documents(question)
            
            # 2. Filter for context generation
            filtered_docs = [doc for doc, score in docs_with_scores if score < settings.similarity_threshold]
            
            # 3. Generate context from documents
            context = self._generate_context(filtered_docs)
            
            # 4. Generate answer
            answer = self._generate_llm_response(question, context, chat_history)
            
            # 5. Prepare sources for frontend
            sources = [
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "score": score
                }
                for doc, score in docs_with_scores
            ]
            
            # 6. Return complete response
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": []
            }
        
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for the query"""
        # TODO: Implement document retrieval
        # - Search vector store for similar documents
        # - Filter by similarity threshold
        # - Return top-k documents
        try:
            # Get documents with scores from vector store
            docs_with_scores = self.vector_store.similarity_search(query, k=settings.retrieval_k)
            logger.info(f"Retrieved {len(docs_with_scores)} documents")
            return docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _generate_context(self, documents: List[Document]) -> str:
        """Generate context from retrieved documents"""
        # TODO: Generate context string from documents
        if not documents:
            return "No relevant documents found."
        
        # Build context with page references
        context_parts = []
        for doc in documents:
            page_info = doc.metadata.get('page', '?')
            content = doc.page_content.strip()
            if content:  # Only add non-empty content
                context_parts.append(f"[Page {page_info}]: {content}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context from {len(documents)} documents ({len(context)} characters)")
        return context
    
    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using LLM"""
        # TODO: Implement LLM response generation
        # - Create prompt with question and context
        # - Call LLM API
        # - Return generated response
        try:
            # Build chat history context
            chat_context = ""
            if chat_history and len(chat_history) > 0:
                recent_messages = chat_history[-3:]  # Last 3 messages
                chat_lines = []
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if role == 'user':
                        chat_lines.append(f"User: {content}")
                    elif role == 'assistant':
                        chat_lines.append(f"Assistant: {content}")
                
                if chat_lines:
                    chat_context = f"Previous conversation:\n{chr(10).join(chat_lines)}\n\n"

            # Create prompt
            prompt = (
                "You are an expert assistant for financial documents.\n"
                f"{chat_context}"
                f"Context from documents:\n{context}\n\n"
                f"Current question: {question}\n"
                "Answer based on the document context above. If the question refers to previous conversation, "
                "use that context appropriately. Always cite page numbers when referencing document content:"
            )
            
            # Call LLM
            response = self.llm.invoke(prompt)
            answer = response.content
            
            logger.info(f"Generated LLM response ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again." 