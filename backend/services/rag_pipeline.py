from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from services.vector_store import VectorStoreService
from config import settings
from langchain_openai import ChatOpenAI
import logging
import re

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
        
        # Initialize multi-currency and multi-year patterns
        self._init_query_enhancement_patterns()
    
    def _init_query_enhancement_patterns(self):
        """Initialize patterns for multi-currency and multi-year query enhancement"""
        
        # Currency mapping for query enhancement
        self.currency_mappings = {
            'usd': ['USD', 'dollar', '$', 'US dollar', 'american dollar'],
            'eur': ['EUR', 'euro', '€', 'european euro'],
            'jpy': ['JPY', 'yen', '¥', 'japanese yen'],
            'krw': ['KRW', 'won', '₩', 'korean won'],
            'idr': ['IDR', 'rupiah', 'Rp', 'indonesian rupiah'],
            'cny': ['CNY', 'yuan', 'rmb', 'chinese yuan'],
            'sgd': ['SGD', 'S$', 'singapore dollar'],
            'thb': ['THB', 'baht', 'thai baht'],
            'myr': ['MYR', 'ringgit', 'malaysian ringgit']
        }
        
        # Financial terms in multiple languages
        self.financial_terms_mappings = {
            'revenue': ['revenue', 'sales', 'pendapatan', 'penjualan', 'total revenue', 'consolidated revenue'],
            'profit': ['profit', 'income', 'earnings', 'keuntungan', 'laba', 'operating profit', 'net income'],
            'loss': ['loss', 'kerugian', 'rugi', 'operating loss', 'net loss'],
            'growth': ['growth', 'pertumbuhan', 'growth rate', 'year-over-year', 'yoy'],
            'cost': ['cost', 'expense', 'biaya', 'pengeluaran', 'operating cost'],
            'debt': ['debt', 'utang', 'liability', 'kewajiban', 'debt ratio'],
            'cash': ['cash', 'kas', 'cash flow', 'arus kas', 'liquidity']
        }
        
        # Year context patterns
        self.year_context_patterns = [
            r'\b(20\d{2})\b',                           # 2023
            r'FY\s*(\d{4})',                            # FY2023
            r'fiscal year\s*(\d{4})',                   # fiscal year 2023
            r'(\d{4})\s*vs\s*(\d{4})',                  # 2023 vs 2022
            r'for\s*(\d{4})',                           # for 2023
            r'in\s*(\d{4})',                            # in 2023
            r'tahun\s*(\d{4})',                         # tahun 2023 (Indonesian)
        ]
    
    def generate_answer(self, question: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate answer using RAG pipeline"""
        try:
            # 1. Analyze the question type
            question_type = self._analyze_question_type(question)
            
            # 2. Extract currency and year from question
            question_metadata = self._extract_question_metadata(question)
            
            # 3. Retrieve documents using enhanced retrieval
            docs_with_scores = self._retrieve_documents(question, question_type, question_metadata)
            
            # 4. Filter for context generation
            filtered_docs = [doc for doc, score in docs_with_scores if score < settings.similarity_threshold]
            
            # 5. Check if we have relevant financial data
            has_relevant_data = self._check_data_availability(filtered_docs, question_type, question_metadata)
            
            # 6. Generate context from documents
            context = self._generate_context(filtered_docs, question_type, question_metadata)
            
            # 7. Generate answer with enhanced prompting
            answer = self._generate_llm_response(question, context, chat_history, question_type, has_relevant_data, question_metadata)
            
            # 8. Prepare sources for frontend
            sources = [
                {
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "score": score,
                    "content_type": doc.metadata.get("content_type", "general"),
                    "has_financial_data": doc.metadata.get("has_financial_data", False),
                    "currencies_found": [c.strip() for c in doc.metadata.get("currencies_found", "").split(',') if c.strip()],
                    "years_found": [y.strip() for y in doc.metadata.get("years_found", "").split(',') if y.strip()]
                }
                for doc, score in docs_with_scores
            ]
            
            # 9. Return complete response
            return {
                "answer": answer,
                "sources": sources,
                "question_type": question_type,
                "data_availability": has_relevant_data,
                "question_metadata": question_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "question_type": "unknown",
                "data_availability": False,
                "question_metadata": {}
            }
    
    def _extract_question_metadata(self, question: str) -> Dict[str, Any]:
        """Extract currency, year, and financial terms from the question"""
        metadata = {
            'currencies_mentioned': [],
            'years_mentioned': [],
            'financial_terms': [],
            'specific_amounts': []
        }
        
        question_lower = question.lower()
        
        # Extract currencies mentioned in question
        for currency_key, currency_variants in self.currency_mappings.items():
            for variant in currency_variants:
                if variant.lower() in question_lower:
                    metadata['currencies_mentioned'].append(currency_key.upper())
                    break
        
        # Extract years mentioned in question
        for pattern in self.year_context_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    metadata['years_mentioned'].extend(match)
                else:
                    metadata['years_mentioned'].append(match)
        
        # Extract financial terms
        for term_key, term_variants in self.financial_terms_mappings.items():
            for variant in term_variants:
                if variant.lower() in question_lower:
                    metadata['financial_terms'].append(term_key)
                    break
        
        # Remove duplicates and sort
        metadata['currencies_mentioned'] = list(set(metadata['currencies_mentioned']))
        metadata['years_mentioned'] = sorted(list(set(metadata['years_mentioned'])))
        metadata['financial_terms'] = list(set(metadata['financial_terms']))
        
        return metadata
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze what type of financial information is being requested"""
        question_lower = question.lower()
        
        # Revenue-related questions
        if any(keyword in question_lower for keyword in ['revenue', 'sales', 'total revenue', 'consolidated revenue', 'pendapatan']):
            return 'revenue'
        
        # Profit/income questions
        elif any(keyword in question_lower for keyword in ['profit', 'income', 'earnings', 'operating profit', 'net income', 'loss', 'laba', 'keuntungan']):
            return 'profit_loss'
        
        # Growth rate questions
        elif any(keyword in question_lower for keyword in ['growth', 'growth rate', 'year-over-year', 'yoy', 'pertumbuhan']):
            return 'growth_analysis'
        
        # Cost-related questions
        elif any(keyword in question_lower for keyword in ['cost', 'expense', 'operating cost', 'main cost', 'biaya']):
            return 'cost_analysis'
        
        # Cash flow questions
        elif any(keyword in question_lower for keyword in ['cash flow', 'cash', 'liquidity', 'kas', 'arus kas']):
            return 'cash_flow'
        
        # Debt/ratio questions
        elif any(keyword in question_lower for keyword in ['debt', 'ratio', 'debt ratio', 'leverage', 'utang']):
            return 'financial_ratios'
        
        # General financial questions
        elif any(keyword in question_lower for keyword in ['financial', 'statement', 'performance', 'keuangan']):
            return 'general_financial'
        
        else:
            return 'general'
    
    def _retrieve_documents(self, query: str, question_type: str, question_metadata: Dict[str, Any]) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents for the query"""
        # TODO: Implement document retrieval
        # - Search vector store for similar documents
        # - Filter by similarity threshold
        # - Return top-k documents
        try:
            # Enhance query based on question type and metadata
            enhanced_query = self._enhance_query_by_type_and_metadata(query, question_type, question_metadata)
            
            # Get documents with scores from vector store
            docs_with_scores = self.vector_store.similarity_search(enhanced_query, k=settings.retrieval_k)
            
            # Filter documents based on metadata matching
            filtered_docs_with_scores = self._filter_by_metadata_relevance(docs_with_scores, question_metadata)
            
            logger.info(f"Retrieved {len(docs_with_scores)} documents for {question_type} query: '{enhanced_query}'")
            logger.info(f"Filtered to {len(filtered_docs_with_scores)} documents based on metadata matching")
            
            # Log retrieval details for debugging
            for i, (doc, score) in enumerate(filtered_docs_with_scores[:5]):  # Show top 5
                content_type = doc.metadata.get('content_type', 'unknown')
                page = doc.metadata.get('page', '?')
                has_financial = doc.metadata.get('has_financial_data', False)
                
                # Convert string metadata back to lists for logging
                currencies_str = doc.metadata.get('currencies_found', '')
                currencies = [c.strip() for c in currencies_str.split(',') if c.strip()] if currencies_str else []
                
                years_str = doc.metadata.get('years_found', '')
                years = [y.strip() for y in years_str.split(',') if y.strip()] if years_str else []
                
                logger.info(f"  Result {i+1}: Page {page}, Type: {content_type}, Financial: {has_financial}, Currencies: {currencies}, Years: {years}, Score: {score:.4f}")
            
            return filtered_docs_with_scores
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _enhance_query_by_type_and_metadata(self, query: str, question_type: str, question_metadata: Dict[str, Any]) -> str:
        """Enhance query based on question type and extracted metadata"""
        base_query = query
        
        # Base enhancement mappings
        enhancement_mappings = {
            'revenue': 'consolidated revenue sales total revenue financial statements income statement',
            'profit_loss': 'operating profit operating income net income earnings loss profit financial performance',
            'growth_analysis': 'year-over-year growth rate percentage change financial performance comparison',
            'cost_analysis': 'operating expenses cost structure main costs operating cost',
            'cash_flow': 'cash flow operating cash free cash flow liquidity',
            'financial_ratios': 'debt ratio financial ratios leverage debt equity ratios',
            'general_financial': 'financial statements consolidated financial data performance'
        }
        
        # Start with base enhancement
        enhanced_query = f"{base_query} {enhancement_mappings.get(question_type, '')}"
        
        # Add currency-specific enhancement
        if question_metadata['currencies_mentioned']:
            currency_terms = []
            for currency in question_metadata['currencies_mentioned']:
                if currency.upper() in ['USD', 'KRW', 'IDR', 'EUR', 'JPY']:
                    currency_terms.extend(self.currency_mappings.get(currency.lower(), [currency]))
            enhanced_query += f" {' '.join(currency_terms)}"
        
        # Add year-specific enhancement
        if question_metadata['years_mentioned']:
            years_str = ' '.join(question_metadata['years_mentioned'])
            enhanced_query += f" {years_str} fiscal year financial year"
        
        # Add financial term enhancement
        if question_metadata['financial_terms']:
            for term in question_metadata['financial_terms']:
                term_variants = self.financial_terms_mappings.get(term, [])
                enhanced_query += f" {' '.join(term_variants)}"
        
        return enhanced_query.strip()
    
    def _filter_by_metadata_relevance(self, docs_with_scores: List[Tuple[Document, float]], 
                                    question_metadata: Dict[str, Any]) -> List[Tuple[Document, float]]:
        """Filter documents based on metadata relevance to question"""
        if not question_metadata['currencies_mentioned'] and not question_metadata['years_mentioned']:
            return docs_with_scores  # No filtering needed
        
        relevant_docs = []
        
        for doc, score in docs_with_scores:
            relevance_boost = 0
            
            # Convert string metadata back to lists for processing
            doc_currencies_str = doc.metadata.get('currencies_found', '')
            doc_currencies = [c.strip() for c in doc_currencies_str.split(',') if c.strip()] if doc_currencies_str else []
            
            doc_years_str = doc.metadata.get('years_found', '')
            doc_years = [y.strip() for y in doc_years_str.split(',') if y.strip()] if doc_years_str else []
            
            # Check currency match
            if question_metadata['currencies_mentioned']:
                currency_match = any(
                    curr in [c.upper() for c in doc_currencies] 
                    for curr in question_metadata['currencies_mentioned']
                )
                if currency_match:
                    relevance_boost += 0.1  # Boost relevance
            
            # Check year match
            if question_metadata['years_mentioned']:
                year_match = any(
                    year in doc_years 
                    for year in question_metadata['years_mentioned']
                )
                if year_match:
                    relevance_boost += 0.1  # Boost relevance
            
            # Adjust score based on relevance
            adjusted_score = score - relevance_boost  # Lower score = higher relevance
            relevant_docs.append((doc, adjusted_score))
        
        # Sort by adjusted score and return
        relevant_docs.sort(key=lambda x: x[1])
        return relevant_docs
    
    def _check_data_availability(self, documents: List[Document], question_type: str, 
                               question_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check if retrieved documents contain the requested financial data"""
        if not documents:
            return {"has_data": False, "reason": "no_documents", "suggestion": "No relevant documents found"}
        
        # Check if any document has financial data
        has_financial_docs = any(doc.metadata.get('has_financial_data', False) for doc in documents)
        
        if not has_financial_docs:
            return {
                "has_data": False, 
                "reason": "no_financial_data", 
                "suggestion": "Documents found but contain governance/compliance information, not detailed financial statements"
            }
        
        # Check currency availability
        if question_metadata['currencies_mentioned']:
            available_currencies = set()
            for doc in documents:
                doc_currencies_str = doc.metadata.get('currencies_found', '')
                doc_currencies = [c.strip().upper() for c in doc_currencies_str.split(',') if c.strip()] if doc_currencies_str else []
                available_currencies.update(doc_currencies)
            
            requested_currencies = set(c.upper() for c in question_metadata['currencies_mentioned'])
            currency_match = bool(requested_currencies.intersection(available_currencies))
            
            if not currency_match:
                return {
                    "has_data": False,
                    "reason": "currency_mismatch",
                    "suggestion": f"Requested currencies {requested_currencies} not found. Available: {available_currencies}"
                }
        
        # Check year availability
        if question_metadata['years_mentioned']:
            available_years = set()
            for doc in documents:
                doc_years_str = doc.metadata.get('years_found', '')
                doc_years = [y.strip() for y in doc_years_str.split(',') if y.strip()] if doc_years_str else []
                available_years.update(doc_years)
            
            requested_years = set(question_metadata['years_mentioned'])
            year_match = bool(requested_years.intersection(available_years))
            
            if not year_match:
                return {
                    "has_data": False,
                    "reason": "year_mismatch", 
                    "suggestion": f"Requested years {requested_years} not found. Available: {available_years}"
                }
        
        # Check content type relevance
        content_types = [doc.metadata.get('content_type', 'general') for doc in documents]
        
        # Map question types to required content types
        required_content_mapping = {
            'revenue': ['revenue_statement', 'financial_summary'],
            'profit_loss': ['profit_statement', 'financial_summary'],
            'growth_analysis': ['revenue_statement', 'profit_statement', 'financial_summary'],
            'cost_analysis': ['profit_statement', 'financial_summary'],
            'cash_flow': ['cash_flow', 'financial_summary'],
            'financial_ratios': ['balance_sheet', 'financial_summary']
        }
        
        if question_type in required_content_mapping:
            required_types = required_content_mapping[question_type]
            has_required_content = any(ct in required_types for ct in content_types)
            
            if not has_required_content:
                return {
                    "has_data": False,
                    "reason": "wrong_document_type",
                    "suggestion": f"Found documents but they don't contain {question_type} data. Need {required_types} type documents."
                }
        
        return {"has_data": True, "reason": "data_available"}
    
    def _generate_context(self, documents: List[Document], question_type: str, question_metadata: Dict[str, Any]) -> str:
        """Generate enhanced context with question-type and metadata awareness"""
        if not documents:
            return "No relevant documents found."
        
        # Prioritize documents with financial data and metadata matches
        financial_docs = [doc for doc in documents if doc.metadata.get('has_financial_data', False)]
        other_docs = [doc for doc in documents if not doc.metadata.get('has_financial_data', False)]
        
        # Further prioritize by currency and year matches
        def get_priority_score(doc):
            score = 0
            
            # Convert string metadata back to lists
            doc_currencies_str = doc.metadata.get('currencies_found', '')
            doc_currencies = [c.strip().upper() for c in doc_currencies_str.split(',') if c.strip()] if doc_currencies_str else []
            
            doc_years_str = doc.metadata.get('years_found', '')
            doc_years = [y.strip() for y in doc_years_str.split(',') if y.strip()] if doc_years_str else []
            
            # Currency match bonus
            if question_metadata['currencies_mentioned']:
                if any(curr in doc_currencies for curr in question_metadata['currencies_mentioned']):
                    score += 2
            
            # Year match bonus
            if question_metadata['years_mentioned']:
                if any(year in doc_years for year in question_metadata['years_mentioned']):
                    score += 2
            
            return score
        
        # Sort financial docs by priority
        financial_docs.sort(key=get_priority_score, reverse=True)
        
        # Build context with prioritization
        context_parts = []
        
        # Add high-priority financial documents first
        for doc in financial_docs[:5]:  # Top 5 financial docs
            page_info = doc.metadata.get('page', '?')
            content_type = doc.metadata.get('content_type', 'general')
            
            # Convert string metadata back to lists for display
            currencies_str = doc.metadata.get('currencies_found', '')
            currencies = [c.strip() for c in currencies_str.split(',') if c.strip()] if currencies_str else []
            
            years_str = doc.metadata.get('years_found', '')
            years = [y.strip() for y in years_str.split(',') if y.strip()] if years_str else []
            
            content = doc.page_content.strip()
            
            if content:
                metadata_info = ""
                if currencies:
                    metadata_info += f"Currencies: {', '.join(currencies)}"
                if years:
                    metadata_info += f"{', ' if metadata_info else ''}Years: {', '.join(years)}"
                
                context_header = f"[Page {page_info} - {content_type.title()}"
                if metadata_info:
                    context_header += f" - {metadata_info}"
                context_header += "]"
                
                context_parts.append(f"{context_header}: {content}")
        
        # Add other documents if needed
        for doc in other_docs[:2]:  # Limit non-financial docs
            page_info = doc.metadata.get('page', '?')
            content_type = doc.metadata.get('content_type', 'general')
            content = doc.page_content.strip()
            if content:
                context_parts.append(f"[Page {page_info} - {content_type.title()}]: {content}")
        
        context = "\n\n".join(context_parts)
        logger.info(f"Generated context from {len(documents)} documents ({len(context)} characters)")
        return context
    
    def _generate_llm_response(self, question: str, context: str, chat_history: List[Dict[str, str]] = None, question_type: str = "general", data_availability: Dict[str, Any] = None, question_metadata: Dict[str, Any] = None) -> str:
        """Generate response using LLM"""
        # TODO: Implement LLM response generation
        # - Create prompt with question and context
        # - Call LLM API
        # - Return generated response
        try:
            # Build chat history context
            chat_context = ""
            if chat_history and len(chat_history) > 0:
                recent_messages = chat_history[-3:]
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

            # Create enhanced prompt based on data availability
            if data_availability and not data_availability.get("has_data", False):
                prompt = self._create_no_data_prompt(question, context, chat_context, question_type, data_availability, question_metadata)
            else:
                prompt = self._create_standard_prompt(question, context, chat_context, question_type, question_metadata)
            
            # Call LLM
            response = self.llm.invoke(prompt)
            answer = response.content
            
            logger.info(f"Generated LLM response ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again."
    
    def _create_no_data_prompt(self, question: str, context: str, chat_context: str, question_type: str, data_availability: Dict[str, Any], question_metadata: Dict[str, Any] = None) -> str:
        """Create prompt when specific financial data is not available"""
        reason = data_availability.get("reason", "unknown")
        suggestion = data_availability.get("suggestion", "")
        
        metadata_context = ""
        if question_metadata:
            if question_metadata.get('currencies_mentioned'):
                metadata_context += f"Requested currencies: {', '.join(question_metadata['currencies_mentioned'])}\n"
            if question_metadata.get('years_mentioned'):
                metadata_context += f"Requested years: {', '.join(question_metadata['years_mentioned'])}\n"
        
        return f"""You are a helpful financial assistant. A user asked a {question_type} question, but the available documents don't contain the specific financial data needed.

{chat_context}Available document context:
{context}

User question: {question}

{metadata_context}
The documents available are primarily governance and compliance reports, not detailed financial statements with specific revenue/profit numbers.

Please respond by:
1. Acknowledging that you understand their {question_type} question
2. Explaining what specific data is missing ({reason})
3. Mentioning what type of information IS available in the documents
4. Suggesting they need access to actual financial statements (10-K, annual reports, etc.) for specific numbers
5. Being helpful about what you CAN tell them from the available documents

Be professional and helpful while being clear about the limitations."""
    
    def _create_standard_prompt(self, question: str, context: str, chat_context: str, question_type: str, question_metadata: Dict[str, Any] = None) -> str:
        """Create standard prompt when relevant data is available"""
        metadata_context = ""
        if question_metadata:
            if question_metadata.get('currencies_mentioned'):
                metadata_context += f"User is specifically asking about: {', '.join(question_metadata['currencies_mentioned'])} currency\n"
            if question_metadata.get('years_mentioned'):
                metadata_context += f"User is specifically asking about: {', '.join(question_metadata['years_mentioned'])}\n"
        
        return f"""You are an expert financial assistant specializing in {question_type} analysis.

{chat_context}Context from financial documents:
{context}

Current question: {question}

{metadata_context}
Please analyze the provided context and answer the question. Pay special attention to:
- Currency-specific data if mentioned in the question
- Year-specific data if mentioned in the question  
- Exact financial figures with proper citations
- Page references for all data sources

If you find specific financial numbers, cite them with page references. If the context doesn't contain the exact information requested, explain what information IS available and what would be needed for a complete answer.

Always cite page numbers when referencing document content and be specific about data sources, currencies, and time periods."""