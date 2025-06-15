import os
from typing import List, Dict, Any, Optional, Tuple, Set
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
import logging
import re

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        # TODO: Initialize text splitter with chunk size and overlap settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        
        # Initialize universal financial patterns
        self._init_financial_patterns()
    
    def _init_financial_patterns(self):
        """Initialize comprehensive financial detection patterns"""
        
        # Major currencies with symbols and codes
        self.currencies = {
            # Currency codes
            'USD', 'EUR', 'GBP', 'JPY', 'KRW', 'IDR', 'CNY', 'SGD', 'THB', 'MYR', 
            'AUD', 'CAD', 'CHF', 'HKD', 'INR', 'PHP', 'VND', 'TWD',
            # Currency symbols
            '$', '€', '£', '¥', '₩', 'Rp', '￥', 'S$', '₹', '₽', 'R$',
            # Indonesian terms
            'rupiah', 'dollar', 'euro', 'yen', 'won'
        }
        
        # Financial units in multiple languages
        self.financial_units = {
            # English
            'million', 'billion', 'trillion', 'thousand', 'mil', 'bil', 'tril',
            # Indonesian  
            'juta', 'milyar', 'triliun', 'ribu', 'jutaan', 'milyaran', 'triliunan',
            # Asian (Chinese/Japanese)
            '万', '億', '兆', '千', '萬',
            # Abbreviations
            'M', 'B', 'T', 'K', 'mln', 'bln'
        }
        
        # Number format patterns (US, EU, Indonesian)
        self.number_patterns = [
            r'[\d]{1,3}(?:,\d{3})*(?:\.\d{1,2})?',  # US: 1,234.56
            r'[\d]{1,3}(?:\.\d{3})*(?:,\d{1,2})?',  # EU: 1.234,56  
            r'[\d]{1,3}(?:\.\d{3})*',               # ID: 1.234
            r'[\d,]+\.?\d*',                        # General: 1,234 or 1,234.5
            r'[\d.]+,?\d*',                         # General: 1.234 or 1.234,5
        ]
        
        # Build comprehensive financial detection patterns
        self._build_financial_patterns()
    
    def _build_financial_patterns(self):
        """Build comprehensive regex patterns for financial data detection"""
        
        # Currency pattern: matches any currency symbol/code
        currency_pattern = '|'.join(re.escape(curr) for curr in self.currencies)
        
        # Unit pattern: matches any financial unit
        unit_pattern = '|'.join(re.escape(unit) for unit in self.financial_units)
        
        # Number pattern: matches various number formats
        number_pattern = '|'.join(f'({pattern})' for pattern in self.number_patterns)
        
        # Build comprehensive financial patterns
        self.financial_detection_patterns = [
            # Currency + Number + Unit: USD 1,234 million
            rf'({currency_pattern})\s*({number_pattern})\s*({unit_pattern})',
            # Number + Unit + Currency: 1,234 million USD  
            rf'({number_pattern})\s*({unit_pattern})\s*({currency_pattern})',
            # Currency + Number: $1,234 or Rp 1.234
            rf'({currency_pattern})\s*({number_pattern})',
            # Number + Currency: 1,234 USD
            rf'({number_pattern})\s*({currency_pattern})',
            # Revenue/profit with numbers: revenue 1,234
            rf'(revenue|profit|income|loss|sales|earnings)\s*[:\-]?\s*({number_pattern})',
            # Year-specific financial data: 2023 revenue 1,234
            rf'(\d{4})\s*(revenue|profit|income|loss|sales|earnings)\s*[:\-]?\s*({number_pattern})',
            # Financial terms with units: operating profit 1,234 million
            rf'(operating profit|net income|total revenue|gross profit|operating income)\s*[:\-]?\s*({number_pattern})\s*({unit_pattern})?',
        ]
        
        # Year detection patterns
        self.year_patterns = [
            r'\b(19|20)\d{2}\b',                    # Years: 1990-2099
            r'FY\s*(\d{4})',                        # FY2023
            r'fiscal year\s*(\d{4})',               # fiscal year 2023
            r'(\d{4})\s*vs\s*(\d{4})',              # 2023 vs 2022
            r'year\s*(\d{4})',                      # year 2023
            r'(\d{4})\s*financial year',            # 2023 financial year
        ]
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return page-wise content"""
        # TODO: Implement PDF text extraction
        # - Use pdfplumber or PyPDF2 to extract text from each page
        # - Return list of dictionaries with page content and metadata
        pages_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue
                    
                # Enhanced text cleaning for financial documents
                cleaned_text = self._clean_financial_text(text)
                
                # Extract financial metadata
                financial_metadata = self._extract_financial_metadata(cleaned_text)
                
                # Detect financial data type
                content_type = self._detect_content_type(cleaned_text)
                
                pages_content.append({
                    'page': i+1,
                    'content': cleaned_text,
                    'content_type': content_type,
                    'has_financial_data': self._has_financial_data(cleaned_text),
                    'currencies_found': financial_metadata['currencies'],
                    'years_found': financial_metadata['years'],
                    'financial_amounts': financial_metadata['amounts']
                }) 
        return pages_content
    
    def _clean_financial_text(self, text: str) -> str:
        """Clean and normalize financial text for multiple currencies and formats"""
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Normalize various currency formats
        # USD formats
        text = re.sub(r'USD?\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'USD \1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'\$\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'USD \1 \2', text, flags=re.IGNORECASE)
        
        # KRW formats  
        text = re.sub(r'KRW\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'KRW \1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'([0-9,.\s]+)\s*(million|billion|trillion)\s*KRW', r'\1 \2 KRW', text, flags=re.IGNORECASE)
        
        # Indonesian Rupiah formats
        text = re.sub(r'Rp\.?\s*([0-9,.\s]+)\s*(juta|milyar|triliun)', r'IDR \1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'([0-9,.\s]+)\s*(juta|milyar|triliun)\s*rupiah', r'\1 \2 IDR', text, flags=re.IGNORECASE)
        
        # EUR formats
        text = re.sub(r'EUR?\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'EUR \1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'€\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'EUR \1 \2', text, flags=re.IGNORECASE)
        
        # JPY formats
        text = re.sub(r'JPY\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'JPY \1 \2', text, flags=re.IGNORECASE)
        text = re.sub(r'¥\s*([0-9,.\s]+)\s*(million|billion|trillion)', r'JPY \1 \2', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_financial_metadata(self, text: str) -> Dict[str, Any]:
        """Extract currencies, years, and financial amounts from text"""
        currencies_found = set()
        years_found = set()
        amounts_found = []
        
        # Extract currencies
        for currency in self.currencies:
            if re.search(rf'\b{re.escape(currency)}\b', text, re.IGNORECASE):
                currencies_found.add(currency.upper())
        
        # Extract years
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    years_found.update(match)
                else:
                    years_found.add(match)
        
        # Extract financial amounts
        for pattern in self.financial_detection_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            amounts_found.extend(matches)
        
        return {
            'currencies': list(currencies_found),
            'years': sorted(list(years_found)),
            'amounts': amounts_found[:10]  # Limit to first 10 amounts
        }
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of financial content"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['consolidated revenue', 'total revenue', 'sales revenue']):
            return 'revenue_statement'
        elif any(keyword in text_lower for keyword in ['operating profit', 'operating income', 'net income']):
            return 'profit_statement'
        elif any(keyword in text_lower for keyword in ['balance sheet', 'assets', 'liabilities']):
            return 'balance_sheet'
        elif any(keyword in text_lower for keyword in ['cash flow', 'operating cash']):
            return 'cash_flow'
        elif any(keyword in text_lower for keyword in ['financial statement', 'consolidated']):
            return 'financial_summary'
        else:
            return 'general'
    
    def _has_financial_data(self, text: str) -> bool:
        """Check if text contains specific financial data using comprehensive patterns"""
        # Check against all financial detection patterns
        for pattern in self.financial_detection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Additional check for pure number patterns with financial context
        financial_context_patterns = [
            r'(revenue|profit|income|loss|sales|earnings|cost|expense).*[\d,]+',
            r'[\d,]+.*\b(million|billion|trillion|juta|milyar|triliun)\b',
            r'\b(total|net|gross|operating).*[\d,]+',
        ]
        
        for pattern in financial_context_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def split_into_chunks(self, pages_content: List[Dict[str, Any]]) -> List[Document]:
        """Split page content into chunks"""
        # TODO: Implement text chunking
        # - Split each page content into smaller chunks
        # - Create Document objects with proper metadata
        # - Return list of Document objects
        docs = []
        
        for page in pages_content:
            content = page['content']
            
            # Use different chunking strategy for financial data
            if page['has_financial_data']:
                chunks = self._chunk_financial_content(content, page)
            else:
                chunks = self.text_splitter.split_text(content)
                chunks = [{'text': chunk, 'context': 'general'} for chunk in chunks]
            
            # Create Document objects with enhanced metadata
            for chunk_data in chunks:
                chunk_text = chunk_data['text'] if isinstance(chunk_data, dict) else chunk_data
                
                # Convert lists to strings for ChromaDB compatibility
                currencies_str = ','.join(page['currencies_found']) if page['currencies_found'] else ''
                years_str = ','.join(page['years_found']) if page['years_found'] else ''
                amounts_str = str(len(page['financial_amounts'])) if page['financial_amounts'] else '0'
                
                metadata = {
                    'page': page['page'],
                    'content_type': page['content_type'],
                    'has_financial_data': page['has_financial_data'],
                    'chunk_context': chunk_data.get('context', 'general') if isinstance(chunk_data, dict) else 'general',
                    'currencies_found': currencies_str,  # String instead of list
                    'years_found': years_str,            # String instead of list
                    'financial_amounts_count': amounts_str  # Count as string instead of list
                }
                
                docs.append(Document(page_content=chunk_text, metadata=metadata))
        
        return docs
    
    def _chunk_financial_content(self, content: str, page_info: Dict) -> List[Dict[str, str]]:
        """Special chunking for financial content to preserve context"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        current_context = "general"
        
        for para in paragraphs:
            # Detect financial context
            para_context = self._get_financial_context(para)
            
            # If adding this paragraph would exceed chunk size or context changes significantly
            if (len(current_chunk + para) > settings.chunk_size and current_chunk) or \
               (para_context != current_context and current_chunk and para_context != "general"):
                
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'context': current_context
                    })
                
                current_chunk = para
                current_context = para_context
            else:
                current_chunk += "\n\n" + para if current_chunk else para
                if current_context == "general" and para_context != "general":
                    current_context = para_context
        
        # Add the final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'context': current_context
            })
        
        return chunks
    
    def _get_financial_context(self, text: str) -> str:
        """Determine financial context of a text chunk"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['revenue', 'sales', 'total revenue']):
            return 'revenue'
        elif any(keyword in text_lower for keyword in ['profit', 'income', 'earnings', 'loss']):
            return 'profit_loss'
        elif any(keyword in text_lower for keyword in ['cash flow', 'operating cash']):
            return 'cash_flow'
        elif any(keyword in text_lower for keyword in ['debt', 'liability', 'assets']):
            return 'balance_sheet'
        else:
            return 'general'
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file and return list of Document objects"""
        # TODO: Implement complete PDF processing pipeline
        # 1. Extract text from PDF
        # 2. Split text into chunks
        # 3. Return processed documents
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Extract text with enhanced metadata
            pages_content = self.extract_text_from_pdf(file_path)
            
            # Log document analysis
            financial_pages = sum(1 for page in pages_content if page['has_financial_data'])
            all_currencies = set()
            all_years = set()
            
            for page in pages_content:
                all_currencies.update(page['currencies_found'])
                all_years.update(page['years_found'])
            
            logger.info(f"Extracted {len(pages_content)} pages, {financial_pages} contain financial data")
            logger.info(f"Currencies found: {sorted(list(all_currencies))}")
            logger.info(f"Years found: {sorted(list(all_years))}")
            
            # Split into enhanced chunks
            documents = self.split_into_chunks(pages_content)
            
            logger.info(f"Created {len(documents)} document chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise 