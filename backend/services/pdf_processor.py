import os
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import settings
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        # TODO: Initialize text splitter with chunk size and overlap settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return page-wise content"""
        # TODO: Implement PDF text extraction
        # - Use pdfplumber or PyPDF2 to extract text from each page
        # - Return list of dictionaries with page content and metadata
        import pdfplumber
        pages_content = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                pages_content.append({
                    'page': i+1,
                    'content': text or ""
                }) 
        return pages_content
    
    def split_into_chunks(self, pages_content: List[Dict[str, Any]]) -> List[Document]:
        """Split page content into chunks"""
        # TODO: Implement text chunking
        # - Split each page content into smaller chunks
        # - Create Document objects with proper metadata
        # - Return list of Document objects
        docs = []
        for page in pages_content:
            chunks = self.text_splitter.split_text(page['content'])
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata={'page': page['page']}))
        return docs
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF file and return list of Document objects"""
        # TODO: Implement complete PDF processing pipeline
        # 1. Extract text from PDF
        # 2. Split text into chunks
        # 3. Return processed documents
        pages_content = self.extract_text_from_pdf(file_path)
        return self.split_into_chunks(pages_content) 