import React, { useState } from 'react';
import Head from 'next/head';
import FileUpload from '../components/FileUpload';
import ChatInterface from '../components/ChatInterface';

interface UploadedDocument {
  filename: string;
  chunk_count: number;
  status: string;
  uploadTime: Date;
}

export default function Home() {
  const [uploadedDocument, setUploadedDocument] = useState<UploadedDocument | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleUploadComplete = (result: any) => {
    console.log('Upload completed:', result);
    setUploadedDocument({
      filename: result.filename,
      chunk_count: result.chunk_count,
      status: result.status || 'processed',
      uploadTime: new Date()
    });
    setUploadError(null);
    setIsUploading(false);
  };

  const handleUploadError = (error: string) => {
    console.error('Upload error:', error);
    setUploadError(error);
    setIsUploading(false);
  };

  const handleUploadStart = () => {
    setIsUploading(true);
    setUploadError(null);
  };

  const resetUpload = () => {
    setUploadedDocument(null);
    setUploadError(null);
    setIsUploading(false);
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f8fafc' }}>
      <Head>
        <title>RAG-based Financial Q&A System</title>
        <meta name="description" content="AI-powered Q&A system for financial documents" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main style={{ 
        maxWidth: '1200px', 
        margin: '0 auto', 
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px'
      }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '20px' }}>
          <h1 style={{ 
            fontSize: '2.5rem', 
            fontWeight: '700',
            color: '#1e293b',
            margin: '0 0 16px 0'
          }}>
            RAG-based Financial Statement Q&A System
          </h1>
          {/* TODO: Implement your components here */}
          {/* 
            Suggested components to implement:
            - FileUpload component for PDF upload
            - ChatInterface component for Q&A
            - DocumentViewer component for document display
          */}
          <p>Welcome to the RAG-based Q&A System!</p>
          <p>Upload a financial statement PDF and start asking questions.</p>
        </div>

        {/* Main Content */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: uploadedDocument ? '1fr 2fr' : '1fr',
          gap: '24px',
          alignItems: 'start'
        }}>
          {/* Left Column - Upload Section */}
          <div style={{
            backgroundColor: 'white',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e2e8f0'
          }}>
            <h2 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '600',
              color: '#1e293b',
              margin: '0 0 16px 0'
            }}>
              📁 Document Upload
            </h2>
            
            {!uploadedDocument ? (
              <div>
                <p style={{ color: '#64748b', marginBottom: '16px' }}>
                  Upload your financial statement PDF to begin analysis
                </p>
                <FileUpload
                  onUploadComplete={handleUploadComplete}
                  onUploadError={handleUploadError}
                  maxFileSizeMB={10}
                />
              </div>
            ) : (
              <div>
                <div style={{
                  padding: '16px',
                  backgroundColor: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  borderRadius: '8px',
                  marginBottom: '16px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                    <span style={{ fontSize: '1.2rem' }}>✅</span>
                    <strong style={{ color: '#15803d' }}>Document Processed Successfully</strong>
                  </div>
                  <div style={{ fontSize: '14px', color: '#166534' }}>
                    <p><strong>File:</strong> {uploadedDocument.filename}</p>
                    <p><strong>Chunks:</strong> {uploadedDocument.chunk_count} text segments</p>
                    <p><strong>Status:</strong> {uploadedDocument.status}</p>
                    <p><strong>Uploaded:</strong> {uploadedDocument.uploadTime.toLocaleString()}</p>
                  </div>
                </div>
                
                <button
                  onClick={resetUpload}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: '#6b7280',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '14px',
                    cursor: 'pointer',
                    transition: 'background-color 0.2s'
                  }}
                  onMouseOver={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#4b5563'}
                  onMouseOut={(e) => (e.target as HTMLButtonElement).style.backgroundColor = '#6b7280'}
                >
                  Upload New Document
                </button>
              </div>
            )}

            {/* Upload Status */}
            {isUploading && (
              <div style={{
                padding: '12px',
                backgroundColor: '#fef3c7',
                border: '1px solid #fbbf24',
                borderRadius: '6px',
                color: '#92400e',
                marginTop: '12px'
              }}>
                ⏳ Processing document...
              </div>
            )}

            {uploadError && (
              <div style={{
                padding: '12px',
                backgroundColor: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: '6px',
                color: '#dc2626',
                marginTop: '12px'
              }}>
                ❌ {uploadError}
              </div>
            )}
          </div>

          {/* Right Column - Chat Interface */}
          {uploadedDocument && (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '16px',
              padding: '24px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
              border: '1px solid #e2e8f0'
            }}>
              <h2 style={{ 
                fontSize: '1.5rem', 
                fontWeight: '600',
                color: '#1e293b',
                margin: '0 0 16px 0'
              }}>
                💬 Q&A Assistant
              </h2>
              <p style={{ color: '#64748b', marginBottom: '16px' }}>
                Ask questions about your financial document and get AI-powered answers with source references
              </p>
              <ChatInterface apiUrl="http://localhost:8000" />
            </div>
          )}
        </div>

        {/* Instructions */}
        {!uploadedDocument && (
          <div style={{
            backgroundColor: 'white',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e2e8f0'
          }}>
            <h3 style={{ 
              fontSize: '1.25rem', 
              fontWeight: '600',
              color: '#1e293b',
              margin: '0 0 16px 0'
            }}>
              🚀 How to Get Started
            </h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '16px' 
            }}>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>1️⃣</div>
                <h4 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Upload PDF</h4>
                <p style={{ margin: 0, fontSize: '14px', color: '#64748b' }}>
                  Upload your financial statement PDF document using the upload area above
                </p>
              </div>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>2️⃣</div>
                <h4 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Document Processing</h4>
                <p style={{ margin: 0, fontSize: '14px', color: '#64748b' }}>
                  AI will process and analyze your document, breaking it into searchable chunks
                </p>
              </div>
              <div style={{ padding: '16px', backgroundColor: '#f8fafc', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', marginBottom: '8px' }}>3️⃣</div>
                <h4 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Ask Questions</h4>
                <p style={{ margin: 0, fontSize: '14px', color: '#64748b' }}>
                  Start asking questions about your financial data and get instant AI-powered answers
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
} 