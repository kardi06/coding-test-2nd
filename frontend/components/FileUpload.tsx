import React, { useState, useRef } from 'react';
import axios from 'axios';
import { on } from 'events';

interface FileUploadProps {
  onUploadComplete?: (result: any) => void;
  onUploadError?: (error: string) => void;
  maxFileSizeMB?: number;
}

export default function FileUpload({ 
  onUploadComplete, 
  onUploadError,
  maxFileSizeMB = 10,
}: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement| null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<String | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  // validate file PDF
  const validateFile = (file:File): string | null => {
    if(!file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are allowed';
    }
    if(file.size > maxFileSizeMB * 1024 * 1024) {
      return `File size must be less than ${maxFileSizeMB}MB`;
    }
    return null;
  }
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    // TODO: Implement file selection
    // 1. Validate file type (PDF only)
    // 2. Validate file size
    // 3. Set selected file
    const selectedFile = e.target.files?.[0];
    if(selectedFile){
      const err = validateFile(selectedFile);
      if(err){
        setError(err);
        setFile(null);
        if (onUploadError) {
          onUploadError(err);
        }
        return;
      }
      setError(null);
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    // TODO: Implement file upload
    // 1. Create FormData with selected file
    // 2. Send POST request to /api/upload
    // 3. Handle upload progress
    // 4. Handle success/error responses
    if(!file) return;
    setIsUploading(true);
    setUploadProgress(0);
    setError(null);
    try{
      const formData = new FormData();
      formData.append('file', file);
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        },
      });
      setIsUploading(false);
      setFile(null);
      setUploadProgress(0);
      if (onUploadComplete) {
        onUploadComplete(response.data);
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Upload failed, Please try again.';
      setError(msg);
      setIsUploading(false);
      setUploadProgress(0);
      if (onUploadError) {
        onUploadError(msg);
      }
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    // TODO: Handle drag over events
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    // TODO: Handle file drop events
    e.preventDefault();
    e.stopPropagation();
    const dropped = e.dataTransfer.files?.[0];
    if(dropped){
      const err = validateFile(dropped);
      if (err){
        setError(err);
        setFile(null);
        if (onUploadError) {
          onUploadError(err);
        }
        return;
      }
      setError(null);
      setFile(dropped);
      // if (fileInputRef.current) {
      //   fileInputRef.current.value = dropped.name;
      // }
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  }

  return (
    <div className="file-upload">
      {/* TODO: Implement file upload UI */}
      
      {/* Drag & Drop area */}
      <div 
        className="upload-area"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={openFileDialog}
        style={{
          border: '2px dashed #ccc',
          borderRadius: '8px',
          padding: '32px',
          textAlign: 'center',
          cursor: 'pointer',
          marginBottom: '12px',
          background: file ? '#f6fff6' : '#fafafa',
        }}
        tabIndex={0}
        aria-label="Upload PDF by dragging or clicking here"
      >
        {/* TODO: Implement drag & drop UI */}
        {file ? (
          <div>
            <strong>File:</strong> {file.name}
          </div>
        ) : (
          <div>
            Drag & drop your PDF here or <span style={{ textDecoration: 'underline' }}>click to select</span>
          </div>
        )}
      </div>

      {/* File input */}
      <input
        type="file"
        accept=".pdf"
        ref={fileInputRef}
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        tabIndex={-1}
      />

      {/* Upload button */}
      <button 
        onClick={handleUpload}
        disabled={!file || isUploading}
        style={{ marginBottom: 8, padding: '8px 18px', borderRadius: '6px' }}
      >
        {isUploading ? 'Uploading...' : 'Upload PDF'}
      </button>

      {/* Progress bar */}
      {isUploading && (
        <div className="progress-bar"
          style={{marginTop: 8, height: 6, background: '#eee', borderRadius: 3}}
        >
          {/* TODO: Implement progress bar */}
          <div
            style={{
              width: `${uploadProgress}%`,
              height: '100%',
              background: '#2b8a3e',
              borderRadius: 3,
              transition: 'width 0.3s',
            }}
          />
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div className="error-message" style={{ color: '#e63946', marginTop: 8 }}>
          {error}
        </div>
      )}

    </div>
  );
} 