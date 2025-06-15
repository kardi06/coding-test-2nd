import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

interface Source {
  content: string;
  page: number;
  score: number;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface ChatInterfaceProps {
  // TODO: Define props interface
  apiUrl?: string;
}

export default function ChatInterface({ apiUrl = 'http://localhost:8000' }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto scroll to bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    // TODO: Implement message sending
    // 1. Add user message to chat
    // 2. Send request to backend API
    // 3. Add assistant response to chat
    // 4. Handle loading states and errors
    const trimmed = input.trim();
    if(!trimmed || isLoading) return;
    
    setIsLoading(true);
    setError(null);
    
    // Add user message to chat
    const userMsg: Message = {
      id: uuidv4(),
      type: "user",
      content: trimmed,
    };
    setMessages((msgs) => [...msgs, userMsg]);
    setInput('');

    try {
      // Prepare chat history for API
      const chat_history = messages.map((msg) => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.content,
      }));

      const response = await axios.post(`${apiUrl}/api/chat`, { 
        question: trimmed, 
        chat_history 
      });

      // Add assistant response to chat
      const assistantMsg: Message = {
        id: uuidv4(),
        type: "assistant",
        content: response.data.answer,
        sources: response.data.sources,
      };
      setMessages((msgs) => [...msgs, assistantMsg]);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || "Something went wrong. Please try again.";
      setError(errorMsg);
    }
    setIsLoading(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // TODO: Handle input changes
    setInput(e.target.value);
    setError(null);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    // TODO: Handle enter key press
    if (e.key === "Enter" && !e.shiftKey && !isLoading) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="chat-interface" style={{ 
      height: '600px', 
      display: 'flex', 
      flexDirection: 'column',
      border: '1px solid #e2e8f0',
      borderRadius: '12px',
      overflow: 'hidden',
      backgroundColor: '#ffffff'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 20px',
        borderBottom: '1px solid #e2e8f0',
        backgroundColor: '#f8fafc',
        fontWeight: '600',
        color: '#1e293b'
      }}>
        💬 Financial Document Q&A Assistant
      </div>

      {/* Messages display area */}
      <div className="messages" style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {/* TODO: Render messages */}
        {messages.length === 0 ? (
          <div style={{
            textAlign: 'center',
            color: '#64748b',
            padding: '40px 20px',
            fontStyle: 'italic'
          }}>
            Start by asking a question about your financial document...
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} style={{
              display: 'flex',
              flexDirection: message.type === 'user' ? 'row-reverse' : 'row',
              gap: '12px'
            }}>
              {/* Avatar */}
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: message.type === 'user' ? '#3b82f6' : '#10b981',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: '14px',
                fontWeight: '600',
                flexShrink: 0
              }}>
                {message.type === 'user' ? 'U' : 'AI'}
              </div>

              {/* Message content */}
              <div style={{
                maxWidth: '70%',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <div style={{
                  padding: '12px 16px',
                  borderRadius: '16px',
                  backgroundColor: message.type === 'user' ? '#3b82f6' : '#f1f5f9',
                  color: message.type === 'user' ? 'white' : '#1e293b',
                  wordWrap: 'break-word',
                  whiteSpace: 'pre-wrap'
                }}>
                  {message.content}
                </div>

                {/* Sources display for assistant messages */}
                {message.type === 'assistant' && message.sources && message.sources.length > 0 && (
                  <div style={{
                    padding: '12px',
                    backgroundColor: '#f8fafc',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    fontSize: '14px'
                  }}>
                    <div style={{ fontWeight: '600', marginBottom: '8px', color: '#475569' }}>
                      📄 Sources:
                    </div>
                    {message.sources.map((source, index) => (
                      <div key={index} style={{
                        marginBottom: '6px',
                        padding: '6px',
                        backgroundColor: 'white',
                        borderRadius: '4px',
                        fontSize: '12px'
                      }}>
                        <strong>Page {source.page}</strong> (Score: {source.score?.toFixed(2)})
                        <div style={{ 
                          marginTop: '4px', 
                          color: '#64748b',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {source.content.substring(0, 100)}...
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div style={{
            display: 'flex',
            gap: '12px'
          }}>
            <div style={{
              width: '32px',
              height: '32px',
              borderRadius: '50%',
              backgroundColor: '#10b981',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'white',
              fontSize: '14px',
              fontWeight: '600'
            }}>
              AI
            </div>
            <div style={{
              padding: '12px 16px',
              borderRadius: '16px',
              backgroundColor: '#f1f5f9',
              color: '#64748b',
              fontStyle: 'italic'
            }}>
              Thinking...
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Error display */}
      {error && (
        <div style={{
          margin: '0 16px',
          padding: '8px 12px',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '6px',
          color: '#dc2626',
          fontSize: '14px'
        }}>
          ❌ {error}
        </div>
      )}

      {/* Input area */}
      <div className="input-area" style={{
        padding: '16px',
        borderTop: '1px solid #e2e8f0',
        backgroundColor: '#f8fafc'
      }}>
        {/* TODO: Implement input field and send button */}
        <div style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'flex-end'
        }}>
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your financial document..."
            disabled={isLoading}
            style={{
              flex: 1,
              padding: '12px 16px',
              border: '1px solid #d1d5db',
              borderRadius: '8px',
              fontSize: '14px',
              outline: 'none',
              backgroundColor: isLoading ? '#f9fafb' : 'white',
              color: isLoading ? '#9ca3af' : '#1f2937'
            }}
          />
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            style={{
              padding: '12px 20px',
              backgroundColor: (!input.trim() || isLoading) ? '#d1d5db' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '14px',
              fontWeight: '600',
              cursor: (!input.trim() || isLoading) ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.2s'
            }}
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
} 