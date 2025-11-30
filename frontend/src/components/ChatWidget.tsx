import React, { useState } from 'react';
import styles from './ChatWidget.module.css';

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch('https://ai-native-hackathon-backend.vercel.app/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      setResponse('Error: Could not get a response.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button className={styles.chatButton} onClick={toggleChat} >
        Chat
      </button>
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h3>AI Assistant</h3>
            <button onClick={toggleChat}>X</button>
          </div>
          <div className={styles.chatBody}>
            <p><strong>You:</strong> {query}</p>
            <p><strong>AI:</strong> {loading ? 'Thinking...' : response}</p>
          </div>
          <form onSubmit={handleSubmit} className={styles.chatInput}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question...."
              disabled={loading}
            />
            <button type="submit" disabled={loading}>Send</button>
          </form>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
