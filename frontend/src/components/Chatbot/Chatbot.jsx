import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import styles from './Chatbot.module.css';

// Placeholder component for the textbook chatbot
const Chatbot = ({ chapterId }) => {
  const [messages, setMessages] = useState([
    { id: 1, role: 'assistant', content: 'Hello! I\'m your AI textbook assistant. How can I help you with the Physical AI & Humanoid Robotics textbook today?' }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();

    if (!inputMessage.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputMessage
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // In a real implementation, this would call the backend API
      // const response = await fetch('/api/v1/chat', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     message: inputMessage,
      //     user_id: 'current-user-id',
      //     context: chapterId
      //   })
      // });
      // const data = await response.json();

      // For now, simulate a response
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Placeholder response
      const botResponse = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `I received your question: "${inputMessage}". This is a placeholder response from the AI textbook assistant. In the full implementation, this would be answered using RAG with textbook content.`
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.chatbot}>
      <div className={styles.chatHeader}>
        <h3>Textbook Assistant</h3>
        <p>Ask questions about the content using AI</p>
      </div>

      <div className={styles.chatMessages}>
        {messages.map((message) => (
          <div
            key={message.id}
            className={clsx(
              styles.message,
              styles[message.role]
            )}
          >
            <div className={styles.messageContent}>
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className={clsx(styles.message, styles.assistant)}>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className={styles.chatInputForm}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Ask a question about the textbook content..."
          className={styles.chatInput}
          disabled={isLoading}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={isLoading || !inputMessage.trim()}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default Chatbot;