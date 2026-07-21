import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import './Chat.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [orderId, setOrderId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const history = messages.map((m) => [m.role, m.content]);
    const nextMessages = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setInput('');
    setError(null);
    setLoading(true);

    try {
      const { data } = await axios.post(`${API_URL}/api/chat`, {
        message: text,
        history,
        order_id: orderId.trim() || null,
      });
      setMessages([
        ...nextMessages,
        {
          role: 'assistant',
          content: data.answer,
          sourceOrders: data.source_orders || [],
        },
      ]);
    } catch (err) {
      setError(
        err.response?.data?.detail || 'Something went wrong reaching the assistant.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat">
      <header className="chat-header">
        <h1>CRM Order Assistant</h1>
        <input
          className="order-filter"
          type="text"
          placeholder="Filter by Order ID (optional)"
          value={orderId}
          onChange={(e) => setOrderId(e.target.value)}
        />
      </header>

      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">Ask a question about an order to get started.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message ${m.role}`}>
            <div className="chat-bubble">{m.content}</div>
            {m.sourceOrders?.length > 0 && (
              <div className="chat-sources">
                Sources:{' '}
                {m.sourceOrders.map((id) => (
                  <span className="source-pill" key={id}>
                    {id}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="chat-bubble chat-loading">Thinking…</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {error && <div className="chat-error">{error}</div>}

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder="Ask about an order..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className="chat-send"
          onClick={sendMessage}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat;
