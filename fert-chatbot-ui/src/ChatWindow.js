import React, { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";

const INITIAL_MESSAGES = [
  { sender: "bot", text: "Hi! How can I help you with fertilisers today?" }
];

function ChatWindow() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Fetch chat history from backend on mount
    fetch("http://localhost:5000/history", {
      method: "GET",
      credentials: "include"
    })
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          // Map backend roles to sender
          setMessages(
            data.map(msg => ({
              sender: msg.role === "ai" ? "bot" : "user",
              text: msg.content
            }))
          );
        } else {
          setMessages(INITIAL_MESSAGES);
        }
      })
      .catch(() => setMessages(INITIAL_MESSAGES));
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage, { sender: "bot", text: "" }]); // Add placeholder bot message
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("http://localhost:5000/session-chat-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ message: input })
      });
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let botMessage = "";
      let idx = messages.length + 1; // index of the bot message
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value);
          botMessage += chunk;
          setMessages(msgs => {
            const newMsgs = [...msgs];
            newMsgs[idx] = { sender: "bot", text: botMessage };
            return newMsgs;
          });
        }
      }
    } catch (e) {
      setMessages(msgs => {
        const newMsgs = [...msgs];
        newMsgs[newMsgs.length - 1] = { sender: "bot", text: "Sorry, there was an error." };
        return newMsgs;
      });
    }
    setLoading(false);
  };

  return (
    <div className="chat-window" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="messages" style={{ flex: 1, overflowY: "auto", padding: "2rem 1.5rem 1rem 1.5rem" }}>
        {messages.map((msg, idx) => {
          // Show typing indicator for the last bot message if loading and no text yet
          const isLast = idx === messages.length - 1;
          const showTyping = loading && msg.sender === "bot" && !msg.text && isLast;
          return (
            <MessageBubble key={idx} sender={msg.sender} text={msg.text} typing={showTyping} />
          );
        })}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && sendMessage()}
          placeholder="Type your question..."
        />
        <button onClick={sendMessage} disabled={loading}>Send</button>
      </div>
    </div>
  );
}

export default ChatWindow;