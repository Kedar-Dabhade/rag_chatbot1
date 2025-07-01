import React from "react";
import "./MessageBubble.css";

function MessageBubble({ sender, text, typing }) {
  const isUser = sender === "user";
  return (
    <div className={`message-bubble ${isUser ? "user" : "bot"}`}>
      {typing ? (
        <span className="typing-indicator">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </span>
      ) : (
        <span style={{ whiteSpace: "pre-line" }}>{text}</span>
      )}
    </div>
  );
}

export default MessageBubble;