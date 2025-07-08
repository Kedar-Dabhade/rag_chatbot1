import React, { useState } from "react";
import "./MessageBubble.css";

function getSessionIdFromCookie() {
  const match = document.cookie.match(/(?:^|; )session=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

function MessageBubble({ sender, text, typing, userMessage }) {
  const isUser = sender === "user";
  const [feedback, setFeedback] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleFeedback = async (type) => {
    setSubmitting(true);
    await fetch("http://localhost:5000/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        user_message: userMessage,
        bot_message: text,
        feedback: type
      })
    });
    setFeedback(type);
    setSubmitting(false);
  };

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
      {!isUser && !typing && text && (
        <div className="feedback-buttons" style={{ marginTop: 8 }}>
          <button
            onClick={() => handleFeedback("like")}
            disabled={feedback === "like" || submitting}
            style={{ marginRight: 8 }}
          >👍 Like</button>
          <button
            onClick={() => handleFeedback("dislike")}
            disabled={feedback === "dislike" || submitting}
          >👎 Dislike</button>
          {feedback && (
            <span style={{ marginLeft: 12, color: feedback === "like" ? "green" : "red" }}>
              {feedback === "like" ? "Liked" : "Disliked"}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

export default MessageBubble;