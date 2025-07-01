import React, { useState } from "react";
import ChatWindow from "./ChatWindow";
import "./ChatWidget.css";

export default function ChatWidget() {
  const [open, setOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  // Persist open/close state and chat in sessionStorage
  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const handleRefresh = () => {
    sessionStorage.removeItem("fertgpt-messages");
    setRefreshKey(k => k + 1);
  };

  return (
    <>
      {!open && (
        <div className="chat-widget-bubble" onClick={handleOpen}>
          Ask Fertiliser Questions
        </div>
      )}
      {open && (
        <div className="chat-widget-panel">
          <div className="chat-widget-header">
            <span className="fertgpt-title-widget">FERT GPT</span>
            <div style={{ display: "flex", alignItems: "center" }}>
              <button className="chat-widget-refresh" onClick={handleRefresh} title="Refresh chat">⟳</button>
              <button className="chat-widget-close" onClick={handleClose}>&times;</button>
            </div>
          </div>
          <ChatWindow key={refreshKey} />
        </div>
      )}
    </>
  );
} 