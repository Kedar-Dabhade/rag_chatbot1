# API Endpoints Documentation

This document describes the REST API endpoints implemented in `fertiliser_chatbot_api_opensearch.py` and explains how each endpoint is used by the frontend React app (`fert-chatbot-ui`).

---

## 1. `POST /session-chat`
**Description:**
- Handles a single-turn chat message for the current user session.
- Receives a JSON payload: `{ "message": "<user message>" }`.
- Uses session-based chat history (persisted in a file per session) to provide context-aware answers.
- Returns: `{ "response": "<bot reply>" }`.

**Frontend Usage:**
- Not directly called by the current UI, but can be used for non-streaming, session-aware chat.

---

## 2. `GET /history`
**Description:**
- Returns the chat history for the current session as a list of messages.
- Each message is an object: `{ "role": "user"|"ai", "content": "..." }`.

**Frontend Usage:**
- Called on mount in `ChatWindow.js` to load previous chat history and display it in the chat window.
- Maps backend roles to UI senders ("user" or "bot").

---

## 3. `POST /chat`
**Description:**
- Handles a single-turn chat message (stateless, not session-specific).
- Receives: `{ "query": "<user question>" }`.
- Retrieves relevant context from the vector store, builds a prompt, and gets an answer from the LLM.
- Returns: `{ "answer": "<bot reply>" }` (optionally could include sources).

**Frontend Usage:**
- Not directly called by the current UI, but can be used for stateless, non-streaming chat.

---

## 4. `POST /chat-stream`
**Description:**
- Handles a single-turn chat message and streams the bot's reply as it is generated.
- Receives: `{ "message": "<user message>" }`.
- Streams the response as plain text (mimetype: `text/plain`).
- Uses a shared (global) chat memory, not session-specific.

**Frontend Usage:**
- Not directly called by the current UI, but could be used for stateless streaming chat.

---

## 5. `POST /session-chat-stream`
**Description:**
- Handles a chat message for the current session and streams the bot's reply as it is generated.
- Receives: `{ "message": "<user message>" }`.
- Streams the response as plain text (mimetype: `text/plain`).
- Uses per-session chat history for context and stores both user and bot messages after streaming completes.

**Frontend Usage:**
- This is the **main endpoint used by the UI** for sending user messages and receiving streaming bot replies.
- In `ChatWindow.js`, when the user sends a message, a POST request is made to `/session-chat-stream` and the streamed response is displayed in real time.

---

# Endpoint-UI Mapping Table

| Endpoint                | Used by UI? | Purpose in UI                                      |
|-------------------------|------------|----------------------------------------------------|
| POST /session-chat      | No         | (Legacy) Session-based, non-streaming chat          |
| GET /history            | Yes        | Load chat history on chat window mount              |
| POST /chat              | No         | (Legacy) Stateless, non-streaming chat              |
| POST /chat-stream       | No         | (Legacy) Stateless, streaming chat                  |
| POST /session-chat-stream | Yes      | Main endpoint for streaming chat with session memory|

---

# Summary
- The frontend (`fert-chatbot-ui`) primarily uses `/history` (to load chat history) and `/session-chat-stream` (to send/receive chat messages in real time).
- Other endpoints exist for non-streaming or stateless chat, but are not used by the current UI implementation.
- All endpoints use session cookies to maintain per-user chat history. 