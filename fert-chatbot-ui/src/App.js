import React from "react";
import Header from "./Header";
import ChatWidget from "./ChatWidget";
import "./App.css";

function App() {
  return (
    <>
      <div className="app-bg">
        <Header />
      </div>
      <ChatWidget />
    </>
  );
}

export default App;