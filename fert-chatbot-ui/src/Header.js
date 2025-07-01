import React from "react";
import "./Header.css";

export default function Header() {
  const logo = process.env.PUBLIC_URL + "/resupply-logo.webp";
  return (
    <>
      <div className="resupply-header">
        <div className="header-left">
          <img src={logo} alt="Resupply Logo" className="resupply-logo" />
        </div>
        <div className="header-center">
          <input className="search-bar" placeholder="Search for..." />
        </div>
        <div className="header-right">
          <button className="account-btn">
            <span role="img" aria-label="account">👤</span> Account
          </button>
          <button className="cart-btn">
            <span role="img" aria-label="cart">🛒</span> $0.00
          </button>
        </div>
      </div>
      <nav className="resupply-nav">
        <a href="#">Fertiliser</a>
        <a href="#">Chemical</a>
        <a href="#">Seed</a>
        <a href="#">Test</a>
        <a href="#">Lifestyle</a>
        <a href="#">Help</a>
      </nav>
    </>
  );
}