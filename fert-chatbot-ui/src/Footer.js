import React from "react";
import "./Footer.css";

export default function Footer() {
  return (
    <footer className="resupply-footer">
      <div className="footer-col">
        <div className="footer-title">Need Help?</div>
        <a href="#">Help with ordering</a>
        <a href="#">FertSearch</a>
        <a href="#">Tips for farming</a>
        <a href="#">Contact us</a>
      </div>
      <div className="footer-col">
        <div className="footer-title">More info</div>
        <a href="#">What's Resupply all about?</a>
      </div>
      <div className="footer-col">
        <div className="footer-title">Terms and conditions</div>
        <a href="#">Terms of trade</a>
        <a href="#">Website terms of use</a>
      </div>
      <div className="footer-col">
        <div className="footer-title">Resupply.co.nz</div>
        <div className="footer-copyright">All rights reserved 2025</div>
      </div>
    </footer>
  );
} 