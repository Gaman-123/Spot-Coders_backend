import React from 'react';
import { Bell, Search, User } from 'lucide-react';
import './Topbar.css';

const Topbar = () => {
  return (
    <header className="topbar">
      <div className="search-container">
        <Search className="search-icon" size={20} />
        <input type="text" placeholder="Search patients, records, or appointments..." className="search-input" />
      </div>
      <div className="topbar-actions">
        <button className="notification-btn">
          <Bell size={20} />
          <span className="badge">3</span>
        </button>
        <div className="user-profile">
          <div className="avatar">
            <User size={20} />
          </div>
          <div className="user-info">
            <span className="user-name">Dr. Sarah Jenkins</span>
            <span className="user-role">Cardiologist</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Topbar;
