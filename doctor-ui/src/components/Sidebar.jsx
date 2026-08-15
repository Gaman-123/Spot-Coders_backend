import React from 'react';
import { Home, Users, Calendar, FileText, Settings, LogOut } from 'lucide-react';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>MedConnect</h2>
      </div>
      <nav className="sidebar-nav">
        <ul>
          <li className="active">
            <a href="#"><Home size={20} /> Dashboard</a>
          </li>
          <li>
            <a href="#"><Users size={20} /> Patients</a>
          </li>
          <li>
            <a href="#"><Calendar size={20} /> Appointments</a>
          </li>
          <li>
            <a href="#"><FileText size={20} /> Records</a>
          </li>
          <li className="settings-link">
            <a href="#"><Settings size={20} /> Settings</a>
          </li>
        </ul>
      </nav>
      <div className="sidebar-footer">
        <button className="logout-btn">
          <LogOut size={20} />
          <span>Log Out</span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
