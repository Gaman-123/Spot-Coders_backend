import React, { useState } from 'react';
import {
  LayoutDashboard, Users, Calendar, FileText,
  Settings, LogOut, Stethoscope, ChevronRight
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'patients', label: 'Patients', icon: Users },
  { id: 'appointments', label: 'Appointments', icon: Calendar },
  { id: 'records', label: 'Records', icon: FileText },
];

const Sidebar = ({ currentView, onNavigate }) => {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-icon">
          <Stethoscope size={20} />
        </div>
        <div>
          <h2>MedConnect</h2>
          <span>Clinical Suite</span>
        </div>
      </div>

      <div className="sidebar-section-label">Main Menu</div>
      <nav className="sidebar-nav">
        <ul>
          {navItems.map(({ id, label, icon: Icon }) => (
            <li key={id} className={currentView === id ? 'active' : ''}>
              <a href="#" onClick={(e) => { e.preventDefault(); onNavigate(id); }}>
                <span className="nav-icon"><Icon size={18} /></span>
                {label}
              </a>
            </li>
          ))}
        </ul>

        <div className="sidebar-divider" />
        <div className="sidebar-section-label">System</div>
        <ul>
          <li>
            <a href="#"><span className="nav-icon"><Settings size={18} /></span>Settings</a>
          </li>
        </ul>
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-user">
          <div className="avatar">SJ</div>
          <div className="sidebar-user-info">
            <div className="sidebar-user-name">Dr. Sarah Jenkins</div>
            <div className="sidebar-user-role">Cardiologist</div>
          </div>
          <LogOut size={16} className="logout-icon" />
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
