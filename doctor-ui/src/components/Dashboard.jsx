import React from 'react';
import { Users, Activity, Clock, FileCheck } from 'lucide-react';
import './Dashboard.css';

const Dashboard = () => {
  return (
    <div className="dashboard">
      <h1 className="page-title">Dashboard Overview</h1>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon bg-primary-light">
            <Users className="text-primary" size={24} />
          </div>
          <div className="stat-info">
            <h3>Active Patients</h3>
            <p className="stat-value">1,248</p>
            <span className="stat-trend positive">+12% this month</span>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon bg-success-light">
            <Activity className="text-success" size={24} />
          </div>
          <div className="stat-info">
            <h3>Today's Appointments</h3>
            <p className="stat-value">24</p>
            <span className="stat-trend neutral">4 remaining</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon bg-warning-light">
            <Clock className="text-warning" size={24} />
          </div>
          <div className="stat-info">
            <h3>Pending Reports</h3>
            <p className="stat-value">7</p>
            <span className="stat-trend negative">Requires attention</span>
          </div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="upcoming-appointments">
          <h2>Upcoming Appointments</h2>
          <div className="appointment-list">
            {[1,2,3].map(i => (
              <div key={i} className="appointment-item">
                <div className="time">09:30 AM</div>
                <div className="details">
                  <h4>John Doe</h4>
                  <p>Follow-up - Cardiology</p>
                </div>
                <button className="btn-secondary">View</button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
