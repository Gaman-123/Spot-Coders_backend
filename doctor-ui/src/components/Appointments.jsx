import React from 'react';
import { ChevronLeft, ChevronRight, Plus, Clock } from 'lucide-react';
import './Appointments.css';

const Appointments = () => {
  return (
    <div className="appointments-view">
      <div className="page-header">
        <h1 className="page-title">Appointments</h1>
        <button className="btn-primary"><Plus size={18} style={{marginRight: '8px'}}/> New Appointment</button>
      </div>

      <div className="calendar-container">
        <div className="calendar-header">
          <div className="calendar-nav">
            <button className="nav-btn"><ChevronLeft size={20} /></button>
            <h2>August 2026</h2>
            <button className="nav-btn"><ChevronRight size={20} /></button>
          </div>
          <div className="calendar-views">
            <button className="view-btn active">Day</button>
            <button className="view-btn">Week</button>
            <button className="view-btn">Month</button>
          </div>
        </div>

        <div className="schedule-grid">
          <div className="time-column">
            <div className="time-slot">08:00 AM</div>
            <div className="time-slot">09:00 AM</div>
            <div className="time-slot">10:00 AM</div>
            <div className="time-slot">11:00 AM</div>
            <div className="time-slot">12:00 PM</div>
            <div className="time-slot">01:00 PM</div>
          </div>
          
          <div className="day-column">
            <div className="day-header">Monday, Aug 17</div>
            <div className="events-container">
              <div className="event-card" style={{ top: '60px', height: '60px', backgroundColor: '#e3fcef', borderLeftColor: 'var(--success)' }}>
                <div className="event-title">John Doe - Checkup</div>
                <div className="event-time"><Clock size={12} /> 09:00 AM - 10:00 AM</div>
              </div>
              <div className="event-card" style={{ top: '180px', height: '90px', backgroundColor: '#fff0b3', borderLeftColor: 'var(--warning)' }}>
                <div className="event-title">Jane Smith - Consultation</div>
                <div className="event-time"><Clock size={12} /> 11:00 AM - 12:30 PM</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Appointments;
