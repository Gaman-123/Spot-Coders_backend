import React from 'react';
import { User, Calendar, Activity, FileText, Phone, Mail } from 'lucide-react';
import './PatientDetails.css';

const PatientDetails = () => {
  return (
    <div className="patient-details">
      <div className="patient-header">
        <div className="patient-profile-info">
          <div className="avatar-large">EP</div>
          <div className="patient-title">
            <h1>Eleanor Pena</h1>
            <p>ID: P-1029 | Female, 42 yrs</p>
          </div>
        </div>
        <div className="patient-actions">
          <button className="btn-secondary">Edit Patient</button>
          <button className="btn-primary">New Appointment</button>
        </div>
      </div>

      <div className="details-grid">
        <div className="details-sidebar">
          <div className="info-card">
            <h3>Contact Information</h3>
            <div className="info-row">
              <Phone size={16} /> <span>(555) 123-4567</span>
            </div>
            <div className="info-row">
              <Mail size={16} /> <span>eleanor.pena@example.com</span>
            </div>
          </div>
          
          <div className="info-card">
            <h3>Vitals Summary</h3>
            <div className="vital-item">
              <span>Blood Pressure</span>
              <strong>120/80 mmHg</strong>
            </div>
            <div className="vital-item">
              <span>Heart Rate</span>
              <strong>72 bpm</strong>
            </div>
            <div className="vital-item">
              <span>Weight</span>
              <strong>145 lbs</strong>
            </div>
          </div>
        </div>

        <div className="details-main">
          <div className="history-section">
            <div className="section-header">
              <h2><Activity size={20} /> Medical History</h2>
            </div>
            <div className="timeline">
              <div className="timeline-item">
                <div className="timeline-date">Aug 10, 2026</div>
                <div className="timeline-content">
                  <h4>Routine Checkup</h4>
                  <p>Patient reported mild fatigue. Prescribed vitamin supplements and advised rest.</p>
                </div>
              </div>
              <div className="timeline-item">
                <div className="timeline-date">Jan 15, 2026</div>
                <div className="timeline-content">
                  <h4>Hypertension Follow-up</h4>
                  <p>Blood pressure stable on current medication. Renewed prescription for 6 months.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientDetails;
