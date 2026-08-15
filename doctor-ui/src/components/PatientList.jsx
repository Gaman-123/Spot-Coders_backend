import React, { useState } from 'react';
import { Search, Filter, MoreVertical } from 'lucide-react';
import './PatientList.css';

const MOCK_PATIENTS = [
  { id: 'P-1029', name: 'Eleanor Pena', age: 42, gender: 'F', lastVisit: '2026-08-10', condition: 'Hypertension', status: 'Stable' },
  { id: 'P-1030', name: 'Wade Warren', age: 28, gender: 'M', lastVisit: '2026-08-14', condition: 'Asthma', status: 'Review' },
  { id: 'P-1031', name: 'Brooklyn Simmons', age: 65, gender: 'F', lastVisit: '2026-07-22', condition: 'Diabetes Type II', status: 'Critical' },
  { id: 'P-1032', name: 'Guy Hawkins', age: 34, gender: 'M', lastVisit: '2026-08-15', condition: 'Post-op Recovery', status: 'Stable' },
  { id: 'P-1033', name: 'Darrell Steward', age: 51, gender: 'M', lastVisit: '2026-08-01', condition: 'Arrhythmia', status: 'Review' },
];

const PatientList = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPatients = MOCK_PATIENTS.filter(p => 
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="patient-list-view">
      <div className="page-header">
        <h1 className="page-title">Patient Directory</h1>
        <button className="btn-primary">Add Patient</button>
      </div>

      <div className="table-controls">
        <div className="search-box">
          <Search size={18} />
          <input 
            type="text" 
            placeholder="Search by name or ID..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        <button className="btn-icon">
          <Filter size={18} /> <span>Filter</span>
        </button>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Name</th>
              <th>Age/Gender</th>
              <th>Primary Condition</th>
              <th>Last Visit</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatients.map(patient => (
              <tr key={patient.id}>
                <td className="fw-500">{patient.id}</td>
                <td>
                  <div className="patient-name-cell">
                    <div className="avatar-small">{patient.name.charAt(0)}</div>
                    <span>{patient.name}</span>
                  </div>
                </td>
                <td>{patient.age} / {patient.gender}</td>
                <td>{patient.condition}</td>
                <td>{patient.lastVisit}</td>
                <td>
                  <span className={`status-badge ${patient.status.toLowerCase()}`}>
                    {patient.status}
                  </span>
                </td>
                <td>
                  <button className="action-btn"><MoreVertical size={18} /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PatientList;
