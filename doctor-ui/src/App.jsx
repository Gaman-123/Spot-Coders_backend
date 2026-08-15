import React, { useState } from 'react';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import PatientList from './components/PatientList';
import PatientDetails from './components/PatientDetails';
import Appointments from './components/Appointments';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  return (
    <Layout>
      <div style={{ marginBottom: '1rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <button 
          onClick={() => setCurrentView('dashboard')}
          style={{ padding: '0.5rem', background: currentView === 'dashboard' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'dashboard' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('patients')}
          style={{ padding: '0.5rem', background: currentView === 'patients' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'patients' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Patient List
        </button>
        <button 
          onClick={() => setCurrentView('details')}
          style={{ padding: '0.5rem', background: currentView === 'details' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'details' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Patient Details
        </button>
        <button 
          onClick={() => setCurrentView('appointments')}
          style={{ padding: '0.5rem', background: currentView === 'appointments' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'appointments' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Appointments
        </button>
      </div>

      {currentView === 'dashboard' && <Dashboard />}
      {currentView === 'patients' && <PatientList />}
      {currentView === 'details' && <PatientDetails />}
      {currentView === 'appointments' && <Appointments />}
    </Layout>
  );
}

export default App;
