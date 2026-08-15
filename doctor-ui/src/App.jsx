import React, { useState } from 'react';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import PatientList from './components/PatientList';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  return (
    <Layout>
      <div style={{ marginBottom: '1rem', display: 'flex', gap: '1rem' }}>
        <button 
          onClick={() => setCurrentView('dashboard')}
          style={{ padding: '0.5rem', background: currentView === 'dashboard' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'dashboard' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Dashboard View
        </button>
        <button 
          onClick={() => setCurrentView('patients')}
          style={{ padding: '0.5rem', background: currentView === 'patients' ? 'var(--primary-color)' : 'var(--bg-surface)', color: currentView === 'patients' ? 'white' : 'black', borderRadius: '4px' }}
        >
          Patient List View
        </button>
      </div>

      {currentView === 'dashboard' ? <Dashboard /> : <PatientList />}
    </Layout>
  );
}

export default App;
