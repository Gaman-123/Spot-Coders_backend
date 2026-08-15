import React from 'react';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

const Layout = ({ children }) => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <Topbar />
        <div className="page-wrapper">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Layout;
