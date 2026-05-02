import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import LiveDetection from './pages/LiveDetection';
import Dashboard from './pages/Dashboard';
import Experiments from './pages/Experiments';
import About from './pages/About';

/**
 * App.js — Root application component
 * Sets up client-side routing with React Router v6.
 * The Navbar sidebar is shared across all pages.
 */
function App() {
  return (
    <Router>
      <div className="app-layout">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/"              element={<Home />} />
            <Route path="/live"          element={<LiveDetection />} />
            <Route path="/dashboard"     element={<Dashboard />} />
            <Route path="/experiments"   element={<Experiments />} />
            <Route path="/about"         element={<About />} />
          </Routes>
        </main>
        <footer className="academic-footer">
          <div style={{ fontWeight: 700, color: '#111827', marginBottom: '8px' }}>
            Real-Time Object Detection System
          </div>
          <div style={{ marginBottom: '16px' }}>
            Applications of AI Project • Developed as a Team
          </div>
          <div style={{ fontSize: '0.75rem' }}>
            Built with YOLOv8, Flask, and React.js
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
