import React from 'react';
import { NavLink } from 'react-router-dom';

/* ──────────────────────────────────────────────────────────
   Navbar.js
   Top navigation bar for a professional academic look.
────────────────────────────────────────────────────────── */

const NAV_ITEMS = [
  { to: '/',            label: 'Home'           },
  { to: '/live',        label: 'Live Detection' },
  { to: '/dashboard',   label: 'Dashboard'      },
  { to: '/experiments', label: 'Experiments'    },
  { to: '/about',       label: 'About'          },
];

export default function Navbar() {
  return (
    <nav className="top-nav">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{
          width: '32px', height: '32px', background: '#2563EB',
          borderRadius: '6px', display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: 'white'
        }}>
          <svg width="20" height="20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <span style={{ fontWeight: 800, color: '#111827', fontSize: '1rem' }}>YOLO Vision</span>
          <span style={{ fontSize: '0.65rem', color: '#6B7280', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Real-Time Object Detection
          </span>
        </div>
      </div>

      <div className="nav-links">
        {NAV_ITEMS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            {label}
          </NavLink>
        ))}
      </div>

      <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 500 }}>
        AI Group Project
      </div>
    </nav>
  );
}
