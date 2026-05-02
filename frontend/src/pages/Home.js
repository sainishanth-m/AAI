import React from 'react';
import { Link } from 'react-router-dom';

/* ──────────────────────────────────────────────────────────
   Home.js
   Landing page with a clean, light academic layout.
────────────────────────────────────────────────────────── */

function FeatureCard({ title, desc, iconColor }) {
  return (
    <div className="card">
      <div style={{
        width: '40px', height: '40px', borderRadius: '8px',
        backgroundColor: `${iconColor}15`, color: iconColor,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        marginBottom: '16px'
      }}>
        <svg width="24" height="24" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 8v8M8 12h8"/>
        </svg>
      </div>
      <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px' }}>{title}</h3>
      <p style={{ fontSize: '0.9375rem', color: '#6B7280' }}>{desc}</p>
    </div>
  );
}

export default function Home() {
  return (
    <div className="animate-fade-in">
      {/* ── Hero Section ──────────────────────────── */}
      <section style={{ textAlign: 'center', marginBottom: '64px', paddingTop: '40px' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          padding: '6px 16px', borderRadius: '999px', background: '#EFF6FF',
          color: '#1D4ED8', fontSize: '0.8125rem', fontWeight: 600,
          marginBottom: '24px'
        }}>
          AI Applications Project • 2024
        </div>
        <h1 className="section-title" style={{ fontSize: '3rem', maxWidth: '800px', margin: '0 auto 16px' }}>
          Real-Time Object Detection using YOLO and Webcam
        </h1>
        <p className="section-sub" style={{ maxWidth: '600px', margin: '0 auto 32px' }}>
          An integrated platform for real-time vision inference, model evaluation, and academic analysis of AI performance metrics.
        </p>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
          <Link to="/live" className="btn btn-primary">Start Live Feed</Link>
          <Link to="/dashboard" className="btn btn-ghost">View Metrics</Link>
        </div>
      </section>

      {/* ── Author Info ───────────────────────────── */}
      <div className="card" style={{ marginBottom: '48px', borderLeft: '4px solid #2563EB' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '0.875rem', fontWeight: 600, color: '#2563EB', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '4px' }}>
              Project Team
            </div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800 }}>AI Course Group</h2>
            <div style={{ color: '#6B7280' }}>Computer Science & Engineering</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.875rem', fontWeight: 500, color: '#111827' }}>Applications of AI Project</div>
            <div style={{ fontSize: '0.875rem', color: '#6B7280' }}>Academic Submission v1.0.0</div>
          </div>
        </div>
      </div>

      {/* ── Features Grid ─────────────────────────── */}
      <div className="grid-3" style={{ marginBottom: '48px' }}>
        <FeatureCard
          title="Real-Time Detection"
          desc="Low-latency inference using YOLOv8 optimized for CPU and GPU environments."
          iconColor="#2563EB"
        />
        <FeatureCard
          title="Model Evaluation"
          desc="Comprehensive analysis of IoU, Precision, Recall, and mAP with live tracking."
          iconColor="#10B981"
        />
        <FeatureCard
          title="Comparative Analysis"
          desc="Visualize the Bias-Variance tradeoff and explore overfitting vs underfitting."
          iconColor="#F59E0B"
        />
      </div>

      {/* ── Technology Stack ─────────────────────── */}
      <h2 style={{ marginBottom: '24px', textAlign: 'center' }}>Technology Stack</h2>
      <div className="grid-3" style={{ gap: '16px', marginBottom: '64px' }}>
        {[
          { name: 'YOLOv8', cat: 'Model Architecture' },
          { name: 'OpenCV', cat: 'Computer Vision' },
          { name: 'Flask', cat: 'Backend API' },
          { name: 'React', cat: 'Frontend UI' },
          { name: 'Recharts', cat: 'Data Visualization' },
          { name: 'NumPy', cat: 'Mathematical Analysis' },
        ].map((tech) => (
          <div key={tech.name} className="card" style={{ padding: '16px', textAlign: 'center' }}>
            <div style={{ fontWeight: 700, color: '#111827' }}>{tech.name}</div>
            <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>{tech.cat}</div>
          </div>
        ))}
      </div>

      {/* ── How it works ─────────────────────────── */}
      <h2 style={{ marginBottom: '24px', textAlign: 'center' }}>System Pipeline</h2>
      <div className="card" style={{ maxWidth: '800px', margin: '0 auto 64px' }}>
        {[
          { id: 1, text: 'Webcam frame captured by OpenCV' },
          { id: 2, text: 'Frame passed to YOLOv8 for inference' },
          { id: 3, text: 'Bounding boxes & labels drawn on frame' },
          { id: 4, text: 'Annotated frame streamed via Flask MJPEG' },
          { id: 5, text: 'React frontend displays the live video feed' },
        ].map((step, i) => (
          <div key={step.id} style={{
            display: 'flex', gap: '20px', padding: '16px 0',
            borderBottom: i < 4 ? '1px solid #E5E7EB' : 'none',
            alignItems: 'center'
          }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '50%',
              backgroundColor: '#F3F4F6', color: '#2563EB',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 700, fontSize: '0.875rem'
            }}>{step.id}</div>
            <div style={{ fontSize: '1rem', color: '#374151' }}>{step.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
