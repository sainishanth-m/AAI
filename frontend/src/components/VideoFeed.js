import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

/* ──────────────────────────────────────────────────────────
   VideoFeed.js
   Centred webcam display with professional light theme controls.
   Optimised for smooth MJPEG streaming with minimal overhead.
────────────────────────────────────────────────────────── */

const API = process.env.REACT_APP_API_URL || '';

export default function VideoFeed({ onLogsUpdate }) {
  const [streaming, setStreaming] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fps, setFps] = useState(0);
  const [objects, setObjects] = useState([]);
  const [streamKey, setStreamKey] = useState(0);
  const pollInterval = useRef(null);

  const startStream = async () => {
    setLoading(true);
    try {
      await axios.post(`${API}/api/start`, {
        model_name: 'yolov8n.pt',
        confidence: 0.25,
      });
      // Bump key to force a fresh <img> connection
      setStreamKey(prev => prev + 1);
      setStreaming(true);
      startPolling();
    } catch (err) {
      console.error('Failed to start stream:', err);
      alert('Could not connect to backend. Is Flask running?');
    } finally {
      setLoading(false);
    }
  };

  const stopStream = async () => {
    try {
      await axios.post(`${API}/api/stop`);
      setStreaming(false);
      stopPolling();
      setFps(0);
      setObjects([]);
    } catch (err) {
      console.error('Failed to stop stream:', err);
    }
  };

  const startPolling = () => {
    stopPolling();
    // Poll less frequently (every 2s instead of 1s) to reduce load
    pollInterval.current = setInterval(async () => {
      try {
        const [statusRes, historyRes] = await Promise.all([
          axios.get(`${API}/api/status`),
          axios.get(`${API}/api/history`),
        ]);
        
        setFps(statusRes.data.fps || 0);

        if (historyRes.data && historyRes.data.length > 0) {
          const latest = historyRes.data[historyRes.data.length - 1];
          // Backend sends {labels: [...], avg_conf: number}
          // Convert to the format the HUD overlay expects: [{label, confidence}]
          const objs = (latest.labels || []).map(lbl => ({
            label: lbl,
            confidence: latest.avg_conf || 0,
          }));
          setObjects(objs);
          if (onLogsUpdate) onLogsUpdate(historyRes.data);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 2000);
  };

  const stopPolling = () => {
    if (pollInterval.current) clearInterval(pollInterval.current);
  };

  useEffect(() => {
    return () => stopPolling();
  }, []);

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div className="card" style={{ padding: '0', overflow: 'hidden', position: 'relative', minHeight: '450px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#f1f5f9' }}>
        {streaming ? (
          <img
            key={streamKey}
            src={`${API}/video_feed`}
            alt="YOLO Stream"
            style={{ width: '100%', height: 'auto', display: 'block' }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <div style={{ color: '#94a3b8', marginBottom: '16px' }}>
              <svg width="64" height="64" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
                <path d="M23 7l-7 5 7 5V7z"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
            </div>
            <div style={{ fontWeight: 600, color: '#64748b' }}>Camera feed is currently inactive</div>
            <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '4px' }}>Click start to begin real-time detection</p>
          </div>
        )}

        {/* ── HUD Overlays ────────────────────────── */}
        {streaming && (
          <div style={{ position: 'absolute', top: '16px', left: '16px', display: 'flex', gap: '8px' }}>
            <div className="badge badge-green">
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981', marginRight: '6px' }}></span>
              LIVE: {fps.toFixed(1)} FPS
            </div>
            <div className="badge badge-blue">
              MODEL: YOLOv8 Nano
            </div>
          </div>
        )}

        {loading && (
          <div style={{ position: 'absolute', inset: 0, backgroundColor: 'rgba(255,255,255,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10 }}>
            <div className="spinner" style={{ width: '40px', height: '40px', borderTopColor: '#2563EB', borderLeftColor: '#2563EB' }}></div>
          </div>
        )}
      </div>

      {/* ── Controls ─────────────────────────────── */}
      <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', marginTop: '24px' }}>
        {!streaming ? (
          <button onClick={startStream} disabled={loading} className="btn btn-primary" style={{ padding: '12px 32px' }}>
            {loading ? 'Starting...' : 'Start Detection'}
          </button>
        ) : (
          <button onClick={stopStream} className="btn btn-danger" style={{ padding: '12px 32px' }}>
            Stop Detection
          </button>
        )}
      </div>

      {/* ── Detection Stats ──────────────────────── */}
      {streaming && objects.length > 0 && (
        <div className="card" style={{ marginTop: '24px', padding: '16px 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.875rem', fontWeight: 700, color: '#111827' }}>Detected Objects:</span>
            {objects.map((obj, i) => (
              <span key={i} className="badge badge-blue" style={{ fontSize: '0.8125rem' }}>
                {obj.label} ({Math.round(obj.confidence * 100)}%)
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
