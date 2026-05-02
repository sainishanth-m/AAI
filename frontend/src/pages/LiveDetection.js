import React, { useState } from 'react';
import VideoFeed from '../components/VideoFeed';

/* ──────────────────────────────────────────────────────────
   LiveDetection.js
   Centred webcam feed with a clean academic layout.
────────────────────────────────────────────────────────── */

export default function LiveDetection() {
  const [history, setHistory] = useState([]);

  return (
    <div className="animate-fade-in">
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <h1 className="section-title">Real-Time Object Detection using YOLO</h1>
        <p className="section-sub">
          Live inference stream using YOLOv8 architecture with on-screen bounding box overlays.
        </p>
      </div>

      {!process.env.REACT_APP_API_URL && (
        <div style={{ maxWidth: '800px', margin: '0 auto 24px', padding: '16px 24px', borderRadius: '8px', backgroundColor: '#EFF6FF', border: '1px solid #BFDBFE', color: '#1E40AF', fontSize: '0.875rem', textAlign: 'center' }}>
          💡 Live detection requires a local Flask backend with webcam access. Run <code style={{ background: '#DBEAFE', padding: '2px 6px', borderRadius: '4px' }}>python run.py</code> locally to enable this feature.
        </div>
      )}

      <VideoFeed onLogsUpdate={setHistory} />

      {/* ── Detection Log ────────────────────────── */}
      <div style={{ maxWidth: '800px', margin: '48px auto 0' }}>
        <h2 style={{ fontSize: '1.25rem', marginBottom: '20px' }}>Recent Detections</h2>
        <div className="card" style={{ padding: '0' }}>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: '#F9FAFB', borderBottom: '1px solid #E5E7EB' }}>
                <tr>
                  <th style={{ padding: '12px 24px', fontSize: '0.75rem', color: '#6B7280', textTransform: 'uppercase' }}>Timestamp</th>
                  <th style={{ padding: '12px 24px', fontSize: '0.75rem', color: '#6B7280', textTransform: 'uppercase' }}>Objects</th>
                  <th style={{ padding: '12px 24px', fontSize: '0.75rem', color: '#6B7280', textTransform: 'uppercase' }}>Avg. Conf</th>
                </tr>
              </thead>
              <tbody>
                {history.length > 0 ? (
                  history.slice(0, 10).map((log, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #F3F4F6' }}>
                      <td style={{ padding: '12px 24px', fontSize: '0.875rem', color: '#374151', fontFamily: 'monospace' }}>
                        {log.timestamp || '—'}
                      </td>
                      <td style={{ padding: '12px 24px', fontSize: '0.875rem' }}>
                        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                          {log.labels && log.labels.length > 0 ? (
                            log.labels.map((label, j) => (
                              <span key={j} style={{ backgroundColor: '#F3F4F6', color: '#2563EB', padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 600 }}>
                                {label}
                              </span>
                            ))
                          ) : (
                            <span style={{ color: '#94a3b8' }}>None</span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '12px 24px', fontSize: '0.875rem', color: '#374151' }}>
                        {log.avg_conf != null ? (log.avg_conf * 100).toFixed(1) + '%' : '0%'}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="3" style={{ padding: '32px', textAlign: 'center', color: '#94a3b8', fontSize: '0.875rem' }}>
                      No detection history available. Start the stream to begin logging.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
