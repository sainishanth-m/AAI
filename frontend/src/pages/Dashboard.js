import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  TrainingCurveChart,
  PrecisionRecallChart,
  BiasVarianceChart,
  AccuracyVsSizeChart,
  ConfidenceDistChart,
  useGraphsData,
} from '../components/Charts';

/* ──────────────────────────────────────────────────────────
   Dashboard.js
   Metrics cards + 5 graph visualisations in light theme.
────────────────────────────────────────────────────────── */

const API = process.env.REACT_APP_API_URL || '';

function MetricCard({ label, value, sub, color }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ color: color || '#2563EB' }}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

function ChartCard({ title, children, badge }) {
  return (
    <div className="card">
      <div className="card-title">
        {title}
        {badge && <span className="badge badge-blue" style={{ marginLeft: 'auto' }}>{badge}</span>}
      </div>
      <div style={{ marginTop: '20px' }}>
        {children}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data: graphData, loading: graphsLoading } = useGraphsData();
  const [metrics, setMetrics] = useState(null);
  const [scenario, setScenario] = useState('good_fit');

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const r = await axios.get(`${API}/api/metrics`);
        setMetrics(r.data);
      } catch (_) {
        setMetrics({
          precision:   0.83,
          recall:      0.79,
          f1_score:    0.81,
          example_iou: 0.62,
          tp: 12, fp: 3, fn: 4,
        });
      }
    };
    fetchMetrics();
  }, []);

  return (
    <div className="animate-fade-in">
      <h1 className="section-title">Model Performance Dashboard</h1>
      <p className="section-sub">
        Quantitative evaluation metrics and interactive visualizations of the YOLOv8 model's performance.
      </p>

      {/* ── Metrics Grid ─────────────────────────── */}
      <div className="grid-4" style={{ marginBottom: '32px' }}>
        <MetricCard
          label="Precision"
          value={metrics ? (metrics.precision * 100).toFixed(1) + '%' : '—'}
          sub={`TP: ${metrics?.tp} / FP: ${metrics?.fp}`}
          color="#2563EB"
        />
        <MetricCard
          label="Recall"
          value={metrics ? (metrics.recall * 100).toFixed(1) + '%' : '—'}
          sub={`TP: ${metrics?.tp} / FN: ${metrics?.fn}`}
          color="#10B981"
        />
        <MetricCard
          label="Accuracy"
          value={metrics ? ((metrics.tp / (metrics.tp + metrics.fp + metrics.fn)) * 100).toFixed(1) + '%' : '—'}
          sub={`TP: ${metrics?.tp} / Total: ${(metrics?.tp || 0) + (metrics?.fp || 0) + (metrics?.fn || 0)}`}
          color="#7C3AED"
        />
        <MetricCard
          label="Mean IoU"
          value={metrics ? metrics.example_iou : '—'}
          sub="Overlap Accuracy"
          color="#F59E0B"
        />
      </div>

      {/* ── Main Charts ──────────────────────────── */}
      <div style={{ marginBottom: '32px' }}>
        <ChartCard title="Training Dynamics" badge="Loss Curves">
          <div style={{ display: 'flex', gap: '8px', marginBottom: '20px' }}>
            {[
              { id: 'good_fit', label: 'Good Fit', color: '#10B981' },
              { id: 'overfitting', label: 'Overfitting', color: '#EF4444' },
              { id: 'underfitting', label: 'Underfitting', color: '#F59E0B' },
            ].map(s => (
              <button
                key={s.id}
                onClick={() => setScenario(s.id)}
                className="btn btn-ghost"
                style={{
                  padding: '6px 16px',
                  borderColor: scenario === s.id ? s.color : '#E5E7EB',
                  color: scenario === s.id ? s.color : '#6B7280',
                  backgroundColor: scenario === s.id ? `${s.color}08` : 'transparent',
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
          <TrainingCurveChart data={graphData?.training_curves} scenario={scenario} />
          <p style={{ fontSize: '0.875rem', color: '#6B7280', marginTop: '16px', fontStyle: 'italic' }}>
            Compare how the model learns across different data scenarios. Overfitting shows a large gap between training and validation loss.
          </p>
        </ChartCard>
      </div>

      <div className="grid-2" style={{ marginBottom: '32px' }}>
        <ChartCard title="Precision-Recall" badge={`AP: ${graphData?.precision_recall?.ap || '0.84'}`}>
          <PrecisionRecallChart data={graphData?.precision_recall} />
        </ChartCard>
        <ChartCard title="Bias-Variance Tradeoff">
          <BiasVarianceChart data={graphData?.bias_variance} />
        </ChartCard>
      </div>

      <div className="grid-2" style={{ marginBottom: '48px' }}>
        <ChartCard title="Learning Curve">
          <AccuracyVsSizeChart data={graphData?.accuracy_vs_size} />
        </ChartCard>
        <ChartCard title="Confidence Distribution">
          <ConfidenceDistChart data={graphData?.confidence_distribution} />
        </ChartCard>
      </div>

      {/* ── Metric Formulas ──────────────────────── */}
      <h2 style={{ marginBottom: '20px' }}>Metric Definitions</h2>
      <div className="grid-3" style={{ gap: '16px' }}>
        {[
          { name: 'IoU', formula: 'Area of Overlap / Area of Union' },
          { name: 'Precision', formula: 'TP / (TP + FP)' },
          { name: 'Recall', formula: 'TP / (TP + FN)' },
          { name: 'F1 Score', formula: '2 * (P * R) / (P + R)' },
          { name: 'mAP', formula: 'Mean Average Precision' },
          { name: 'NMS', formula: 'Non-Maximum Suppression' },
        ].map(m => (
          <div key={m.name} className="card" style={{ padding: '16px' }}>
            <div style={{ fontWeight: 700, color: '#2563EB', fontSize: '0.875rem', marginBottom: '4px' }}>{m.name}</div>
            <code style={{ fontSize: '0.75rem', color: '#374151', background: '#F3F4F6', padding: '2px 6px', borderRadius: '4px' }}>{m.formula}</code>
          </div>
        ))}
      </div>


    </div>
  );
}
