import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

/* ──────────────────────────────────────────────────────────
   Experiments.js
   Scenario comparisons for academic AI project.
────────────────────────────────────────────────────────── */

const scenarioData = [
  { metric: 'Accuracy',  underfitting: 52, good_fit: 87, overfitting: 99 },
  { metric: 'Precision', underfitting: 48, good_fit: 85, overfitting: 98 },
  { metric: 'Recall',    underfitting: 45, good_fit: 84, overfitting: 97 },
  { metric: 'F1 Score',  underfitting: 46, good_fit: 84, overfitting: 97 },
];

const radarData = [
  { subject: 'Accuracy',  A: 52, B: 87, C: 99, fullMark: 100 },
  { subject: 'Precision', A: 48, B: 85, C: 98, fullMark: 100 },
  { subject: 'Recall',    A: 45, B: 84, C: 97, fullMark: 100 },
  { subject: 'F1',        A: 46, B: 84, C: 97, fullMark: 100 },
  { subject: 'Gen.',      A: 95, B: 82, C: 40, fullMark: 100 },
];

export default function Experiments() {
  return (
    <div className="animate-fade-in">
      <h1 className="section-title">Bias vs Variance Analysis</h1>
      <p className="section-sub">
        Comparative study of model behavior under different training constraints: Underfitting, Good Fit, and Overfitting.
      </p>

      {/* ── Comparison Charts ────────────────────── */}
      <div className="grid-2" style={{ marginBottom: '32px' }}>
        <div className="card">
          <div className="card-title">Performance Metrics Comparison</div>
          <div style={{ height: '350px', width: '100%', marginTop: '20px' }}>
            <ResponsiveContainer>
              <BarChart data={scenarioData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                <Tooltip cursor={{ fill: '#F9FAFB' }} />
                <Legend iconType="circle" />
                <Bar name="Underfitting" dataKey="underfitting" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                <Bar name="Good Fit"     dataKey="good_fit"     fill="#10B981" radius={[4, 4, 0, 0]} />
                <Bar name="Overfitting"  dataKey="overfitting"  fill="#EF4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Multi-Dimensional Analysis</div>
          <div style={{ height: '350px', width: '100%', marginTop: '20px' }}>
            <ResponsiveContainer>
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="#E5E7EB" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#6B7280' }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="Underfitting" dataKey="A" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.4} />
                <Radar name="Good Fit"     dataKey="B" stroke="#10B981" fill="#10B981" fillOpacity={0.4} />
                <Radar name="Overfitting"  dataKey="C" stroke="#EF4444" fill="#EF4444" fillOpacity={0.4} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── Key Insights ─────────────────────────── */}
      <h2 style={{ marginBottom: '24px' }}>Experimental Insights</h2>
      <div className="grid-3" style={{ gap: '24px' }}>
        <div className="card" style={{ borderTop: '4px solid #F59E0B' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '12px', color: '#B45309' }}>Underfitting</h3>
          <ul style={{ fontSize: '0.875rem', color: '#4B5563', paddingLeft: '18px' }}>
            <li style={{ marginBottom: '8px' }}>High Bias, Low Variance.</li>
            <li style={{ marginBottom: '8px' }}>Model is too simple to learn the underlying pattern.</li>
            <li>Both training and validation errors are high.</li>
          </ul>
        </div>
        <div className="card" style={{ borderTop: '4px solid #10B981' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '12px', color: '#047857' }}>Good Fit</h3>
          <ul style={{ fontSize: '0.875rem', color: '#4B5563', paddingLeft: '18px' }}>
            <li style={{ marginBottom: '8px' }}>Optimal Bias-Variance Tradeoff.</li>
            <li style={{ marginBottom: '8px' }}>Model generalizes well to unseen data.</li>
            <li>Low training error and low validation error.</li>
          </ul>
        </div>
        <div className="card" style={{ borderTop: '4px solid #EF4444' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '12px', color: '#B91C1C' }}>Overfitting</h3>
          <ul style={{ fontSize: '0.875rem', color: '#4B5563', paddingLeft: '18px' }}>
            <li style={{ marginBottom: '8px' }}>Low Bias, High Variance.</li>
            <li style={{ marginBottom: '8px' }}>Model memorizes noise in the training data.</li>
            <li>Low training error but high validation error.</li>
          </ul>
        </div>
      </div>

      {/* ── Summary Card ─────────────────────────── */}
      <div className="card" style={{ marginTop: '32px', backgroundColor: '#F9FAFB' }}>
        <h3 style={{ marginBottom: '12px' }}>Summary of Findings</h3>
        <p style={{ fontSize: '0.9375rem', color: '#374151', lineHeight: 1.6 }}>
          The experiments demonstrate that while increasing model complexity (overfitting) leads to near-perfect accuracy on training data, it significantly reduces the model's ability to generalize. The <strong>Good Fit</strong> scenario, achieved through balanced complexity and regularization, provides the most reliable performance for real-world object detection tasks.
        </p>
      </div>
    </div>
  );
}
