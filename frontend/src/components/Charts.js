import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  LineChart, Line, AreaChart, Area,
  BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts';

/* ──────────────────────────────────────────────────────────
   Charts.js
   Clean, professional chart components for academic presentation.
────────────────────────────────────────────────────────── */

const API = process.env.REACT_APP_API_URL || '';

// Custom theme constants
const COLORS = {
  blue:   '#2563EB', // Primary
  green:  '#10B981', // Success
  red:    '#EF4444', // Error / Overfit
  amber:  '#F59E0B', // Warning / Underfit
  gray:   '#94A3B8', // Muted
  text:   '#374151',
  grid:   '#F1F5F9',
};

// ─── Fallback simulated data ─────────────────────────────
// Used when the backend server is not running.

function generateFallbackTrainingCurves() {
  const makeScenario = (scenario) => {
    const epochs = 50;
    const data = [];
    for (let i = 1; i <= epochs; i++) {
      let train, val;
      if (scenario === 'underfitting') {
        train = 2.5 - 0.015 * i + (Math.random() - 0.5) * 0.1;
        val   = 2.6 - 0.012 * i + (Math.random() - 0.5) * 0.12;
      } else if (scenario === 'overfitting') {
        train = 2.5 * Math.exp(-0.12 * i) + (Math.random() - 0.5) * 0.04;
        const rise = Math.max(0, i - 20);
        val   = 2.5 * Math.exp(-0.07 * i) + 0.003 * Math.pow(rise, 1.5) + (Math.random() - 0.5) * 0.08;
      } else {
        train = 2.5 * Math.exp(-0.09 * i) + (Math.random() - 0.5) * 0.06;
        val   = 2.5 * Math.exp(-0.08 * i) + (Math.random() - 0.5) * 0.08;
      }
      data.push({ epoch: i, train_loss: Math.max(0.05, train), val_loss: Math.max(0.05, val) });
    }
    return data;
  };
  return {
    good_fit:     makeScenario('good_fit'),
    overfitting:  makeScenario('overfitting'),
    underfitting: makeScenario('underfitting'),
  };
}

function generateFallbackPrecisionRecall() {
  const points = [];
  for (let i = 0; i < 20; i++) {
    const recall    = i / 19 * 0.95;
    const precision = Math.max(0.1, Math.min(1.0, 1.0 - 0.6 * Math.pow(recall, 1.5) + (Math.random() - 0.5) * 0.04));
    points.push({ recall: parseFloat(recall.toFixed(3)), precision: parseFloat(precision.toFixed(3)) });
  }
  return { points, ap: 0.84 };
}

function generateFallbackBiasVariance() {
  const data = [];
  for (let c = 1; c <= 15; c++) {
    const bias       = 2.5 / (0.3 * c + 0.5);
    const variance   = 0.05 * Math.pow(c, 1.4);
    const totalError = bias + variance + (Math.random() - 0.5) * 0.1;
    data.push({
      complexity:  c,
      bias:        parseFloat(bias.toFixed(3)),
      variance:    parseFloat(variance.toFixed(3)),
      total_error: parseFloat(Math.max(0, totalError).toFixed(3)),
    });
  }
  return data;
}

function generateFallbackAccuracyVsSize() {
  const sizes = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000];
  return sizes.map(s => ({
    size:      s,
    train_acc: parseFloat(Math.min(0.999, (0.98 - 20 / (s + 40)) * 100).toFixed(1)),
    val_acc:   parseFloat(Math.min(0.98, (0.93 - 50 / (s + 80)) * 100).toFixed(1)),
  }));
}

function generateFallbackConfidenceDist() {
  const centers = [];
  for (let i = 0; i < 20; i++) {
    centers.push(parseFloat(((i + 0.5) / 20).toFixed(3)));
  }
  // Simulated distribution: more detections at high confidence
  const counts = [5, 8, 12, 15, 10, 8, 6, 5, 7, 9, 12, 18, 22, 28, 35, 42, 48, 55, 40, 30];
  return centers.map((bin, i) => ({ bin: bin.toFixed(2), count: counts[i] }));
}

// ─── Data transformation helpers ─────────────────────────
// The backend returns flat arrays; Recharts needs arrays of objects.

function transformTrainingCurves(raw) {
  if (!raw) return null;
  const result = {};
  for (const scenario of ['good_fit', 'overfitting', 'underfitting']) {
    const sc = raw[scenario];
    if (!sc) continue;
    // Already array of objects (properly formatted)?
    if (Array.isArray(sc) && sc.length > 0 && typeof sc[0] === 'object' && 'epoch' in sc[0]) {
      result[scenario] = sc;
      continue;
    }
    // Backend format: { epochs: [], train_loss: [], val_loss: [] }
    if (sc.epochs && sc.train_loss && sc.val_loss) {
      result[scenario] = sc.epochs.map((e, i) => ({
        epoch:      e,
        train_loss: parseFloat(sc.train_loss[i].toFixed(4)),
        val_loss:   parseFloat(sc.val_loss[i].toFixed(4)),
      }));
    }
  }
  return Object.keys(result).length > 0 ? result : null;
}

function transformPrecisionRecall(raw) {
  if (!raw) return null;
  // Already has .points array?
  if (raw.points && Array.isArray(raw.points)) return raw;
  // Backend format: { precisions: [], recalls: [], ap: number }
  if (raw.precisions && raw.recalls) {
    const points = raw.recalls.map((r, i) => ({
      recall:    parseFloat(r.toFixed(3)),
      precision: parseFloat(raw.precisions[i].toFixed(3)),
    }));
    // Sort by recall ascending
    points.sort((a, b) => a.recall - b.recall);
    return { points, ap: raw.ap };
  }
  return null;
}

function transformBiasVariance(raw) {
  if (!raw) return null;
  // Already array of objects?
  if (Array.isArray(raw)) return raw;
  // Backend format: { complexity: [], bias: [], variance: [], total_error: [] }
  if (raw.complexity && raw.bias && raw.variance && raw.total_error) {
    return raw.complexity.map((c, i) => ({
      complexity:  c,
      bias:        parseFloat(raw.bias[i].toFixed(3)),
      variance:    parseFloat(raw.variance[i].toFixed(3)),
      total_error: parseFloat(raw.total_error[i].toFixed(3)),
    }));
  }
  return null;
}

function transformAccuracyVsSize(raw) {
  if (!raw) return null;
  // Already array of objects?
  if (Array.isArray(raw)) return raw;
  // Backend format: { dataset_sizes: [], train_accuracy: [], val_accuracy: [] }
  if (raw.dataset_sizes && raw.train_accuracy && raw.val_accuracy) {
    return raw.dataset_sizes.map((s, i) => ({
      size:      s,
      train_acc: parseFloat((raw.train_accuracy[i] * 100).toFixed(1)),
      val_acc:   parseFloat((raw.val_accuracy[i] * 100).toFixed(1)),
    }));
  }
  return null;
}

function transformConfidenceDist(raw) {
  if (!raw) return null;
  // Already array of objects?
  if (Array.isArray(raw)) return raw;
  // Backend format: { bins: [], counts: [] }
  if (raw.bins && raw.counts) {
    return raw.bins.map((b, i) => ({
      bin:   b.toFixed ? b.toFixed(2) : String(b),
      count: raw.counts[i],
    }));
  }
  return null;
}

// ── Hook for graph data ──────────────────────────────────
export function useGraphsData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchGraphs = async () => {
      try {
        const r = await axios.get(`${API}/api/graphs-data`);
        const raw = r.data;

        // Transform backend data into the format Recharts expects
        setData({
          training_curves:         transformTrainingCurves(raw.training_curves),
          precision_recall:        transformPrecisionRecall(raw.precision_recall),
          bias_variance:           transformBiasVariance(raw.bias_variance),
          accuracy_vs_size:        transformAccuracyVsSize(raw.accuracy_vs_size),
          confidence_distribution: transformConfidenceDist(raw.confidence_distribution),
        });
      } catch (e) {
        setError(e);
        // Use fallback simulated data so the dashboard still works
        setData({
          training_curves:         generateFallbackTrainingCurves(),
          precision_recall:        generateFallbackPrecisionRecall(),
          bias_variance:           generateFallbackBiasVariance(),
          accuracy_vs_size:        generateFallbackAccuracyVsSize(),
          confidence_distribution: generateFallbackConfidenceDist(),
        });
      } finally {
        setLoading(false);
      }
    };
    fetchGraphs();
  }, []);

  return { data, loading, error };
}

// ── Chart 1: Training curves ─────────────────────────────
export function TrainingCurveChart({ data, scenario }) {
  if (!data || !data[scenario]) return <div className="spinner"></div>;
  const chartData = data[scenario];

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <ResponsiveContainer>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLORS.grid} />
          <XAxis 
            dataKey="epoch" 
            tick={{ fill: COLORS.text, fontSize: 11 }} 
            axisLine={{ stroke: COLORS.grid }}
          />
          <YAxis 
            tick={{ fill: COLORS.text, fontSize: 11 }} 
            axisLine={{ stroke: COLORS.grid }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#fff', border: '1px solid #E5E7EB', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
          <Line 
            type="monotone" 
            dataKey="train_loss" 
            stroke={COLORS.blue} 
            strokeWidth={2.5} 
            dot={false}
            name="Training Loss" 
          />
          <Line 
            type="monotone" 
            dataKey="val_loss" 
            stroke={COLORS.amber} 
            strokeWidth={2.5} 
            dot={false}
            name="Validation Loss" 
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Chart 2: Precision–Recall Curve ──────────────────────
export function PrecisionRecallChart({ data }) {
  if (!data || !data.points) return <div className="spinner"></div>;

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <ResponsiveContainer>
        <AreaChart data={data.points} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLORS.grid} />
          <XAxis 
            dataKey="recall" 
            type="number" 
            domain={[0, 1]} 
            tick={{ fill: COLORS.text, fontSize: 11 }}
            axisLine={{ stroke: COLORS.grid }}
          />
          <YAxis 
            domain={[0, 1]} 
            tick={{ fill: COLORS.text, fontSize: 11 }}
            axisLine={{ stroke: COLORS.grid }}
          />
          <Tooltip />
          <Area 
            type="stepAfter" 
            dataKey="precision" 
            stroke={COLORS.blue} 
            fill={`${COLORS.blue}15`} 
            strokeWidth={2} 
            name="Precision"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Chart 3: Bias vs Variance ────────────────────────────
export function BiasVarianceChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="spinner"></div>;

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="complexity" label={{ value: 'Model Complexity', position: 'insideBottom', offset: -5, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: '11px' }} />
          <Line type="monotone" dataKey="bias" stroke={COLORS.red} strokeWidth={2} dot={false} name="Bias (Error)" />
          <Line type="monotone" dataKey="variance" stroke={COLORS.blue} strokeWidth={2} dot={false} name="Variance" />
          <Line type="monotone" dataKey="total_error" stroke={COLORS.amber} strokeWidth={2.5} strokeDasharray="5 5" dot={false} name="Total Error" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Chart 4: Accuracy vs Size ─────────────────────────────
export function AccuracyVsSizeChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="spinner"></div>;

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="size" tick={{ fontSize: 11 }} />
          <YAxis domain={[50, 100]} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: '11px' }} />
          <Line type="monotone" dataKey="train_acc" stroke={COLORS.blue} strokeWidth={2} name="Train Acc" />
          <Line type="monotone" dataKey="val_acc" stroke={COLORS.green} strokeWidth={2} name="Val Acc" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Chart 5: Confidence Dist ──────────────────────────────
export function ConfidenceDistChart({ data }) {
  if (!data || !Array.isArray(data) || data.length === 0) return <div className="spinner"></div>;

  return (
    <div style={{ height: '300px', width: '100%' }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLORS.grid} />
          <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Tooltip cursor={{ fill: '#F9FAFB' }} />
          <Bar dataKey="count" fill={COLORS.blue} radius={[4, 4, 0, 0]} name="Frame Count" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
