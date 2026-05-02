import React from 'react';

/* ──────────────────────────────────────────────────────────
   About.js
   Theory and documentation in a clean academic style.
────────────────────────────────────────────────────────── */

function ConceptSection({ title, children }) {
  return (
    <div style={{ marginBottom: '48px' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '20px', borderBottom: '2px solid #E5E7EB', paddingBottom: '8px' }}>
        {title}
      </h2>
      <div style={{ fontSize: '1rem', color: '#374151', lineHeight: 1.7 }}>
        {children}
      </div>
    </div>
  );
}

export default function About() {
  return (
    <div className="animate-fade-in" style={{ maxWidth: '900px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '64px' }}>
        <h1 className="section-title">Academic Theory & Concepts</h1>
        <p className="section-sub">
          Detailed documentation on YOLO architecture, evaluation metrics, and model performance analysis.
        </p>
      </div>

      <ConceptSection title="1. YOLO Architecture Overview">
        <p style={{ marginBottom: '16px' }}>
          <strong>YOLO (You Only Look Once)</strong> is a state-of-the-art, real-time object detection system. Unlike traditional detectors that perform detection in multiple stages, YOLO treats object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probabilities.
        </p>
        <p style={{ marginBottom: '16px' }}>
          This project utilizes <strong>YOLOv8</strong>, the latest iteration by Ultralytics, which introduces an anchor-free design and architectural improvements that significantly boost both speed and accuracy.
        </p>
        <ul style={{ paddingLeft: '20px' }}>
          <li style={{ marginBottom: '8px' }}><strong>Backbone:</strong> Feature extraction using a Modified CSPDarknet53.</li>
          <li style={{ marginBottom: '8px' }}><strong>Neck:</strong> PANet for multi-scale feature fusion.</li>
          <li><strong>Head:</strong> Decoupled head for predicting classes and bounding boxes separately.</li>
        </ul>
      </ConceptSection>

      <ConceptSection title="2. Evaluation Metrics">
        <div className="grid-2" style={{ gap: '24px' }}>
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: '#2563EB' }}>IoU (Intersection over Union)</h3>
            <p style={{ fontSize: '0.875rem' }}>
              Measures the overlap between the predicted bounding box and the ground truth. An IoU &gt; 0.5 is typically required for a "True Positive" detection.
            </p>
          </div>
          <div className="card">
            <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: '#10B981' }}>mAP (mean Average Precision)</h3>
            <p style={{ fontSize: '0.875rem' }}>
              The primary benchmark for object detection. It is the mean of Average Precision calculated across all categories at specific IoU thresholds.
            </p>
          </div>
        </div>
      </ConceptSection>

      <ConceptSection title="3. Bias-Variance Tradeoff">
        <p style={{ marginBottom: '24px' }}>
          In machine learning, the goal is to minimize two sources of error that prevent supervised learning algorithms from generalizing beyond their training set:
        </p>
        <div className="card" style={{ borderLeft: '4px solid #EF4444', marginBottom: '16px' }}>
          <h4 style={{ fontWeight: 700, marginBottom: '4px' }}>Bias</h4>
          <p style={{ fontSize: '0.9375rem' }}>
            Error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (Underfitting).
          </p>
        </div>
        <div className="card" style={{ borderLeft: '4px solid #2563EB' }}>
          <h4 style={{ fontWeight: 700, marginBottom: '4px' }}>Variance</h4>
          <p style={{ fontSize: '0.9375rem' }}>
            Error due to too much complexity in the learning algorithm. High variance can cause an algorithm to model the random noise in the training data (Overfitting).
          </p>
        </div>
      </ConceptSection>

      <div className="card" style={{ marginTop: '64px', backgroundColor: '#F3F4F6', border: 'none', textAlign: 'center' }}>
        <h3 style={{ marginBottom: '8px' }}>Project Information</h3>
        <p style={{ fontSize: '0.875rem', color: '#6B7280' }}>
          Real-Time Object Detection System • v1.0.0 <br />
          Developed as part of the "Applications of AI" course. <br />
          Developed by: AI Group Project
        </p>
      </div>
    </div>
  );
}
