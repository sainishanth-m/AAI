"""
metrics.py
==========
AI Evaluation Metrics for Object Detection
-------------------------------------------
Implements the core evaluation metrics used to assess object detection
model performance, along with explanations of fundamental AI concepts:
Bias, Variance, Overfitting, and Underfitting.

Author  : AI Project Team
Course  : Artificial Intelligence
"""

import numpy as np
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DETECTION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_iou(box_pred: list, box_gt: list) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between the predicted box and the ground-truth
    box. A value of 1.0 means perfect overlap; 0.0 means no overlap.

    Parameters
    ----------
    box_pred : list  [x1, y1, x2, y2]  — predicted bounding box
    box_gt   : list  [x1, y1, x2, y2]  — ground-truth bounding box

    Returns
    -------
    float : IoU value in [0, 1]

    Formula
    -------
        IoU = Area(Intersection) / Area(Union)
        Area(Union) = Area(pred) + Area(gt) - Area(Intersection)
    """
    # Coordinates of the intersection rectangle
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    # Intersection area (clamped to 0 if no overlap)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    # Individual box areas
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])

    union = area_pred + area_gt - intersection
    return round(intersection / union, 4) if union > 0 else 0.0


def compute_precision(tp: int, fp: int) -> float:
    """
    Compute Precision.

    Precision answers: "Of all the detections the model made, how many
    were actually correct?"

    Parameters
    ----------
    tp : int  — True Positives  (correct detections)
    fp : int  — False Positives (incorrect / spurious detections)

    Returns
    -------
    float : Precision in [0, 1]

    Formula
    -------
        Precision = TP / (TP + FP)
    """
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return round(tp / denominator, 4)


def compute_recall(tp: int, fn: int) -> float:
    """
    Compute Recall (Sensitivity / True Positive Rate).

    Recall answers: "Of all the real objects in the scene, how many did
    the model successfully detect?"

    Parameters
    ----------
    tp : int  — True Positives  (correctly detected objects)
    fn : int  — False Negatives (missed objects)

    Returns
    -------
    float : Recall in [0, 1]

    Formula
    -------
        Recall = TP / (TP + FN)
    """
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return round(tp / denominator, 4)


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute F1-Score (harmonic mean of Precision and Recall).

    Formula
    -------
        F1 = 2 × (Precision × Recall) / (Precision + Recall)
    """
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return round(2 * precision * recall / denominator, 4)


def compute_average_precision(
    precisions: list[float],
    recalls: list[float],
) -> float:
    """
    Compute Average Precision (AP) using the area under the P-R curve
    via the trapezoidal rule.

    AP summarises the Precision–Recall curve as a single scalar. It is
    computed by integrating precision over the recall axis.

    Parameters
    ----------
    precisions : list[float]  — precision values at each threshold
    recalls    : list[float]  — corresponding recall values

    Returns
    -------
    float : AP in [0, 1]
    """
    if not precisions or not recalls:
        return 0.0

    # Sort by recall (ascending)
    pairs = sorted(zip(recalls, precisions))
    recalls_sorted    = [r for r, _ in pairs]
    precisions_sorted = [p for _, p in pairs]

    # Prepend (0, 1) and append (1, 0) for proper integration bounds
    recalls_sorted    = [0.0] + recalls_sorted    + [1.0]
    precisions_sorted = [1.0] + precisions_sorted + [0.0]

    # Trapezoidal integration
    ap = float(np.trapz(precisions_sorted, recalls_sorted))
    return round(abs(ap), 4)


def compute_map(
    per_class_ap: dict[str, float],
) -> float:
    """
    Compute mean Average Precision (mAP) across all classes.

    mAP = (1/N) × Σ AP_i   for i in 1..N classes

    Parameters
    ----------
    per_class_ap : dict mapping class_name → AP value

    Returns
    -------
    float : mAP in [0, 1]
    """
    if not per_class_ap:
        return 0.0
    return round(sum(per_class_ap.values()) / len(per_class_ap), 4)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SIMULATED EXPERIMENT DATA
# ═════════════════════════════════════════════════════════════════════════════

def generate_training_curves(
    epochs: int = 50,
    scenario: str = "good_fit",
) -> dict:
    """
    Generate realistic simulated training/validation loss curves.

    Scenarios
    ---------
    'underfitting'  — both losses remain high (model too simple)
    'good_fit'      — training loss ↓, validation loss ↓ in parallel
    'overfitting'   — training loss ↓ fast, validation loss starts ↑

    Returns
    -------
    dict with keys: 'epochs', 'train_loss', 'val_loss'
    """
    np.random.seed(7)
    ep = np.arange(1, epochs + 1)
    noise = lambda scale: np.random.normal(0, scale, epochs)

    if scenario == "underfitting":
        # High bias — model can't learn well
        train_loss = 2.5 - 0.015 * ep + noise(0.05)
        val_loss   = 2.6 - 0.012 * ep + noise(0.06)

    elif scenario == "overfitting":
        # Low bias but high variance — memorises training data
        train_loss = 2.5 * np.exp(-0.12 * ep) + noise(0.02)
        val_loss   = (2.5 * np.exp(-0.07 * ep)
                      + 0.003 * (ep - 20).clip(0) ** 1.5
                      + noise(0.04))

    else:  # good_fit
        # Balanced learning
        train_loss = 2.5 * np.exp(-0.09 * ep) + noise(0.03)
        val_loss   = 2.5 * np.exp(-0.08 * ep) + noise(0.04)

    return {
        "epochs":     ep.tolist(),
        "train_loss": train_loss.clip(0.05).tolist(),
        "val_loss":   val_loss.clip(0.05).tolist(),
    }


def generate_precision_recall_curve(num_points: int = 20) -> dict:
    """
    Generate a realistic simulated Precision–Recall curve.

    Returns
    -------
    dict with keys: 'precisions', 'recalls', 'thresholds', 'ap'
    """
    np.random.seed(13)
    thresholds = np.linspace(0.95, 0.05, num_points)
    recalls    = np.linspace(0.0,  0.95, num_points) + np.random.normal(0, 0.02, num_points)
    recalls    = recalls.clip(0, 1)
    precisions = (1.0 - 0.6 * recalls ** 1.5
                  + np.random.normal(0, 0.02, num_points)).clip(0.1, 1.0)

    ap = compute_average_precision(precisions.tolist(), recalls.tolist())

    return {
        "precisions": precisions.tolist(),
        "recalls":    recalls.tolist(),
        "thresholds": thresholds.tolist(),
        "ap":         ap,
    }


def generate_bias_variance_data() -> dict:
    """
    Generate data illustrating the Bias–Variance trade-off.

    As model complexity increases:
      • High-complexity  → low bias,  high variance
      • Low-complexity   → high bias, low variance
      • Sweet spot       → balanced bias & variance

    Returns
    -------
    dict with keys: 'complexity', 'bias', 'variance', 'total_error'
    """
    np.random.seed(21)
    complexity  = np.arange(1, 16)
    bias        = 2.5 / (0.3 * complexity + 0.5)
    variance    = 0.05 * complexity ** 1.4
    total_error = bias + variance + np.random.normal(0, 0.05, len(complexity))

    return {
        "complexity":   complexity.tolist(),
        "bias":         bias.tolist(),
        "variance":     variance.tolist(),
        "total_error":  total_error.clip(0).tolist(),
    }


def generate_accuracy_vs_dataset_size() -> dict:
    """
    Show how accuracy improves with more training data.

    Returns
    -------
    dict with keys: 'dataset_sizes', 'train_accuracy', 'val_accuracy'
    """
    np.random.seed(33)
    sizes        = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    train_acc    = [0.98 - 20 / (s + 40) + np.random.uniform(-0.01, 0.01) for s in sizes]
    val_acc      = [0.93 - 50 / (s + 80) + np.random.uniform(-0.02, 0.02) for s in sizes]

    return {
        "dataset_sizes":  sizes,
        "train_accuracy": [round(min(a, 0.999), 4) for a in train_acc],
        "val_accuracy":   [round(min(a, 0.980), 4) for a in val_acc],
    }


def generate_confidence_distribution(n_detections: int = 500) -> dict:
    """
    Generate a simulated confidence-score distribution.

    Returns
    -------
    dict with keys: 'bins', 'counts'
    """
    np.random.seed(55)
    # Mix of high-confidence true detections and low-confidence noise
    high_conf = np.random.beta(8, 2, int(n_detections * 0.65))
    low_conf  = np.random.beta(2, 5, int(n_detections * 0.35))
    scores    = np.concatenate([high_conf, low_conf]).clip(0, 1)

    counts, bin_edges = np.histogram(scores, bins=20, range=(0, 1))
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2
                   for i in range(len(bin_edges) - 1)]

    return {
        "bins":   [round(b, 3) for b in bin_centers],
        "counts": counts.tolist(),
    }


def generate_experiment_comparison() -> dict:
    """
    Generate comparison data for three experimental scenarios:
      1. Small dataset  → Underfitting
      2. Medium dataset → Good fit
      3. Large dataset (over-trained) → Overfitting

    Returns
    -------
    dict with three scenario keys, each containing accuracy, precision,
    recall, f1, train_loss, val_loss.
    """
    return {
        "underfitting": {
            "label":      "Small Dataset (Underfitting)",
            "accuracy":   0.52,
            "precision":  0.48,
            "recall":     0.44,
            "f1":         0.46,
            "train_loss": 1.85,
            "val_loss":   1.92,
        },
        "good_fit": {
            "label":      "Medium Dataset (Good Fit)",
            "accuracy":   0.87,
            "precision":  0.85,
            "recall":     0.83,
            "f1":         0.84,
            "train_loss": 0.31,
            "val_loss":   0.36,
        },
        "overfitting": {
            "label":      "Large Dataset (Overfitting)",
            "accuracy":   0.99,
            "precision":  0.98,
            "recall":     0.97,
            "f1":         0.975,
            "train_loss": 0.04,
            "val_loss":   0.88,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AI CONCEPT EXPLANATIONS
# ═════════════════════════════════════════════════════════════════════════════

AI_CONCEPTS = {
    "bias": {
        "title": "Bias",
        "definition": (
            "Bias is the error introduced by approximating a real-world problem "
            "(which may be complex) by a simplified model. A model with HIGH bias "
            "pays little attention to the training data and oversimplifies the "
            "underlying relationship — leading to poor performance on both training "
            "and unseen data."
        ),
        "example": (
            "A linear regression line fitted to clearly non-linear data will have "
            "high bias because it can never capture the true curve."
        ),
        "effect":  "Underfitting — both training error and validation error are high.",
    },
    "variance": {
        "title": "Variance",
        "definition": (
            "Variance is the model's sensitivity to small fluctuations in the "
            "training data. A HIGH-variance model learns the training data too "
            "well — including its noise — and fails to generalise to new, unseen "
            "examples."
        ),
        "example": (
            "A very deep neural network trained on a small dataset may memorise "
            "every training sample but perform poorly on the test set."
        ),
        "effect":  "Overfitting — low training error but high validation error.",
    },
    "overfitting": {
        "title": "Overfitting",
        "definition": (
            "Overfitting occurs when a model learns the training data too closely, "
            "capturing noise and irrelevant patterns rather than the true underlying "
            "structure. The model performs excellently on training data but poorly "
            "on new data."
        ),
        "causes":   ["Too many parameters", "Too little training data",
                     "Training for too many epochs", "Lack of regularisation"],
        "remedies": ["Dropout", "L1/L2 regularisation", "Early stopping",
                     "Data augmentation", "Collecting more data"],
    },
    "underfitting": {
        "title": "Underfitting",
        "definition": (
            "Underfitting occurs when a model is too simple to capture the underlying "
            "patterns in the data. The model performs poorly on both training and "
            "test data."
        ),
        "causes":   ["Model too simple", "Too few training epochs",
                     "Excessive regularisation", "Insufficient features"],
        "remedies": ["Use a more complex model", "Train for more epochs",
                     "Reduce regularisation", "Feature engineering"],
    },
}
