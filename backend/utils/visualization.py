"""
visualization.py
================
Visualization utilities for object detection results.
Draws bounding boxes, labels, confidence bars, and
frame statistics onto OpenCV frames.

Author  : AI Project Team
Course  : Artificial Intelligence
"""

import cv2
import numpy as np
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (BGR)
# ─────────────────────────────────────────────────────────────────────────────
OVERLAY_BG   = (15,  15,  15)   # near-black HUD background
TEXT_WHITE   = (240, 240, 240)  # primary text
TEXT_GREEN   = ( 80, 220, 100)  # positive indicators
TEXT_YELLOW  = (  0, 215, 255)  # warning / mid-level
TEXT_RED     = ( 60,  60, 220)  # alert / high value
CONF_BAR_FG  = ( 50, 205, 130)  # confidence bar fill
CONF_BAR_BG  = ( 60,  60,  60)  # confidence bar background


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    show_confidence_bar: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels onto the frame for each detection.

    Parameters
    ----------
    frame        : BGR image (H × W × 3)
    detections   : list of dicts from YOLOModel.predict()
    show_confidence_bar : draw a small confidence progress bar on each box

    Returns
    -------
    np.ndarray : annotated frame (drawn in-place for performance)
    """
    # Draw directly on frame to avoid expensive copies
    output = frame
    h, w   = output.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label  = det["label"]
        conf   = det["confidence"]
        color  = det.get("color", (0, 200, 100))  # default green-ish

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # ── Label background chip (opaque, no expensive addWeighted) ─────
        text     = f"{label}  {conf:.0%}"
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.50
        thick    = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)

        chip_x1 = x1
        chip_y1 = max(y1 - th - 10, 0)
        chip_x2 = x1 + tw + 10
        chip_y2 = y1

        # Solid chip background (fast, no alpha blending needed)
        cv2.rectangle(output, (chip_x1, chip_y1), (chip_x2, chip_y2), color, -1)

        # Label text
        cv2.putText(output, text,
                    (chip_x1 + 5, chip_y2 - 4),
                    font, scale, TEXT_WHITE, thick, cv2.LINE_AA)

        # ── Confidence bar ────────────────────────────────────────────────
        if show_confidence_bar:
            bar_x1 = x1
            bar_y1 = y2 + 4
            bar_x2 = x2
            bar_y2 = y2 + 10
            bar_fill = int(x1 + (x2 - x1) * conf)

            cv2.rectangle(output, (bar_x1, bar_y1), (bar_x2, bar_y2),
                          CONF_BAR_BG, -1)
            cv2.rectangle(output, (bar_x1, bar_y1), (bar_fill, bar_y2),
                          CONF_BAR_FG, -1)

    return output


def draw_hud(
    frame: np.ndarray,
    detection_count: int,
    fps: float,
    inference_ms: float,
    model_name: str = "YOLOv8n",
) -> np.ndarray:
    """
    Draw a HUD (Heads-Up Display) overlay in the top-left corner showing
    real-time statistics: FPS, inference time, detected objects, model name.

    Parameters
    ----------
    frame           : annotated BGR frame
    detection_count : number of objects detected in this frame
    fps             : frames per second estimate
    inference_ms    : inference time in milliseconds
    model_name      : YOLO model identifier string

    Returns
    -------
    np.ndarray : frame with HUD overlay
    """
    output = frame
    h, w   = output.shape[:2]

    hud_w, hud_h = 220, 110

    # Solid dark background (no expensive alpha blending)
    cv2.rectangle(output, (8, 8), (8 + hud_w, 8 + hud_h), OVERLAY_BG, -1)

    # Header bar
    cv2.rectangle(output, (8, 8), (8 + hud_w, 28), (50, 100, 220), -1)
    cv2.putText(output, f"  {model_name} — Live Detection",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                TEXT_WHITE, 1, cv2.LINE_AA)

    # Stats lines
    stats = [
        (f"FPS:         {fps:.1f}",          TEXT_GREEN),
        (f"Inference:   {inference_ms:.1f} ms", TEXT_YELLOW),
        (f"Objects:     {detection_count}",   TEXT_WHITE),
        (f"Time:        {time.strftime('%H:%M:%S')}", TEXT_WHITE),
    ]
    for i, (text, color) in enumerate(stats):
        cv2.putText(output, text,
                    (16, 44 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    color, 1, cv2.LINE_AA)

    # Outer border
    cv2.rectangle(output, (8, 8), (8 + hud_w, 8 + hud_h),
                  (80, 120, 200), 1)

    return output


def encode_frame_to_jpeg(
    frame: np.ndarray,
    quality: int = 65,
) -> Optional[bytes]:
    """
    Encode a BGR frame to JPEG bytes for HTTP streaming (MJPEG).

    Parameters
    ----------
    frame   : BGR numpy array
    quality : JPEG quality (1–100). Lower = smaller file = faster streaming.

    Returns
    -------
    bytes | None : JPEG bytes, or None on encoding failure
    """
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        return None
    return buffer.tobytes()


def resize_frame(
    frame: np.ndarray,
    max_width: int = 640,
    max_height: int = 480,
) -> np.ndarray:
    """
    Resize frame to fit within max dimensions while preserving aspect ratio.
    """
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame
