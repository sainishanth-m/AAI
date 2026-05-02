# Real-Time Object Detection using YOLO and Webcam
# =========================================================================
# Project entry-point: run this file to start the Flask backend server.
# =========================================================================

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from backend.app import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  [START] YOLO Vision -- AI Object Detection Backend")
    print("="*60)
    print("  Server  : http://localhost:5000")
    print("  Docs    : See README.md for full API reference")
    print("="*60 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
