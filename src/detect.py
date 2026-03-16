"""
Drone Detector — wraps a YOLOv8 model for image and video inference.
Since the model is single-class (drone only), any detection = drone detected.
"""

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BOX_COLOR   = (0, 220, 0)    # green
TEXT_COLOR  = (0, 0, 0)      # black
ALERT_COLOR = (0, 0, 210)    # red


class DroneDetector:
    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.45):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.conf    = conf
        self.iou     = iou

        self.model   = YOLO(self.weights)
        self.device  = self._auto_device()

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(self, img: np.ndarray) -> list[dict]:
        """
        Run inference on a BGR numpy image.
        Returns list of dicts: {confidence, bbox_xyxy}
        Single-class model — every detection is a drone.
        """
        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy":  box.xyxy[0].tolist(),
                })
        return detections

    def draw(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
            label = f"drone {det['confidence']:.2f}"

            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), BOX_COLOR, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

        # Alert banner at the bottom when drone(s) found
        if detections:
            cv2.rectangle(out, (0, h - 40), (w, h), ALERT_COLOR, -1)
            msg = f"DRONE DETECTED  ({len(detections)})"
            cv2.putText(out, msg, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return out
