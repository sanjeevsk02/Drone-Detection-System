"""
Drone Detector — wraps a YOLOv8 model for image and video inference.
Since the model is single-class (drone only), any detection = drone detected.

Tracking (ByteTrack) is available via predict_track() for video pipelines.
"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BOX_COLOR    = (0, 220, 0)    # green  — plain detections
TRACK_COLORS = [               # per-ID colours for tracked drones
    (0, 200, 255), (255, 120, 0), (180, 0, 255), (0, 255, 180),
    (255, 220, 0), (0, 140, 255), (200, 255, 0), (255, 0, 140),
]
TEXT_COLOR   = (0, 0, 0)      # black
ALERT_COLOR  = (0, 0, 210)    # red


def _track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


class DroneDetector:
    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.45):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.conf    = conf
        self.iou     = iou

        self.model   = YOLO(self.weights)
        self.device  = self._auto_device()

        # Re-ID state: map new ByteTrack IDs back to canonical IDs by position
        self._last_pos:  dict[int, tuple[float, float, int]] = {}  # id → (cx, cy, frame)
        self._id_map:    dict[int, int] = {}   # new_id → canonical_id
        self._known_ids: set[int] = set()
        self._frame_idx: int = 0
        self._REID_DIST  = 120   # pixels — max distance to re-match a lost drone
        self._REID_FRAMES = 300  # frames — how long to remember a lost drone position

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ── Stateless per-frame detection (image tab / no tracking) ──────────────

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

    # ── Stateful per-frame tracking (video tab with ByteTrack) ───────────────

    def predict_track(self, img: np.ndarray, imgsz: int = 1280) -> list[dict]:
        """
        Run ByteTrack on a BGR frame.
        Returns list of dicts: {confidence, bbox_xyxy, track_id}
        Call reset_tracker() between separate videos.
        Use imgsz=640 for live webcam feeds to improve FPS.
        """
        tracker_cfg = str(ROOT / "configs" / "bytetrack_drone.yaml")
        results = self.model.track(
            source=img,
            conf=self.conf,
            iou=0.30,
            imgsz=imgsz,
            device=self.device,
            tracker=tracker_cfg,
            persist=True,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                detections.append({
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy":  box.xyxy[0].tolist(),
                    "track_id":   track_id,
                })
        return self._apply_reid(detections)

    def reset_tracker(self):
        """Clear ByteTrack state between videos."""
        self.model.predictor = None
        self._last_pos.clear()
        self._id_map.clear()
        self._known_ids.clear()
        self._frame_idx = 0

    def _apply_reid(self, detections: list[dict]) -> list[dict]:
        """Remap new ByteTrack IDs back to canonical IDs using last known position."""
        self._frame_idx += 1

        # First pass: resolve canonical IDs for all current detections
        for d in detections:
            tid = d["track_id"]
            if tid < 0:
                continue
            if tid in self._id_map:
                d["track_id"] = self._id_map[tid]

        # Set of canonical IDs currently active in this frame
        active_canonical = {d["track_id"] for d in detections if d["track_id"] >= 0}

        # Second pass: try to re-match brand new IDs to recently lost tracks
        for d in detections:
            raw_tid = None
            # Find original ByteTrack ID for this detection
            for orig, can in self._id_map.items():
                if can == d["track_id"]:
                    raw_tid = orig
                    break
            if raw_tid is None:
                raw_tid = d["track_id"]

            canonical = d["track_id"]
            if canonical in self._known_ids:
                # Already known — just update position
                cx = (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2
                cy = (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2
                self._last_pos[canonical] = (cx, cy, self._frame_idx)
                continue

            # New canonical ID — try to match to a recently lost track
            cx = (d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2
            cy = (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2

            best_dist, best_old = float("inf"), None
            for old_id, (ox, oy, frame) in self._last_pos.items():
                if old_id in active_canonical:
                    continue
                if self._frame_idx - frame > self._REID_FRAMES:
                    continue
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_old = dist, old_id

            if best_old is not None and best_dist < self._REID_DIST:
                # Remap to the old canonical ID
                self._id_map[raw_tid] = best_old
                d["track_id"] = best_old
                active_canonical.discard(canonical)
                active_canonical.add(best_old)
                self._last_pos.pop(best_old, None)
                canonical = best_old

            self._known_ids.add(canonical)
            self._last_pos[canonical] = (cx, cy, self._frame_idx)

        return detections

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, img: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes. Works for both plain detect and tracked dicts."""
        out = img.copy()
        h, w = out.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
            track_id = det.get("track_id", -1)

            color = _track_color(track_id) if track_id >= 0 else BOX_COLOR
            label = (f"#{track_id} drone {det['confidence']:.2f}"
                     if track_id >= 0
                     else f"drone {det['confidence']:.2f}")

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

        # Alert banner at the bottom when drone(s) found
        if detections:
            cv2.rectangle(out, (0, h - 40), (w, h), ALERT_COLOR, -1)
            unique_ids = {d["track_id"] for d in detections if d.get("track_id", -1) >= 0}
            if unique_ids:
                msg = f"DRONE DETECTED  ({len(unique_ids)} tracked ID{'s' if len(unique_ids) != 1 else ''})"
            else:
                msg = f"DRONE DETECTED  ({len(detections)})"
            cv2.putText(out, msg, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return out
