"""
Test-set evaluation for the Drone Detection System.

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights runs/train/drone_detector/weights/best.pt
    python src/evaluate.py --weights best.pt --conf 0.35 --iou 0.45 --max-vis 50

Outputs (all under runs/eval/<timestamp>/):
    metrics.json          — mAP50, mAP50-95, precision, recall
    results_per_image.csv — per-image TP/FP/FN + inference time
    visualizations/       — side-by-side GT vs Prediction PNGs (up to --max-vis)
    summary.png           — bar chart of key metrics
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_best_weights() -> Path:
    candidates = sorted(
        (ROOT / "runs" / "train").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No best.pt found under runs/train/. Train first.")
    return candidates[0]


def load_gt_boxes(label_path: Path) -> list[list[float]]:
    """Read YOLO-format label file → list of [cx, cy, w, h] (normalised)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append([float(x) for x in parts[1:]])   # skip class id
    return boxes


def yolo_to_xyxy(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


def iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_boxes_xyxy, pred_boxes_xyxy, iou_thresh=0.5):
    """Simple greedy matching → returns (tp, fp, fn)."""
    matched_gt = set()
    tp = 0
    for pred in pred_boxes_xyxy:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes_xyxy):
            if i in matched_gt:
                continue
            v = iou(pred, gt)
            if v > best_iou:
                best_iou, best_idx = v, i
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred_boxes_xyxy) - tp
    fn = len(gt_boxes_xyxy) - tp
    return tp, fp, fn


def draw_side_by_side(img, gt_xyxy, pred_dets, img_name: str) -> np.ndarray:
    """Returns a side-by-side BGR image: GT on left, Predictions on right."""
    h, w = img.shape[:2]
    gt_img   = img.copy()
    pred_img = img.copy()

    # Draw ground truth (blue)
    for x1, y1, x2, y2 in gt_xyxy:
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (210, 100, 0), 2)
        cv2.putText(gt_img, "GT", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 100, 0), 1, cv2.LINE_AA)

    # Draw predictions (green)
    for det in pred_dets:
        x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        lbl = f"drone {det['confidence']:.2f}"
        cv2.putText(pred_img, lbl, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA)

    # Header bars
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    gt_bar   = bar.copy(); gt_bar[:]   = (180, 80, 0)
    pred_bar = bar.copy(); pred_bar[:] = (0, 140, 0)
    cv2.putText(gt_bar,   "GROUND TRUTH",  (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(pred_bar, "PREDICTION",    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    left  = np.vstack([gt_bar,   gt_img])
    right = np.vstack([pred_bar, pred_img])

    # Divider
    divider = np.zeros((left.shape[0], 4, 3), dtype=np.uint8)
    return np.hstack([left, divider, right])


def make_summary_chart(metrics: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
        values = [
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("map50", 0),
            metrics.get("map50_95", 0),
        ]
        colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Test Set Metrics — Drone Detection (YOLOv8n)")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except ImportError:
        print("  [warn] matplotlib not installed — skipping summary chart")


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(weights: str, conf: float, iou_thresh: float, max_vis: int):
    from ultralytics import YOLO
    import pandas as pd

    out_dir = ROOT / "runs" / "eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Drone Detection — Test Set Evaluation")
    print(f"  Weights : {weights}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}\n")

    # ── 1. Official YOLO val on test split ──────────────────────────────────
    print("[1/3] Running YOLO val on test split …")
    model = YOLO(weights)
    val_results = model.val(
        data=str(ROOT / "configs" / "drone_dataset.yaml"),
        split="test",
        conf=conf,
        iou=iou_thresh,
        verbose=False,
        save=False,
    )

    metrics = {
        "precision": float(val_results.box.mp),
        "recall":    float(val_results.box.mr),
        "map50":     float(val_results.box.map50),
        "map50_95":  float(val_results.box.map),
    }
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  mAP50     : {metrics['map50']:.4f}")
    print(f"  mAP50-95  : {metrics['map50_95']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── 2. Per-image loop ────────────────────────────────────────────────────
    print("\n[2/3] Running per-image inference …")
    img_dir   = ROOT / "data" / "test" / "images"
    label_dir = ROOT / "data" / "test" / "labels"
    img_paths = sorted(img_dir.glob("*"))

    rows = []
    vis_count = 0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        gt_yolo = load_gt_boxes(label_dir / (img_path.stem + ".txt"))
        gt_xyxy = [yolo_to_xyxy(b, w, h) for b in gt_yolo]

        # Prediction
        t0 = time.perf_counter()
        results = model.predict(source=img, conf=conf, iou=iou_thresh,
                                device="0" if _cuda_available() else "cpu",
                                verbose=False)
        ms = (time.perf_counter() - t0) * 1000

        pred_dets = []
        for r in results:
            for box in r.boxes:
                pred_dets.append({
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy":  box.xyxy[0].tolist(),
                })

        pred_xyxy = [d["bbox_xyxy"] for d in pred_dets]
        tp, fp, fn = match_detections(gt_xyxy, pred_xyxy)

        rows.append({
            "image":      img_path.name,
            "gt_count":   len(gt_xyxy),
            "pred_count": len(pred_dets),
            "TP": tp, "FP": fp, "FN": fn,
            "inf_ms":     round(ms, 1),
        })

        # Save visualization for first max_vis images
        if vis_count < max_vis:
            vis = draw_side_by_side(img, gt_xyxy, pred_dets, img_path.stem)
            cv2.imwrite(str(vis_dir / f"{img_path.stem}_vis.jpg"), vis)
            vis_count += 1

    # ── 3. Save per-image CSV ────────────────────────────────────────────────
    print(f"\n[3/3] Saving results …")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_per_image.csv", index=False)

    total_tp = df["TP"].sum()
    total_fp = df["FP"].sum()
    total_fn = df["FN"].sum()
    avg_ms   = df["inf_ms"].mean()

    print(f"  Images evaluated : {len(df)}")
    print(f"  Total TP / FP / FN : {total_tp} / {total_fp} / {total_fn}")
    print(f"  Avg inference time : {avg_ms:.1f} ms")
    print(f"  Visualizations saved : {vis_count}")

    make_summary_chart(metrics, out_dir / "summary.png")

    print(f"\n  Results → {out_dir}\n")
    return out_dir, metrics


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate drone detector on test set")
    parser.add_argument("--weights", default="", help="Path to best.pt (auto-detected if omitted)")
    parser.add_argument("--conf",    type=float, default=0.35)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--max-vis", type=int,   default=50,
                        help="Max side-by-side visualizations to save")
    args = parser.parse_args()

    w = args.weights or str(find_best_weights())
    evaluate(w, args.conf, args.iou, args.max_vis)
