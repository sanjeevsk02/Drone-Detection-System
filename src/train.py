"""
Train YOLOv8 for drone detection.
Prints accuracy metrics after every epoch and a final summary at the end.

Usage:
    python src/train.py
    python src/train.py --model yolov8s.pt --epochs 100 --batch 16
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "configs" / "drone_dataset.yaml"


# ─── Per-epoch callback ───────────────────────────────────────────────────────

def on_epoch_end(trainer) -> None:
    e       = trainer.epoch + 1
    epochs  = trainer.epochs
    metrics = trainer.metrics          # populated after validation each epoch
    loss    = trainer.loss.item() if hasattr(trainer.loss, "item") else float(trainer.loss)

    map50    = metrics.get("metrics/mAP50(B)",    0.0)
    map5095  = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision= metrics.get("metrics/precision(B)",0.0)
    recall   = metrics.get("metrics/recall(B)",   0.0)

    bar_len  = 20
    filled   = int(bar_len * e / epochs)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(
        f"\n[{bar}] Epoch {e:>3}/{epochs}"
        f"  loss={loss:.4f}"
        f"  mAP50={map50:.4f}"
        f"  mAP50-95={map5095:.4f}"
        f"  P={precision:.4f}"
        f"  R={recall:.4f}"
    )


# ─── Train ────────────────────────────────────────────────────────────────────

def train(model_name: str, epochs: int, batch: int, imgsz: int) -> Path:
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"Dataset config not found: {DATA_YAML}")
        print("Run:  python src/download_kaggle.py --local <your_dataset_folder>")
        sys.exit(1)

    device = _auto_device()

    print("\n" + "=" * 60)
    print("  Drone Detection — YOLOv8 Training")
    print("=" * 60)
    print(f"  Model  : {model_name}")
    print(f"  Data   : {DATA_YAML}")
    print(f"  Epochs : {epochs}")
    print(f"  Batch  : {batch}")
    print(f"  ImgSz  : {imgsz}")
    print(f"  Device : {'GPU (CUDA)' if device == '0' else 'CPU'}")
    print("=" * 60 + "\n")

    model = YOLO(model_name)
    model.add_callback("on_fit_epoch_end", on_epoch_end)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="drone_detector",
        exist_ok=True,
        single_cls=True,
        patience=20,
        verbose=False,     # suppress per-batch noise; we print our own epoch summary
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    _final_summary(model, best_weights)
    return best_weights


# ─── Final summary ────────────────────────────────────────────────────────────

def _final_summary(model, best_weights: Path) -> None:
    print("\n" + "=" * 60)
    print("  Training Complete — Final Validation")
    print("=" * 60)

    if not best_weights.exists():
        print("  best.pt not found.")
        return

    from ultralytics import YOLO
    best = YOLO(str(best_weights))
    metrics = best.val(data=str(DATA_YAML), verbose=False)

    box = metrics.box
    print(f"  mAP50        : {box.map50:.4f}  ({box.map50*100:.1f}%)")
    print(f"  mAP50-95     : {box.map:.4f}   ({box.map*100:.1f}%)")
    print(f"  Precision    : {box.mp:.4f}  ({box.mp*100:.1f}%)")
    print(f"  Recall       : {box.mr:.4f}  ({box.mr*100:.1f}%)")
    print("=" * 60)
    print(f"\n  Best weights saved to:\n  {best_weights}\n")
    print("  Run the app:  streamlit run app.py")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 drone detector")
    p.add_argument("--model",  default="yolov8n.pt", help="Base model (yolov8n/s/m/l/x.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--imgsz",  type=int, default=640)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.model, args.epochs, args.batch, args.imgsz)
