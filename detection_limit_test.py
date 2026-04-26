"""
Detection limit ablation study.
Progressively shrinks each test image and finds the minimum drone size
at which the model can still detect a drone.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WEIGHTS   = str(ROOT / "runs" / "train" / "drone_detector" / "weights" / "best.pt")
IMG_DIR   = ROOT / "data" / "test" / "images"
LABEL_DIR = ROOT / "data" / "test" / "labels"
CONF      = 0.20
IOU       = 0.45

# Scale factors to test (fraction of original size)
SCALES = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05]

from ultralytics import YOLO
model = YOLO(WEIGHTS)

def get_drone_pixel_size(img_path, label_path):
    """Return the largest drone bounding box area in the original image (pixels²)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]
    if not label_path.exists():
        return None, None
    sizes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        sizes.append(bw * w * bh * h)  # pixel area
    return img, max(sizes) if sizes else None


results_per_scale = {s: {"detected": 0, "total": 0} for s in SCALES}
min_detected_area = []   # pixel areas that were successfully detected
max_missed_area   = []   # pixel areas that were NOT detected

img_paths = sorted(IMG_DIR.glob("*.png")) + sorted(IMG_DIR.glob("*.jpg"))
print(f"Testing on {len(img_paths)} images across {len(SCALES)} scale factors...\n")

for img_path in img_paths:
    label_path = LABEL_DIR / (img_path.stem + ".txt")
    orig_img, orig_area = get_drone_pixel_size(img_path, label_path)
    if orig_img is None or orig_area is None:
        continue

    h, w = orig_img.shape[:2]

    for scale in SCALES:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(orig_img, (new_w, new_h))

        r = model.predict(source=resized, conf=CONF, iou=IOU, verbose=False)
        detected = any(len(res.boxes) > 0 for res in r)

        results_per_scale[scale]["total"] += 1
        if detected:
            results_per_scale[scale]["detected"] += 1
            min_detected_area.append(orig_area * scale * scale)
        else:
            max_missed_area.append(orig_area * scale * scale)

print("=" * 55)
print(f"{'Scale':>8}  {'Detected':>10}  {'Total':>7}  {'Rate':>8}")
print("-" * 55)
for scale in SCALES:
    d = results_per_scale[scale]["detected"]
    t = results_per_scale[scale]["total"]
    rate = d / t * 100 if t > 0 else 0
    print(f"{scale:>8.2f}  {d:>10}  {t:>7}  {rate:>7.1f}%")
print("=" * 55)

if min_detected_area:
    print(f"\nSmallest drone area still detected : {min(min_detected_area):.0f} px²  "
          f"({min(min_detected_area)**0.5:.1f} × {min(min_detected_area)**0.5:.1f} px equiv.)")
if max_missed_area:
    print(f"Largest drone area missed           : {max(max_missed_area):.0f} px²  "
          f"({max(max_missed_area)**0.5:.1f} × {max(max_missed_area)**0.5:.1f} px equiv.)")

# Find the scale where detection rate first drops below 50%
threshold_scale = None
for scale in SCALES:
    d = results_per_scale[scale]["detected"]
    t = results_per_scale[scale]["total"]
    if t > 0 and d / t < 0.50:
        threshold_scale = scale
        break

if threshold_scale:
    print(f"\nDetection rate drops below 50% at scale {threshold_scale:.2f} "
          f"(image shrunk to {threshold_scale*100:.0f}% of original size)")
