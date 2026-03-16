"""
Prepare the Kaggle drone dataset for YOLOv8 training.

Dataset : https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav
Format  : Pascal VOC XML  →  converted to YOLO .txt by this script

Usage:
    # Point to wherever you extracted the Kaggle zip
    python src/download_kaggle.py --local "C:/Users/sanju/Downloads/drone-dataset"
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─── XML → YOLO conversion ────────────────────────────────────────────────────

def xml_to_yolo(xml_path: Path) -> list[str]:
    """Convert one Pascal VOC XML file to YOLO format lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    size  = root.find("size")
    img_w = int(size.findtext("width",  "0"))
    img_h = int(size.findtext("height", "0"))
    if img_w == 0 or img_h == 0:
        return []

    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))

        xc = max(0.0, min(1.0, ((xmin + xmax) / 2) / img_w))
        yc = max(0.0, min(1.0, ((ymin + ymax) / 2) / img_h))
        bw = max(0.0, min(1.0, (xmax - xmin) / img_w))
        bh = max(0.0, min(1.0, (ymax - ymin) / img_h))

        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")  # class 0 = drone

    return lines


# ─── Collect pairs ────────────────────────────────────────────────────────────

def collect_pairs(src: Path, tmp_labels: Path) -> list[tuple[Path, Path]]:
    tmp_labels.mkdir(parents=True, exist_ok=True)
    pairs   = []
    skipped = 0

    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print(f"Found {len(images)} images. Converting XML → YOLO …")

    for img in images:
        xml = img.with_suffix(".xml")
        if not xml.exists():
            skipped += 1
            continue

        lines = xml_to_yolo(xml)
        if not lines:
            skipped += 1
            continue

        txt = tmp_labels / (img.stem + ".txt")
        txt.write_text("\n".join(lines))
        pairs.append((img, txt))

    print(f"Converted {len(pairs)} pairs  |  skipped {skipped}")
    return pairs


# ─── Split & copy ─────────────────────────────────────────────────────────────

def split_and_copy(pairs: list, ratios=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    random.shuffle(pairs)

    n  = len(pairs)
    n1 = int(n * ratios[0])
    n2 = int(n * ratios[1])

    splits = {
        "train": pairs[:n1],
        "valid": pairs[n1:n1 + n2],
        "test":  pairs[n1 + n2:],
    }

    for split, items in splits.items():
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in items:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)

        print(f"  {split:6s}: {len(items)} images")


# ─── Write configs/drone_dataset.yaml ────────────────────────────────────────

def write_yaml():
    cfg = {
        "path":  str(DATA_DIR),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    1,
        "names": ["drone"],
    }
    yaml_path = ROOT / "configs" / "drone_dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved → {yaml_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(local_dir: Path):
    if not local_dir.exists():
        print(f"Folder not found: {local_dir}")
        sys.exit(1)

    tmp_labels = DATA_DIR / "_tmp_labels"

    pairs = collect_pairs(local_dir, tmp_labels)
    if not pairs:
        print("No valid image+XML pairs found. Check the folder path.")
        sys.exit(1)

    print(f"\nSplitting {len(pairs)} pairs (80 / 10 / 10) …")
    split_and_copy(pairs)

    write_yaml()

    shutil.rmtree(tmp_labels, ignore_errors=True)

    print("\n" + "=" * 45)
    print("  Dataset ready!")
    print("=" * 45)
    print("  Next:  python src/train.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local", required=True,
                   help="Path to extracted Kaggle dataset folder")
    args = p.parse_args()
    run(Path(args.local))
