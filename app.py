"""
Drone Detection System — Streamlit App
Run: streamlit run app.py
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drone Detection",
    page_icon="🚁",
    layout="wide",
)

st.markdown("""
<style>
.drone-alert {
    background: #e53935;
    color: white;
    font-size: 1.4rem;
    font-weight: bold;
    text-align: center;
    padding: 16px;
    border-radius: 8px;
    animation: blink 1s step-start infinite;
}
.safe-box {
    background: #43a047;
    color: white;
    font-size: 1.1rem;
    text-align: center;
    padding: 14px;
    border-radius: 8px;
}
@keyframes blink { 50% { opacity: 0.4; } }
</style>
""", unsafe_allow_html=True)


# ─── Find best weights ────────────────────────────────────────────────────────

def find_best_weights() -> str:
    candidates = sorted(
        (ROOT / "runs" / "train").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if (ROOT / "runs" / "train").exists() else []
    return str(candidates[0]) if candidates else ""


# ─── Load model (cached) ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_detector(weights: str, conf: float, iou: float):
    from src.detect import DroneDetector
    return DroneDetector(weights=weights, conf=conf, iou=iou)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    default_weights = find_best_weights()
    weights = st.text_input(
        "Model weights (.pt)",
        value=default_weights,
        placeholder="runs/train/drone_detector/weights/best.pt",
        help="Path to your trained best.pt file",
    )

    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05)
    iou  = st.slider("NMS IoU threshold",    0.10, 0.95, 0.45, 0.05)

    st.markdown("---")
    if not weights:
        st.warning("No weights found. Train the model first:\n```\npython src/train.py\n```")
    else:
        st.success(f"Model: `{Path(weights).name}`")

    st.caption("Drone Detection · YOLOv8")


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🚁 Drone Detection System")

if not weights:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

tab_image, tab_video = st.tabs(["📷 Image", "🎬 Video"])


# ═══════════════════════════════════════════
# TAB 1 — Image
# ═══════════════════════════════════════════

with tab_image:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Original", use_container_width=True)

        with st.spinner("Detecting …"):
            detector   = load_detector(weights, conf, iou)
            t0         = time.perf_counter()
            detections = detector.predict(img)
            ms         = (time.perf_counter() - t0) * 1000
            annotated  = detector.draw(img, detections)

        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Result", use_container_width=True)

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Drones detected", len(detections))
        c2.metric("Inference time",  f"{ms:.0f} ms")
        c3.metric("Top confidence",
                  f"{max(d['confidence'] for d in detections):.2f}" if detections else "—")

        # Alert
        if detections:
            st.markdown(
                f'<div class="drone-alert">⚠️ DRONE DETECTED — {len(detections)} drone(s) in frame</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="safe-box">✅ No drone detected</div>',
                        unsafe_allow_html=True)

        # Detection details table
        if detections:
            st.subheader("Detections")
            rows = []
            for i, d in enumerate(detections, 1):
                x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
                rows.append({
                    "#": i,
                    "Confidence": f"{d['confidence']:.3f}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "W": x2 - x1, "H": y2 - y1,
                })
            st.dataframe(rows, use_container_width=True)

        # Download
        _, buf = cv2.imencode(".jpg", annotated)
        st.download_button("💾 Download result", buf.tobytes(),
                           "drone_detection.jpg", "image/jpeg")


# ═══════════════════════════════════════════
# TAB 2 — Video
# ═══════════════════════════════════════════

with tab_video:
    vid_file   = st.file_uploader("Upload a video",
                                   type=["mp4", "avi", "mov", "mkv"],
                                   key="vid")
    frame_skip = st.slider("Process every Nth frame (1 = every frame)", 1, 10, 1)

    if vid_file and st.button("▶ Run Detection"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(vid_file.read())
            in_path = Path(tmp_in.name)

        out_path = in_path.with_stem(in_path.stem + "_out")

        detector    = load_detector(weights, conf, iou)
        cap         = cv2.VideoCapture(str(in_path))
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (vid_w, vid_h),
        )

        progress     = st.progress(0, text="Processing …")
        drone_frames = 0
        last_dets    = []
        idx          = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % frame_skip == 0:
                last_dets = detector.predict(frame)
                if last_dets:
                    drone_frames += 1

            writer.write(detector.draw(frame, last_dets))

            if total > 0:
                progress.progress(min(idx / total, 1.0),
                                   text=f"Frame {idx}/{total}")
            idx += 1

        cap.release()
        writer.release()
        progress.empty()

        st.success(f"Done! {idx} frames processed. "
                   f"Drone detected in {drone_frames} frames.")

        with open(out_path, "rb") as f:
            st.download_button("💾 Download annotated video",
                               f.read(), "drone_result.mp4", "video/mp4")

        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
