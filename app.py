"""
Drone Detection System — Streamlit App
Run: python -m streamlit run app.py
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

st.set_page_config(
    page_title="Drone Detection",
    page_icon=None,
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def find_best_weights() -> str:
    candidates = sorted(
        (ROOT / "runs" / "train").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if (ROOT / "runs" / "train").exists() else []
    return str(candidates[0]) if candidates else ""

@st.cache_resource(show_spinner="Loading model …")
def load_detector(weights: str, conf: float):
    from src.detect import DroneDetector
    return DroneDetector(weights=weights, conf=conf, iou=0.45)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.20, 0.05)
    st.markdown("---")
    weights = find_best_weights()
    if not weights:
        st.warning("No model found. Run `python src/train.py` first.")
    else:
        st.success("Model loaded ✓")
    st.caption("Drone Detection · YOLOv8n")


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("Drone Detection System")

if not weights:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

tab_image, tab_video, tab_live, tab_eval = st.tabs(["📷 Image", "🎬 Video", "📡 Live Camera", "📊 Evaluation"])


# ═══════════════════════════════════════════
# TAB 1 — Image
# ═══════════════════════════════════════════

with tab_image:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        with st.spinner("Detecting …"):
            detector   = load_detector(weights, conf)
            t0         = time.perf_counter()
            detections = detector.predict(img)
            ms         = (time.perf_counter() - t0) * 1000
            annotated  = detector.draw(img, detections)

        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Drones detected", len(detections))
        c2.metric("Inference time",  f"{ms:.0f} ms")
        c3.metric("Top confidence",
                  f"{max(d['confidence'] for d in detections):.2f}" if detections else "—")

        if detections:
            st.markdown(
                f'<div class="drone-alert">⚠️ DRONE DETECTED — {len(detections)} drone(s) in frame</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="safe-box">✅ No drone detected</div>', unsafe_allow_html=True)

        _, buf = cv2.imencode(".jpg", annotated)
        st.download_button("💾 Download result", buf.tobytes(), "drone_detection.jpg", "image/jpeg")


# ═══════════════════════════════════════════
# TAB 2 — Video
# ═══════════════════════════════════════════

with tab_video:
    vid_file     = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    use_tracking = st.toggle("Enable object tracking", value=True,
                             help="Assigns a persistent ID to each drone across frames")

    with st.expander("Line crossing counter"):
        line_enabled = st.checkbox("Enable")
        line_axis    = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True)
        if line_axis == "Horizontal":
            line_pos_pct = st.slider("Line position (% from top)", 10, 90, 50)
        else:
            line_pos_pct = st.slider("Line position (% from left)", 10, 90, 50)

    if vid_file and st.button("▶ Run Detection"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
            tmp_in.write(vid_file.read())
            in_path = Path(tmp_in.name)

        out_path = in_path.with_stem(in_path.stem + "_out")

        detector = load_detector(weights, conf)
        if use_tracking:
            detector.reset_tracker()

        cap   = cv2.VideoCapture(str(in_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h),
        )

        # Horizontal line: position is Y (% of height). Vertical: position is X (% of width)
        line_px = int(line_pos_pct / 100 * (vid_h if line_axis == "Horizontal" else vid_w))

        progress    = st.progress(0, text="Processing …")
        fps_display = st.empty()
        drone_frames = 0
        last_dets    = []
        all_track_ids: set[int] = set()
        crossing_count: int = 0
        counted: set[int] = set()       # IDs already counted for this line touch
        off_line: dict[int, int] = {}   # ID → frame when it left the line

        idx = 0
        t_last_fps = time.perf_counter()
        fps_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_frames += 1
            now = time.perf_counter()
            if now - t_last_fps >= 1.0:
                fps_display.markdown(f"**Processing speed:** {fps_frames / (now - t_last_fps):.1f} fps")
                fps_frames = 0
                t_last_fps = now

            last_dets = detector.predict_track(frame) if use_tracking else detector.predict(frame)

            if last_dets:
                drone_frames += 1

            for d in last_dets:
                tid = d.get("track_id", -1)
                if tid >= 0:
                    all_track_ids.add(tid)

            if line_enabled and use_tracking:
                ON_LINE_IDS = set()
                for d in last_dets:
                    tid = d.get("track_id", -1)
                    if tid < 0:
                        continue
                    x1, y1, x2, y2 = d["bbox_xyxy"]
                    is_touching = (y1 <= line_px <= y2) if line_axis == "Horizontal" else (x1 <= line_px <= x2)
                    if is_touching:
                        ON_LINE_IDS.add(tid)
                        off_line.pop(tid, None)
                        if tid not in counted:
                            crossing_count += 1
                            counted.add(tid)
                    else:
                        if tid in counted and tid not in off_line:
                            off_line[tid] = idx
                # Reset counted for drones that have been off the line for 20+ frames
                for tid in list(off_line):
                    if idx - off_line[tid] >= 20:
                        counted.discard(tid)
                        del off_line[tid]

            annotated = detector.draw(frame, last_dets)

            if line_enabled:
                color = (0, 220, 220)
                if line_axis == "Horizontal":
                    cv2.line(annotated, (0, line_px), (vid_w, line_px), color, 2)
                    cv2.putText(annotated, f"Crossings: {crossing_count}",
                                (8, line_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                else:
                    cv2.line(annotated, (line_px, 0), (line_px, vid_h), color, 2)
                    cv2.putText(annotated, f"Crossings: {crossing_count}",
                                (line_px + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            writer.write(annotated)

            if total > 0:
                progress.progress(min(idx / total, 1.0), text=f"Frame {idx}/{total}")
            idx += 1

        cap.release()
        writer.release()
        progress.empty()
        fps_display.empty()

        st.success(f"Done! {idx} frames processed.")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Frames with drones", drone_frames)
        sc2.metric("Unique drones tracked", len(all_track_ids) if use_tracking else "—")
        if line_enabled and use_tracking:
            sc3.metric("Line crossings", crossing_count)

        with open(out_path, "rb") as f:
            st.download_button("💾 Download annotated video", f.read(), "drone_result.mp4", "video/mp4")

        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════
# TAB 3 — Live Camera
# ═══════════════════════════════════════════

with tab_live:
    st.markdown("### Live Drone Detection")
    st.caption("Runs the detector on your webcam feed in real time.")

    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av

    class _DroneProcessor(VideoProcessorBase):
        def __init__(self):
            from src.detect import DroneDetector
            self.detector    = DroneDetector(weights=weights, conf=conf, iou=0.45)
            self.unique_ids: set[int] = set()
            self._prev_time  = time.perf_counter()

        def recv(self, frame):
            now            = time.perf_counter()
            fps            = 1.0 / max(now - self._prev_time, 1e-6)
            self._prev_time = now

            img        = frame.to_ndarray(format="bgr24")
            detections = self.detector.predict_track(img, imgsz=640)
            for d in detections:
                tid = d.get("track_id", -1)
                if tid >= 0:
                    self.unique_ids.add(tid)
            annotated  = self.detector.draw(img, detections)
            h, w       = annotated.shape[:2]
            cv2.putText(annotated, f"Drones in frame: {len(detections)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Total seen: {len(self.unique_ids)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="drone-live",
        video_processor_factory=_DroneProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 30, "max": 60}}, "audio": False},
    )


# ═══════════════════════════════════════════
# TAB 4 — Evaluation
# ═══════════════════════════════════════════

with tab_eval:
    st.header("Test Set Evaluation")
    st.write("Results from the held-out test set (111 images).")

    eval_root = ROOT / "runs" / "eval"
    prev_runs = sorted(eval_root.glob("*/metrics.json"), reverse=True) if eval_root.exists() else []

    if prev_runs:
        selected = st.selectbox(
            "Evaluation run",
            options=[str(p.parent) for p in prev_runs],
            format_func=lambda p: Path(p).name,
        )
        sel_dir = Path(selected)

        import json as _json
        metrics = _json.loads((sel_dir / "metrics.json").read_text())

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Precision",  f"{metrics['precision']:.4f}")
        mc2.metric("Recall",     f"{metrics['recall']:.4f}")
        mc3.metric("mAP50",      f"{metrics['map50']:.4f}")
        mc4.metric("mAP50-95",   f"{metrics['map50_95']:.4f}")

        summary_png = sel_dir / "summary.png"
        if summary_png.exists():
            st.image(str(summary_png), use_container_width=True)

        csv_path = sel_dir / "results_per_image.csv"
        if csv_path.exists():
            import pandas as _pd
            df = _pd.read_csv(csv_path)

            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("Total TP", int(df["TP"].sum()))
            dc2.metric("Total FP", int(df["FP"].sum()))
            dc3.metric("Total FN", int(df["FN"].sum()))
            dc4.metric("Avg inference", f"{df['inf_ms'].mean():.1f} ms")

            with st.expander("Per-image results"):
                st.dataframe(df, use_container_width=True)
                st.download_button("💾 Download CSV", df.to_csv(index=False).encode(),
                                   "eval_results.csv", "text/csv")

        vis_dir = sel_dir / "visualizations"
        vis_imgs = sorted(vis_dir.glob("*.jpg")) if vis_dir.exists() else []
        if vis_imgs:
            with st.expander(f"Sample visualizations ({len(vis_imgs)} saved)"):
                cols = st.columns(2)
                for i, img_path in enumerate(vis_imgs[:10]):
                    cols[i % 2].image(str(img_path), caption=img_path.stem, use_container_width=True)

    st.markdown("---")
    if st.button("▶ Run Evaluation on Test Set"):
        from src.evaluate import evaluate as run_eval
        with st.spinner("Evaluating … this may take a few minutes"):
            out_dir, metrics = run_eval(weights, 0.20, 0.45, 50)
        st.success("Done!")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Precision",  f"{metrics['precision']:.4f}")
        mc2.metric("Recall",     f"{metrics['recall']:.4f}")
        mc3.metric("mAP50",      f"{metrics['map50']:.4f}")
        mc4.metric("mAP50-95",   f"{metrics['map50_95']:.4f}")
        st.rerun()
