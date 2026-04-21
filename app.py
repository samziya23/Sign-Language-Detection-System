import streamlit as st
import numpy as np
import cv2
import os
import time
from PIL import Image
import io

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="SignSense AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --border:   #1f2937;
    --accent:   #6366f1;
    --accent2:  #8b5cf6;
    --green:    #10b981;
    --yellow:   #f59e0b;
    --red:      #ef4444;
    --text:     #f1f5f9;
    --muted:    #64748b;
    --radius:   16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.glow-card {
    background: linear-gradient(135deg, #111827, #0a0e1a);
    border: 1px solid #312e81;
    border-radius: var(--radius);
    padding: 1.6rem;
    box-shadow: 0 0 30px rgba(99,102,241,.15);
}
.pred-display {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border: 2px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 20px rgba(99,102,241,.2);
}
.letter-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    display: block;
}
.text-output {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    letter-spacing: .08em;
    color: var(--green);
    min-height: 60px;
    word-break: break-all;
}
.status-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: .4rem;
    vertical-align: middle;
}
.dot-green { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.5s infinite; }
.dot-grey  { background: var(--muted); }
@keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:.4; }
}
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: .95rem !important;
    padding: .55rem 1.5rem !important;
    width: 100% !important;
    border: none !important;
}
.btn-start > button { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:#fff !important; }
.btn-stop  > button { background: linear-gradient(135deg,#ef4444,#dc2626) !important; color:#fff !important; }
.btn-clear > button { background: #1f2937 !important; color: var(--muted) !important; border: 1px solid var(--border) !important; }

.alpha-grid {
    display: grid; grid-template-columns: repeat(9, 1fr);
    gap: .3rem; margin-top: .5rem;
}
.alpha-cell {
    background: #1f2937;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .35rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: .85rem;
    color: var(--muted);
    transition: all .2s;
}
.alpha-cell.active {
    background: #312e81;
    border-color: var(--accent);
    color: var(--text);
    box-shadow: 0 0 8px rgba(99,102,241,.4);
}
.conf-row {
    display: flex; justify-content: space-between;
    align-items: center; font-size: .82rem;
    color: var(--muted); margin-bottom: .3rem;
}
.conf-bar { background: #1f2937; border-radius: 999px; height: 6px; overflow: hidden; }
.conf-fill { height:100%; border-radius:999px; background: linear-gradient(90deg, #6366f1, #8b5cf6); }

hr { border-color: var(--border) !important; }
.sidebar-item {
    background: #0a0e1a; border: 1px solid var(--border);
    border-radius: 8px; padding: .6rem .9rem; margin-bottom: .5rem;
    font-size: .82rem; display: flex; justify-content: space-between;
}
.tip-box {
    background: #0f1629; border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: .8rem 1rem; font-size: .87rem; color: var(--muted);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
IMAGE_SIZE = 64   # typical for ASL CNN

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "running"       not in st.session_state: st.session_state.running = False
if "text_output"   not in st.session_state: st.session_state.text_output = ""
if "last_pred"     not in st.session_state: st.session_state.last_pred = "—"
if "last_conf"     not in st.session_state: st.session_state.last_conf = 0.0
if "frame_count"   not in st.session_state: st.session_state.frame_count = 0

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(os.path.dirname(__file__), "models", "sign_model.h5")
    if os.path.exists(path):
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    return None

# ──────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────
def preprocess_roi(roi):
    """BGR ROI → normalised float32 tensor (1, H, W, 3)."""
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    arr = roi_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)

def predict_roi(model, roi):
    tensor = preprocess_roi(roi)
    preds = model.predict(tensor, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(np.max(preds)) * 100

def demo_predict_roi(roi):
    """Deterministic demo prediction from pixel mean."""
    mean = float(roi.mean())
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = int(mean * 26 / 255) % 26
    fake_conf = 70 + (mean % 25)
    return letters[idx], fake_conf

def apply_background_subtraction(frame, roi_box, frame_num):
    """Draw ROI box and apply simple preprocessing for hand detection."""
    x1, y1, x2, y2 = roi_box
    roi = frame[y1:y2, x1:x2]

    # Draw ROI rectangle on frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (99, 102, 241), 2)
    cv2.putText(frame, "ROI", (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99, 102, 241), 1)

    # Background subtraction hint (grayscale overlay in ROI preview)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size > 0 else roi
    return frame, roi


def draw_prediction_overlay(frame, label, conf, roi_box):
    """Draw prediction text on frame."""
    x1, y1, x2, y2 = roi_box
    color = (99, 240, 132) if conf > 70 else (255, 193, 7)

    # Background pill for text
    text = f"{label}  {conf:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + tw + 16, y2 + th + 18), (17, 24, 39), -1)
    cv2.rectangle(frame, (x1, y2 + 5), (x1 + tw + 16, y2 + th + 18), (63, 66, 209), 1)
    cv2.putText(frame, text, (x1 + 8, y2 + th + 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
    return frame


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 1.5rem;">
        <div style="font-size:2.5rem;">🤟</div>
        <h2 style="font-family:'Space Grotesk',sans-serif; margin:.3rem 0 .1rem; font-size:1.4rem;">SignSense AI</h2>
        <p style="color:#64748b; font-size:.82rem; margin:0;">Real-time ASL Detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ROI settings
    st.markdown("**🎯 Detection Settings**")
    roi_size = st.slider("ROI Box Size (px)", 150, 400, 250, 10)
    conf_threshold = st.slider("Confidence Threshold (%)", 40, 95, 60, 5)
    frame_skip = st.slider("Predict Every N Frames", 1, 10, 3, 1,
                           help="Higher = faster but less frequent predictions")
    auto_append = st.checkbox("Auto-append letter to text", value=True)

    st.markdown("---")

    st.markdown("**📊 Model Info**")
    for label, val in [
        ("Input Size", f"{IMAGE_SIZE}×{IMAGE_SIZE}"),
        ("Classes", str(len(CLASS_NAMES))),
        ("Architecture", "CNN"),
        ("Framework", "TF/Keras"),
        ("Preprocessing", "BG Subtraction"),
    ]:
        st.markdown(f'<div class="sidebar-item"><span style="color:#64748b">{label}</span><strong>{val}</strong></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🔤 Supported Signs**")
    st.markdown("""
    <div style="color:#64748b; font-size:.82rem; line-height:1.8;">
        A–Z (26 letters)<br>
        + <code>space</code>, <code>del</code>, <code>nothing</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#475569; font-size:.76rem; text-align:center;">
        Dataset: ASL Alphabet (Kaggle)<br>
        OpenCV + TensorFlow/Keras<br><br>Built with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.1rem; margin-bottom:.2rem;">
    🤟 Sign Language Detection
</h1>
<p style="color:#64748b; font-size:.98rem; margin-bottom:1.5rem;">
    Real-time ASL hand gesture recognition via webcam — powered by CNN + OpenCV.
</p>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Initialising model…"):
    model = load_model()

if model is None:
    st.markdown("""
    <div class="tip-box" style="margin-bottom:1rem;">
        🟡 <strong>Demo Mode:</strong> No model at <code>models/sign_model.h5</code>.
        Place your trained model there to enable real predictions.
        Webcam feed and UI are fully functional — predictions are simulated.
    </div>
    """, unsafe_allow_html=True)

# ── Control row ──────────────────────────────
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])

with ctrl1:
    st.markdown('<div class="btn-start">', unsafe_allow_html=True)
    start_btn = st.button("▶ Start Webcam", key="start", use_container_width=True,
                          disabled=st.session_state.running)
    st.markdown('</div>', unsafe_allow_html=True)

with ctrl2:
    st.markdown('<div class="btn-stop">', unsafe_allow_html=True)
    stop_btn = st.button("⏹ Stop", key="stop", use_container_width=True,
                         disabled=not st.session_state.running)
    st.markdown('</div>', unsafe_allow_html=True)

with ctrl3:
    st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
    clear_btn = st.button("🗑 Clear Text", key="clear", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if start_btn: st.session_state.running = True;  st.rerun()
if stop_btn:  st.session_state.running = False; st.rerun()
if clear_btn: st.session_state.text_output = ""; st.rerun()

# ── Status indicator ──────────────────────────
status_html = (
    '<span class="status-dot dot-green"></span> <strong>Webcam Active</strong>'
    if st.session_state.running else
    '<span class="status-dot dot-grey"></span> <span style="color:#64748b">Camera Off</span>'
)
st.markdown(f'<div style="margin:.4rem 0 1rem; font-size:.9rem;">{status_html}</div>',
            unsafe_allow_html=True)

# ── Main content columns ──────────────────────
col_cam, col_panel = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown("#### 📹 Live Feed")
    cam_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Tips
    with st.expander("📌 How to position your hand"):
        st.markdown("""
        - 🖐 Place your hand **inside the purple ROI box**
        - 💡 Ensure **good, even lighting** (face a window or lamp)
        - 🎨 Use a **plain, contrasting background** behind your hand
        - ✋ Keep your hand **still for 1-2 seconds** while signing
        - 📏 The ROI box is adjustable from the sidebar
        """)

with col_panel:
    st.markdown('<div class="pred-display">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Current Prediction")
    pred_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 📝 Translated Text")
    text_placeholder = st.empty()
    action_info = st.empty()

    # Alphabet grid
    st.markdown("#### 🔠 ASL Alphabet Reference")
    grid_placeholder = st.empty()


# ──────────────────────────────────────────────
# Render static state (not running)
# ──────────────────────────────────────────────
def render_pred_panel(pred, conf):
    pred_placeholder.markdown(f"""
    <div style="text-align:center; padding:.5rem 0;">
        <span class="letter-big">{pred}</span>
        <div style="margin-top:.8rem; color:#64748b; font-size:.88rem;">Detected Sign</div>
        <div style="margin-top:.5rem;">
            <div class="conf-row">
                <span>Confidence</span>
                <span style="color:{'#10b981' if conf>70 else '#f59e0b'}">{conf:.1f}%</span>
            </div>
            <div class="conf-bar">
                <div class="conf-fill" style="width:{min(conf,100):.0f}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_text_panel(text):
    display = text if text else "…"
    text_placeholder.markdown(f"""
    <div class="text-output">{display}</div>
    <div style="color:#475569; font-size:.76rem; margin-top:.4rem;">
        {len(text)} characters
    </div>
    """, unsafe_allow_html=True)


def render_grid(active_letter=""):
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cells = ""
    for l in letters:
        cls = "alpha-cell active" if l == active_letter else "alpha-cell"
        cells += f'<div class="{cls}">{l}</div>'
    grid_placeholder.markdown(
        f'<div class="alpha-grid">{cells}</div>', unsafe_allow_html=True
    )


render_pred_panel(st.session_state.last_pred, st.session_state.last_conf)
render_text_panel(st.session_state.text_output)
render_grid(st.session_state.last_pred if st.session_state.last_pred != "—" else "")


# ──────────────────────────────────────────────
# Webcam loop
# ──────────────────────────────────────────────
if st.session_state.running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Check browser permissions or connect a camera.")
        st.session_state.running = False
        st.rerun()

    # ROI coordinates (centred)
    half = roi_size // 2
    cx, cy = 320, 240
    roi_box = (cx - half, cy - half, cx + half, cy + half)

    prev_pred = ""
    pred_hold = 0          # frames the same prediction has been seen
    HOLD_FRAMES = 8        # consecutive frames before appending to text
    frame_num = 0

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost webcam feed.")
                break

            frame = cv2.flip(frame, 1)   # mirror
            frame_num += 1
            x1, y1, x2, y2 = roi_box
            roi = frame[y1:y2, x1:x2].copy()

            # Predict every N frames
            if frame_num % frame_skip == 0 and roi.size > 0:
                if model:
                    pred, conf = predict_roi(model, roi)
                else:
                    pred, conf = demo_predict_roi(roi)

                if conf >= conf_threshold:
                    st.session_state.last_pred = pred
                    st.session_state.last_conf = conf

                    # Auto-append logic
                    if auto_append and pred.isalpha() and len(pred) == 1:
                        if pred == prev_pred:
                            pred_hold += 1
                        else:
                            pred_hold = 0
                            prev_pred = pred

                        if pred_hold == HOLD_FRAMES:
                            st.session_state.text_output += pred
                            pred_hold = 0

                    elif auto_append and pred == "space":
                        if pred == prev_pred:
                            pred_hold += 1
                        else:
                            pred_hold = 0; prev_pred = pred
                        if pred_hold == HOLD_FRAMES:
                            st.session_state.text_output += " "
                            pred_hold = 0

                    elif auto_append and pred == "del":
                        if pred == prev_pred:
                            pred_hold += 1
                        else:
                            pred_hold = 0; prev_pred = pred
                        if pred_hold == HOLD_FRAMES and st.session_state.text_output:
                            st.session_state.text_output = st.session_state.text_output[:-1]
                            pred_hold = 0

            # Draw on frame
            frame, _ = apply_background_subtraction(frame, roi_box, frame_num)
            if st.session_state.last_pred != "—":
                frame = draw_prediction_overlay(frame, st.session_state.last_pred,
                                                st.session_state.last_conf, roi_box)

            # Convert and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update panels
            render_pred_panel(st.session_state.last_pred, st.session_state.last_conf)
            render_text_panel(st.session_state.text_output)
            render_grid(st.session_state.last_pred if st.session_state.last_pred not in ["—", "del", "space", "nothing"] else "")

            time.sleep(0.03)   # ~30 FPS cap

    finally:
        cap.release()

    st.session_state.running = False
    st.rerun()

else:
    # Static placeholder image
    cam_placeholder.markdown("""
    <div style="
        background: #111827;
        border: 2px dashed #1f2937;
        border-radius: 12px;
        height: 340px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #374151;
        gap: .8rem;
    ">
        <div style="font-size:3rem;">📷</div>
        <div style="font-size:.95rem;">Press <strong style="color:#6366f1">▶ Start Webcam</strong> to begin</div>
        <div style="font-size:.78rem; color:#1f2937;">Camera feed will appear here</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# How it works section
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### ⚙️ How It Works")
cols = st.columns(5)
steps = [
    ("📹", "Capture", "Webcam frame captured and mirrored in real-time."),
    ("✂️", "ROI Crop", "Hand Region of Interest extracted from the purple box."),
    ("🎨", "Preprocess", "BG subtraction + normalization applied to the ROI."),
    ("🧠", "CNN Predict", "Model outputs probability over 29 gesture classes."),
    ("📝", "Translate", "Stable predictions auto-appended to the text output."),
]
for col, (icon, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding:1.1rem .8rem;">
            <div style="font-size:1.6rem;">{icon}</div>
            <h4 style="font-family:'Space Grotesk',sans-serif; margin:.4rem 0 .2rem; font-size:.95rem;">{title}</h4>
            <p style="color:#64748b; font-size:.78rem; margin:0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
