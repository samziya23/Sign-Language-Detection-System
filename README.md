# 🤟 SignSense AI — Real-time Sign Language Detection

A polished **Streamlit** web app that uses your webcam and a **CNN model** to recognise **American Sign Language (ASL)** hand gestures in real-time and translate them into text.

---

## ✨ Features

- 📹 **Live Webcam Feed** — real-time frame capture with mirroring
- ✂️ **Adjustable ROI Box** — focus the detection on just your hand
- 🎨 **Background Subtraction** — isolates hand from background noise
- 🧠 **CNN Inference** — per-frame gesture classification
- 📝 **Auto Text Builder** — stable predictions auto-append to a text panel
- 🗑 **Delete & Space** — special gestures for editing the output
- 🎛 **Sidebar Controls** — confidence threshold, frame-skip, ROI size
- 🟡 **Demo Mode** — fully interactive UI without a trained model

---

## 🔤 Supported Gestures

| Type | Gestures |
|------|----------|
| Letters | A – Z (26 signs) |
| Special | `space`, `del`, `nothing` |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit 1.35+ |
| Webcam Capture | OpenCV |
| Model | TensorFlow 2.x / Keras |
| Preprocessing | OpenCV, NumPy |
| Dataset | ASL Alphabet (Kaggle) |

---

## 📁 Project Structure

```
sign_language_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   ├── README.txt          # Instructions for placing model
│   └── sign_model.h5       # ← Place your trained model here
└── utils/
    ├── __init__.py
    └── detector.py         # ROI, BG subtraction, preprocessing utilities
```

---

## 🚀 Setup & Run

### 1. Clone / download

```bash
git clone <repo-url>
cd sign_language_app
```

### 2. Virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your trained model

```python
# In your training script:
model.save("sign_model.h5")
```

Copy `sign_model.h5` → `models/sign_model.h5`.

Expected model I/O:
- **Input**: `(None, 64, 64, 3)` — float32, values in `[0, 1]`
- **Output**: `(None, 29)` — softmax over 29 classes

### 5. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** — allow camera permissions when prompted.

---

## 🎮 Usage Guide

1. Click **▶ Start Webcam** — the camera feed appears
2. Place your hand **inside the purple ROI box**
3. Sign a letter and **hold it steady** for ~0.5 seconds
4. The detected letter auto-appends to the **Translated Text** panel
5. Use the `del` gesture to backspace, `space` to add a space
6. Click **⏹ Stop** to end the session
7. Adjust **Confidence Threshold** from the sidebar to reduce false positives

---

## 🧠 Model Architecture (recommended)

```
Input (64×64×3)
  → Conv2D(32) + BatchNorm + MaxPool
  → Conv2D(64) + BatchNorm + MaxPool
  → Conv2D(128) + BatchNorm + MaxPool
  → Flatten → Dense(512) → Dropout(0.5)
  → Dense(29, softmax)
```

---

---

## ⚠️ Notes

- Use `opencv-python-headless` on servers; use `opencv-python` locally for full GUI support.
- The webcam loop runs directly in Streamlit — for production deployments consider a WebRTC-based approach.
- For best accuracy: plain background, good lighting, hand centred in ROI box.

---

## 📜 Author-Samridhi
