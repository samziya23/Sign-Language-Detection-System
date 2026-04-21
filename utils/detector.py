"""
utils/detector.py
Hand detection, ROI extraction, and preprocessing utilities
for real-time sign language recognition.
"""
import cv2
import numpy as np

IMAGE_SIZE = 64

# ──────────────────────────────────────────────
# Background Subtraction
# ──────────────────────────────────────────────
class BackgroundSubtractor:
    """
    Accumulates a running average background model and subtracts it
    to isolate the hand region from the ROI.
    """

    def __init__(self, accumulate_weight: float = 0.5):
        self.weight = accumulate_weight
        self.background = None

    def accumulate(self, gray_frame: np.ndarray):
        if self.background is None:
            self.background = gray_frame.astype("float")
        else:
            cv2.accumulateWeighted(gray_frame, self.background, self.weight)

    def subtract(self, gray_frame: np.ndarray, threshold: int = 25):
        """
        Returns a binary mask highlighting the hand against the background.
        """
        diff = cv2.absdiff(self.background.astype("uint8"), gray_frame)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        return thresh

    def is_ready(self) -> bool:
        return self.background is not None


# ──────────────────────────────────────────────
# ROI Utilities
# ──────────────────────────────────────────────
def get_roi_box(frame_shape, size: int = 250):
    """
    Compute centred ROI bounding box coordinates.

    Returns:
        (x1, y1, x2, y2)
    """
    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    return x1, y1, x2, y2


def extract_roi(frame: np.ndarray, roi_box: tuple) -> np.ndarray:
    x1, y1, x2, y2 = roi_box
    return frame[y1:y2, x1:x2].copy()


def draw_roi_overlay(frame: np.ndarray, roi_box: tuple,
                     label: str = "", conf: float = 0.0) -> np.ndarray:
    """Draw ROI box and optional prediction label on the frame."""
    x1, y1, x2, y2 = roi_box
    color = (99, 102, 241)  # indigo
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    corner_len = 18
    corner_thickness = 3
    for (cx, cy, sx, sy) in [
        (x1, y1,  1,  1),
        (x2, y1, -1,  1),
        (x1, y2,  1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + sx * corner_len, cy), color, corner_thickness)
        cv2.line(frame, (cx, cy), (cx, cy + sy * corner_len), color, corner_thickness)

    # Label pill
    if label:
        text = f"{label}  {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        pill_x2 = x1 + tw + 20
        pill_y2 = y2 + th + 18
        cv2.rectangle(frame, (x1, y2 + 4), (pill_x2, pill_y2), (15, 20, 35), -1)
        cv2.rectangle(frame, (x1, y2 + 4), (pill_x2, pill_y2), color, 1)
        text_color = (99, 240, 132) if conf >= 70 else (255, 193, 7)
        cv2.putText(frame, text, (x1 + 10, pill_y2 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, text_color, 2)

    return frame


# ──────────────────────────────────────────────
# Preprocessing for model input
# ──────────────────────────────────────────────
def preprocess_for_model(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR ROI to a normalised float32 tensor of shape (1, 64, 64, 3).
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(roi_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalised = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalised, axis=0)


def preprocess_with_bg_subtraction(roi_bgr: np.ndarray,
                                    bg_mask: np.ndarray = None) -> np.ndarray:
    """
    Optionally apply background subtraction mask before resizing.
    Falls back to plain colour preprocessing if no mask provided.
    """
    if bg_mask is not None:
        # Apply mask to each channel
        mask_3ch = cv2.merge([bg_mask, bg_mask, bg_mask])
        roi_bgr = cv2.bitwise_and(roi_bgr, mask_3ch)
    return preprocess_for_model(roi_bgr)


# ──────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────
class PredictionStabiliser:
    """
    Smooths noisy per-frame predictions using a sliding window majority vote.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self._history = []

    def update(self, label: str, conf: float) -> tuple:
        self._history.append((label, conf))
        if len(self._history) > self.window:
            self._history.pop(0)

        # Majority vote
        labels = [l for l, _ in self._history]
        stable = max(set(labels), key=labels.count)
        avg_conf = np.mean([c for l, c in self._history if l == stable])
        return stable, float(avg_conf)

    def reset(self):
        self._history.clear()
