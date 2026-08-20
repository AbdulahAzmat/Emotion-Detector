"""
Face detection + emotion classification. No GUI code in here, so it can be
tested on its own:  python emotion_detector.py

For each frame: grayscale it, find faces with a Haar cascade, crop each face to
64x64 and run it through the FER+ model, then softmax the 8 scores.
"""

from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

# Don't reorder this. Index 0 really is "neutral" in the model's output.
EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# Panel colours: one shade per emotion, blended between these two endpoints.
# Change these and the whole panel re-themes.
RAMP_LIGHT = "#dbe7ff"
RAMP_DARK = "#3f70d6"

# Boxes drawn on faces use their own palette instead of the ramp. A photo can be
# any colour, and the pale end of the ramp vanished on light backgrounds and
# made the white label text unreadable. These are all >= 3:1 against white.
OVERLAY_COLORS = {
    "neutral":   "#64748b",
    "happiness": "#16a34a",
    "surprise":  "#b8790a",
    "sadness":   "#2563eb",
    "anger":     "#dc2626",
    "disgust":   "#9333ea",
    "fear":      "#ea580c",
    "contempt":  "#0d9488",
}
OVERLAY_MUTED = (130, 130, 130)   # BGR already: faces you haven't selected


def mix_hex(color_a: str, color_b: str, t: float) -> str:
    """Blend two hex colours. t=0 gives color_a, t=1 gives color_b."""
    a = color_a.lstrip("#")
    b = color_b.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        start, end = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        channels.append(round(start + (end - start) * t))
    return "#{:02x}{:02x}{:02x}".format(*channels)


EMOTION_COLORS = {
    emotion: mix_hex(RAMP_LIGHT, RAMP_DARK, i / (len(EMOTIONS) - 1))
    for i, emotion in enumerate(EMOTIONS)
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "emotion-ferplus-8.onnx"
)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """OpenCV wants Blue-Green-Red, not RGB."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# Roughly how often each emotion shows up in the FER+ training labels. It's very
# unbalanced -- mostly neutral and happy -- which is why the model rarely says
# fear. Only the relative sizes matter.
CLASS_PRIORS = np.array(
    [0.36, 0.26, 0.13, 0.11, 0.08, 0.01, 0.03, 0.02],   # same order as EMOTIONS
    dtype=np.float32,
)

# Without a floor, the two rarest classes get such a huge boost they fire
# constantly.
PRIOR_FLOOR = 0.02


def rebalance(probabilities: np.ndarray, strength: float) -> np.ndarray:
    """Undo the training data's imbalance. strength 0 = off, 1 = full.

    Divide each probability by how common that emotion was in training, then
    renormalise. Full strength assumes all 8 are equally likely, which is too
    much for a webcam since people really are neutral most of the time.
    """
    if strength <= 0:
        return probabilities

    priors = np.maximum(CLASS_PRIORS, PRIOR_FLOOR)
    adjusted = probabilities / (priors ** strength)
    return adjusted / adjusted.sum()


def softmax(scores: np.ndarray) -> np.ndarray:
    """Raw scores -> probabilities summing to 1. The max subtraction stops
    np.exp() overflowing."""
    exp = np.exp(scores - np.max(scores))
    return exp / np.sum(exp)


class Face:
    """One detected face and its emotion scores."""

    def __init__(self, box: tuple[int, int, int, int], probabilities: np.ndarray):
        self.x, self.y, self.w, self.h = box
        self.probabilities = probabilities
        self.index = int(np.argmax(probabilities))

    @property
    def label(self) -> str:
        return EMOTIONS[self.index]

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.index])

    @property
    def color(self) -> str:
        return EMOTION_COLORS[self.label]

    def scaled(self, factor: float) -> "Face":
        """Copy with the box resized. A photo is analysed full size but shown
        shrunk to fit, so the boxes have to move to match."""
        return Face(
            (int(self.x * factor), int(self.y * factor),
             int(self.w * factor), int(self.h * factor)),
            self.probabilities,
        )

    def contains(self, px: int, py: int) -> bool:
        """For click-to-select."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class EmotionDetector:
    """Loads both models once, then analyses frames on demand."""

    DETECT_SCALE = 0.5

    def __init__(self, smoothing: int = 6, model_path: str = MODEL_PATH,
                 balance: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Emotion model not found at:\n  " + model_path + "\n\n"
                "Run this first:  python download_model.py"
            )

        # OpenCV 5.0 removed CascadeClassifier. Without this you just get
        # "module 'cv2' has no attribute 'CascadeClassifier'", which gives no
        # hint that the version is the problem.
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This app needs OpenCV 4.x, but OpenCV {} is installed, and "
                "version 5 removed the face detector this app uses.\n\n"
                "Fix it with:\n"
                '    pip install "opencv-python>=4.8,<5"'.format(cv2.__version__)
            )

        # These XML files ship inside OpenCV, so there's nothing to download.
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load the face cascade: " + cascade_file)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Per-frame predictions jitter a lot, so we average the last few.
        self.history: deque[np.ndarray] = deque(maxlen=max(1, smoothing))
        self.balance = balance

    def reset(self) -> None:
        self.history.clear()

    def warm_up(self) -> None:
        """Run both models once on dummy data. The first Haar call and first
        ONNX inference take about 2s between them versus ~0.1s after, and
        without this that delay lands on whatever the user does first."""
        blank = np.zeros((240, 320), dtype=np.uint8)
        self.face_cascade.detectMultiScale(blank, 1.1, 6, minSize=(30, 30))
        self._predict_emotion(blank[:64, :64])
        self.history.clear()

    def _predict_emotion(self, gray_face: np.ndarray) -> np.ndarray:
        # FER+ wants 64x64 grayscale as (batch, channel, H, W), with plain
        # 0-255 values as float32 -- no normalisation.
        face = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = face.astype(np.float32).reshape(1, 1, 64, 64)

        scores = self.session.run(None, {self.input_name: tensor})[0][0]
        return rebalance(softmax(scores), self.balance)

    def detect(
        self,
        frame: np.ndarray,
        max_faces: int = 4,
        smooth: bool = True,
        min_face: int = 80,
        scale: float | None = None,
    ) -> list[Face]:
        """Find and classify every face. Sorted largest first, so faces[0] is
        whoever is nearest the camera.

        Live video and still photos want different settings, hence the args:
        smooth should be off for a photo (there are no recent frames to average
        with), and min_face lower, since group photos have smaller faces.
        """
        scale = self.DETECT_SCALE if scale is None else scale
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Equalising helps a lot in a dim room.
        equalised = cv2.equalizeHist(gray)

        # Searching a half-size copy is ~25% faster and moves the boxes by a
        # pixel or two, which nobody notices.
        if scale == 1.0:
            small = equalised
        else:
            small = cv2.resize(equalised, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        min_side = max(24, int(min_face * scale))

        boxes = self.face_cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=6,    # higher = fewer false positives
            minSize=(min_side, min_side),
        )

        # Biggest first, so hitting max_faces keeps the nearest people.
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]

        faces: list[Face] = []
        for (x, y, w, h) in boxes:
            x, y, w, h = (int(v / scale) for v in (x, y, w, h))

            # Clamp, or rounding can push a box off the edge and the crop
            # comes back empty.
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w <= 0 or h <= 0:
                continue

            # Classify from the original gray, not the equalised copy -- the
            # model was trained on normal photos.
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            faces.append(Face((x, y, w, h), self._predict_emotion(crop)))

        # Only the main face gets smoothed, and only for video.
        if smooth:
            if faces:
                main = faces[0]
                self.history.append(main.probabilities)
                smoothed = np.mean(self.history, axis=0)
                faces[0] = Face((main.x, main.y, main.w, main.h), smoothed)
            else:
                self.history.clear()

        return faces


def load_image(path: str) -> np.ndarray | None:
    """Load a picture, or None if it isn't one.

    Not cv2.imread(): on Windows that silently returns None for any path with
    non-English characters in it.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Everything downstream assumes 3-channel BGR.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def draw_overlay(
    frame: np.ndarray,
    faces: list[Face],
    highlight: int = 0,
    numbered: bool = False,
) -> np.ndarray:
    """Draw a box and label on each face, in place.

    highlight is the face the panel is reporting on (-1 for none). numbered
    prefixes "1", "2"... to match the face picker, for still images.
    """
    for i, face in enumerate(faces):
        x, y, w, h = face.x, face.y, face.w, face.h
        chosen = (i == highlight)
        thickness = 2 if chosen else 1

        color = hex_to_bgr(OVERLAY_COLORS[face.label]) if chosen else OVERLAY_MUTED

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        if numbered:
            text = "{}".format(i + 1)
            if chosen:
                text += "  {}  {:.0f}%".format(face.label, face.confidence * 100)
        else:
            text = "{}  {:.0f}%".format(face.label, face.confidence * 100)

        scale_ = 0.6 if chosen else 0.5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_, 2)

        # Filled strip so the text stays readable on any background.
        top = max(0, y - th - 12)
        cv2.rectangle(frame, (x, top), (x + tw + 12, y), color, -1)
        cv2.putText(frame, text, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    scale_, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    # Quick test without the GUI. Press Q to quit.
    print("Loading models...")
    detector = EmotionDetector()

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam.")

    print("Running. Press Q in the video window to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Emotion Detector (test)", draw_overlay(frame, detector.detect(frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
