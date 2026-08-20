"""
emotion_detector.py
===================
The "brain" of the app. It knows nothing about windows or buttons -- it only
takes a camera frame (a NumPy image) and answers: where are the faces, and
what emotion is each one showing?

Keeping this separate from the GUI is a habit worth learning: you can test and
reuse this file on its own (see the __main__ block at the bottom).

The pipeline for every frame:
    1. Convert the frame to grayscale (colour adds nothing here and is slower).
    2. Find faces with a Haar cascade -- a classic, fast, CPU-friendly detector
       that ships inside OpenCV, so there is nothing extra to download.
    3. Crop each face, resize it to 64x64, and feed it to the FER+ neural
       network (an ONNX file) which outputs 8 numbers -- one score per emotion.
    4. Turn those raw scores into percentages with softmax.
    5. Smooth the percentages over time so the labels don't flicker wildly.
"""

from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

# The FER+ model was trained to output these 8 emotions, in exactly this order.
# Do not reorder this list -- index 0 really is "neutral" in the model's output.
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

# A colour for each emotion, used by both the GUI bars and the on-video boxes.
# Stored as hex for the GUI; converted to BGR for OpenCV when needed.
EMOTION_COLORS = {
    "neutral":   "#94a3b8",
    "happiness": "#22c55e",
    "surprise":  "#eab308",
    "sadness":   "#3b82f6",
    "anger":     "#ef4444",
    "disgust":   "#a855f7",
    "fear":      "#f97316",
    "contempt":  "#14b8a6",
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "emotion-ferplus-8.onnx"
)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#22c55e' -> (94, 197, 34). OpenCV wants Blue-Green-Red, not RGB."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn raw model scores (any range) into probabilities that sum to 1.0.

    Subtracting the max first is a standard numerical-stability trick: it stops
    np.exp() from overflowing on large scores, without changing the result.
    """
    exp = np.exp(scores - np.max(scores))
    return exp / np.sum(exp)


class Face:
    """One detected face plus its emotion reading for the current frame."""

    def __init__(self, box: tuple[int, int, int, int], probabilities: np.ndarray):
        self.x, self.y, self.w, self.h = box
        self.probabilities = probabilities          # 8 floats, sums to 1.0
        self.index = int(np.argmax(probabilities))  # winning emotion's index

    @property
    def label(self) -> str:
        return EMOTIONS[self.index]

    @property
    def confidence(self) -> float:
        """How sure the model is about the winning emotion, as 0.0 - 1.0."""
        return float(self.probabilities[self.index])

    @property
    def color(self) -> str:
        return EMOTION_COLORS[self.label]


class EmotionDetector:
    """Loads the models once, then analyses frames on demand."""

    # Faces are searched for on an image shrunk by this factor. See detect().
    DETECT_SCALE = 0.5

    def __init__(self, smoothing: int = 6, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Emotion model not found at:\n  " + model_path + "\n\n"
                "Run this first:  python download_model.py"
            )

        # --- Face detector -------------------------------------------------
        # OpenCV 5.0 dropped CascadeClassifier entirely. Without this check you
        # get a bare "module 'cv2' has no attribute 'CascadeClassifier'", which
        # gives you no clue that the real problem is the installed version.
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This app needs OpenCV 4.x, but OpenCV {} is installed, and "
                "version 5 removed the face detector this app uses.\n\n"
                "Fix it with:\n"
                '    pip install "opencv-python>=4.8,<5"'.format(cv2.__version__)
            )

        # cv2.data.haarcascades points at the XML files bundled with OpenCV,
        # so this works on any machine without downloading anything.
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load the face cascade: " + cascade_file)

        # --- Emotion model -------------------------------------------------
        # ONNX Runtime runs the neural network. CPUExecutionProvider = plain
        # CPU, which is plenty fast for one small 64x64 image at a time.
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # --- Temporal smoothing --------------------------------------------
        # Raw per-frame predictions jitter (happiness -> neutral -> happiness).
        # Averaging the last few frames makes the readout feel stable, the way
        # the apps you have seen on Instagram do.
        self.history: deque[np.ndarray] = deque(maxlen=max(1, smoothing))

    def reset(self) -> None:
        """Forget the smoothing history, e.g. when the camera restarts."""
        self.history.clear()

    def _predict_emotion(self, gray_face: np.ndarray) -> np.ndarray:
        """Run one cropped grayscale face through FER+ and return 8 probabilities."""
        # FER+ expects a 64x64 grayscale image shaped (batch, channel, H, W)
        # with plain 0-255 pixel values stored as float32 -- no normalisation.
        face = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
        tensor = face.astype(np.float32).reshape(1, 1, 64, 64)

        scores = self.session.run(None, {self.input_name: tensor})[0][0]
        return softmax(scores)

    def detect(self, frame: np.ndarray, max_faces: int = 4) -> list[Face]:
        """Find every face in a BGR frame and classify its emotion.

        The list is sorted largest-face-first, so faces[0] is the person
        closest to the camera -- that is the one the side panel reports on.
        """
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Contrast-equalise: helps the detector a lot in a dim room.
        equalised = cv2.equalizeHist(gray)

        # Searching for faces is the slowest step, and its cost grows with the
        # number of pixels. Halving the image makes it ~25% faster and moves
        # the detected boxes by only a pixel or two, which nobody can see.
        small = cv2.resize(equalised, None, fx=self.DETECT_SCALE,
                           fy=self.DETECT_SCALE, interpolation=cv2.INTER_AREA)
        min_side = max(30, int(80 * self.DETECT_SCALE))

        boxes = self.face_cascade.detectMultiScale(
            small,
            scaleFactor=1.1,   # shrink the image 10% per pass while searching
            minNeighbors=6,    # higher = fewer false positives, fewer detections
            minSize=(min_side, min_side),
        )

        # Biggest faces first, so that if we hit the max_faces cap we keep the
        # people closest to the camera rather than whoever happened to be first.
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_faces]

        faces: list[Face] = []
        for (x, y, w, h) in boxes:
            # Undo the shrink so the box lines up with the full-size frame.
            x, y, w, h = (int(v / self.DETECT_SCALE) for v in (x, y, w, h))

            # Clamp to the frame: rounding can push a box a pixel past the edge,
            # and slicing outside the image would give us an empty crop.
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w <= 0 or h <= 0:
                continue

            # Classify from the *original* grayscale, not the equalised copy:
            # the model was trained on normal photos, not contrast-stretched ones.
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            faces.append(Face((x, y, w, h), self._predict_emotion(crop)))

        # Smooth only the main (largest) face -- that is what the panel shows.
        if faces:
            main = faces[0]
            self.history.append(main.probabilities)
            smoothed = np.mean(self.history, axis=0)
            faces[0] = Face((main.x, main.y, main.w, main.h), smoothed)
        else:
            self.history.clear()

        return faces


def draw_overlay(frame: np.ndarray, faces: list[Face]) -> np.ndarray:
    """Draw a coloured box and a label above each face. Modifies frame in place."""
    for i, face in enumerate(faces):
        x, y, w, h = face.x, face.y, face.w, face.h
        color = hex_to_bgr(face.color)
        thickness = 3 if i == 0 else 2  # the main face gets a bolder box

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        text = "{}  {:.0f}%".format(face.label, face.confidence * 100)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Filled strip behind the text so it stays readable on any background.
        top = max(0, y - th - 12)
        cv2.rectangle(frame, (x, top), (x + tw + 12, y), color, -1)
        cv2.putText(frame, text, (x + 6, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


if __name__ == "__main__":
    # A tiny self-test you can run on its own:  python emotion_detector.py
    # It opens the webcam in a bare OpenCV window. Press Q to quit.
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
        frame = cv2.flip(frame, 1)  # mirror, so moving right moves right
        cv2.imshow("Emotion Detector (test)", draw_overlay(frame, detector.detect(frame)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
