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

# Every emotion gets its own shade of one colour, rather than eight unrelated
# hues. Eight competing colours made the panel look like a pie chart; a single
# tonal ramp reads as one designed object, and the emotion is already named in
# words right next to its bar, so the colour does not have to identify it.
#
# The ramp runs light to dark down the list. Change these two endpoints and the
# whole app re-themes -- the bars, the big label, and the boxes drawn on faces.
RAMP_LIGHT = "#dbe7ff"   # palest shade, used for the first emotion
RAMP_DARK = "#3f70d6"    # deepest shade, used for the last

# Boxes drawn on faces get their own palette, and deliberately do NOT use the
# ramp above. Two different jobs:
#
#   The panel ramp sits on a known dark background and is a design choice.
#   The box has to work on top of an arbitrary photo, and it has to make a
#   change of emotion obvious at a glance -- when your expression shifts, the
#   box should visibly change colour.
#
# Pale tints fail at both: against a light background they vanish, and white
# label text on top of them is unreadable. These are saturated mid-tones,
# chosen dark enough that white text stays legible on every one of them (all
# clear 3:1 against white) while still being vivid and clearly distinct.
OVERLAY_COLORS = {
    "neutral":   "#64748b",   # slate
    "happiness": "#16a34a",   # green
    "surprise":  "#b8790a",   # amber (darkened from #ca8a04, which only
                              # reached 2.94:1 against the white label text)
    "sadness":   "#2563eb",   # blue
    "anger":     "#dc2626",   # red
    "disgust":   "#9333ea",   # purple
    "fear":      "#ea580c",   # orange
    "contempt":  "#0d9488",   # teal
}
OVERLAY_MUTED = (130, 130, 130)   # already BGR: faces you have not selected


def mix_hex(color_a: str, color_b: str, t: float) -> str:
    """Blend two hex colours. t=0 gives color_a, t=1 gives color_b.

    This is linear interpolation, the same idea as a gradient: for each of the
    red, green and blue channels, walk t of the way from one value to the other.
    """
    a = color_a.lstrip("#")
    b = color_b.lstrip("#")
    channels = []
    for i in (0, 2, 4):
        start, end = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        channels.append(round(start + (end - start) * t))
    return "#{:02x}{:02x}{:02x}".format(*channels)


# Spread the eight emotions evenly along the ramp. Used by the GUI bars and the
# on-video boxes alike; converted to BGR for OpenCV where needed.
EMOTION_COLORS = {
    emotion: mix_hex(RAMP_LIGHT, RAMP_DARK, i / (len(EMOTIONS) - 1))
    for i, emotion in enumerate(EMOTIONS)
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "emotion-ferplus-8.onnx"
)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#22c55e' -> (94, 197, 34). OpenCV wants Blue-Green-Red, not RGB."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# Roughly how often each emotion appears in the FER+ training labels. The data
# is badly unbalanced: well over half of it is neutral or happiness, while fear,
# disgust and contempt together are only a few percent.
#
# A model trained on that learns a habit as much as a skill -- guessing
# "neutral" is right often enough to be a good bet, so genuinely sad or fearful
# faces get pulled toward neutral. This is why the app kept answering neutral,
# happiness or surprise and almost never fear.
#
# These are approximate proportions, which is fine: only their relative sizes
# matter for the correction below.
CLASS_PRIORS = np.array(
    [0.36, 0.26, 0.13, 0.11, 0.08, 0.01, 0.03, 0.02],  # matches EMOTIONS order
    dtype=np.float32,
)

# The two rarest classes are so rare that dividing by their true share would
# make them fire constantly. Treating anything rarer than this as if it were
# this common keeps the correction sane.
PRIOR_FLOOR = 0.02


def rebalance(probabilities: np.ndarray, strength: float) -> np.ndarray:
    """Counteract the training data's imbalance. strength 0 = off, 1 = full.

    The model reports P(emotion | face) as learned from data where neutral was
    common and fear was rare, so its answers inherit that imbalance. Dividing
    each probability by how common that emotion was in training removes the
    built-in head start, which is the standard correction for an unbalanced
    classifier (dividing by the prior, then renormalising so it sums to 1).

    strength dials how much of that correction to apply. Full strength assumes
    every emotion is equally likely, which overshoots for a webcam -- people
    really are neutral most of the time. Partial strength keeps some of that
    sensible bias while still giving the rare emotions a chance to win.
    """
    if strength <= 0:
        return probabilities

    priors = np.maximum(CLASS_PRIORS, PRIOR_FLOOR)
    adjusted = probabilities / (priors ** strength)
    return adjusted / adjusted.sum()


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

    def scaled(self, factor: float) -> "Face":
        """A copy with the box resized, keeping the same emotion scores.

        Needed because a still photo is analysed at full resolution but shown
        shrunk to fit the window -- the boxes have to be moved to match.
        """
        return Face(
            (int(self.x * factor), int(self.y * factor),
             int(self.w * factor), int(self.h * factor)),
            self.probabilities,
        )

    def contains(self, px: int, py: int) -> bool:
        """Is the point (px, py) inside this face's box? Used for click-to-select."""
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class EmotionDetector:
    """Loads the models once, then analyses frames on demand."""

    # Faces are searched for on an image shrunk by this factor. See detect().
    DETECT_SCALE = 0.5

    def __init__(self, smoothing: int = 6, model_path: str = MODEL_PATH,
                 balance: float = 0.5):
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

        # How hard to correct for the training data's imbalance. See
        # rebalance(). Change it live with detector.balance = 0.0 ... 1.0
        self.balance = balance

    def reset(self) -> None:
        """Forget the smoothing history, e.g. when the camera restarts."""
        self.history.clear()

    def warm_up(self) -> None:
        """Run both models once on dummy data, to pay their start-up cost now.

        The first call into a Haar cascade and the first ONNX inference are far
        slower than every call after them -- together they measured about two
        seconds, versus a tenth of a second once warm. Without this, that delay
        lands on whatever the user does first and looks like the app hanging.
        Doing it at start-up, while the window is already on screen, hides it.
        """
        blank = np.zeros((240, 320), dtype=np.uint8)
        self.face_cascade.detectMultiScale(blank, 1.1, 6, minSize=(30, 30))
        self._predict_emotion(blank[:64, :64])
        self.history.clear()   # discard the dummy result

    def _predict_emotion(self, gray_face: np.ndarray) -> np.ndarray:
        """Run one cropped grayscale face through FER+ and return 8 probabilities."""
        # FER+ expects a 64x64 grayscale image shaped (batch, channel, H, W)
        # with plain 0-255 pixel values stored as float32 -- no normalisation.
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
        """Find every face in a BGR frame and classify its emotion.

        The list is sorted largest-face-first, so faces[0] is the person
        closest to the camera -- that is the one the live panel reports on.

        The arguments matter because live video and still photos want opposite
        trade-offs:

        max_faces  how many faces to bother classifying (each costs ~10 ms).
        smooth     average with recent frames. Right for video, wrong for a
                   still photo, where there are no "recent frames" and the
                   leftover history would corrupt the answer.
        min_face   ignore faces smaller than this, in pixels. Live video wants
                   this high (you are close to the camera, and it rejects
                   false positives). A group photo wants it lower.
        scale      shrink the image by this much before searching. Lower is
                   faster but slightly less precise; None uses DETECT_SCALE.
        """
        scale = self.DETECT_SCALE if scale is None else scale
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Contrast-equalise: helps the detector a lot in a dim room.
        equalised = cv2.equalizeHist(gray)

        # Searching for faces is the slowest step, and its cost grows with the
        # number of pixels. Halving the image makes it ~25% faster and moves
        # the detected boxes by only a pixel or two, which nobody can see.
        if scale == 1.0:
            small = equalised
        else:
            small = cv2.resize(equalised, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        min_side = max(24, int(min_face * scale))

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
            x, y, w, h = (int(v / scale) for v in (x, y, w, h))

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

        # Smooth only the main (largest) face -- that is what the live panel
        # shows. Skipped entirely for still images: a photo is a single moment,
        # so averaging it with whatever was on camera earlier would be wrong.
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
    """Read a picture from disk, returning None if it is not a usable image.

    This deliberately does not use cv2.imread(). On Windows, imread() fails and
    silently returns None whenever the path contains non-English characters --
    an accented name, or a OneDrive folder in another language. Reading the
    bytes with NumPy first and decoding them in memory sidesteps that entirely.
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

    # Some phone photos and PNGs decode with an alpha channel or as grayscale;
    # everything downstream assumes plain 3-channel BGR.
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
    """Draw a coloured box and a label on each face. Modifies frame in place.

    highlight  index of the face to emphasise -- the one the panel is
               reporting on. Pass -1 for none.
    numbered   prefix each label with "1", "2"... so the boxes can be matched
               against the face picker. Used for still images.
    """
    for i, face in enumerate(faces):
        x, y, w, h = face.x, face.y, face.w, face.h
        chosen = (i == highlight)
        thickness = 2 if chosen else 1

        # The selected face is drawn in its emotion's colour, so the box
        # visibly changes the moment the expression does. Faces the panel is
        # not reporting on stay grey.
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

        # Filled strip behind the text so it stays readable on any background.
        top = max(0, y - th - 12)
        cv2.rectangle(frame, (x, top), (x + tw + 12, y), color, -1)
        cv2.putText(frame, text, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    scale_, (255, 255, 255), 2, cv2.LINE_AA)

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
