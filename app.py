"""
app.py
======
The graphical app. Run it with:  python app.py

How it is put together
----------------------
There are two things happening at once:

  * The CAMERA THREAD grabs frames from the webcam and runs them through the
    EmotionDetector. This is the slow part (tens of milliseconds per frame).
  * The MAIN THREAD runs the GUI and redraws the window.

They must be separate. If you grabbed camera frames inside the GUI thread, the
window would freeze solid between frames -- no button clicks, no dragging. So
the camera thread just keeps a "latest result" in a shared box, and the GUI
peeks into that box about 30 times a second and paints whatever it finds.

`threading.Lock` guards that shared box so the two threads never read and write
it at the same instant.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from emotion_detector import EMOTIONS, EMOTION_COLORS, EmotionDetector, draw_overlay

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_DARK = "#0f1117"      # window background
BG_CARD = "#171a23"      # panels sitting on the background
BG_INSET = "#1f2430"     # inputs / bar troughs
TEXT_MAIN = "#e8ecf4"
TEXT_DIM = "#8b93a7"
ACCENT = "#5b8cff"

VIDEO_W, VIDEO_H = 640, 480

PLACEHOLDER_TEXT = "Camera is off\n\nPress “Start camera” below"


class CameraThread(threading.Thread):
    """Reads the webcam and runs emotion detection, off the GUI thread."""

    def __init__(self, detector: EmotionDetector, camera_index: int = 0):
        super().__init__(daemon=True)  # daemon = dies automatically with the app
        self.detector = detector
        self.camera_index = camera_index

        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        # The shared box the GUI reads from.
        self._frame = None      # the annotated BGR image
        self._faces = []        # list[Face]
        self._fps = 0.0
        self.error: str | None = None

    def latest(self):
        """Thread-safe snapshot of the most recent result."""
        with self._lock:
            return self._frame, self._faces, self._fps

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        # CAP_DSHOW is the DirectShow backend. On Windows it opens noticeably
        # faster than the default and avoids a long black-screen delay.
        camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)

        if not camera.isOpened():
            self.error = (
                "Could not open camera {}.\n\n"
                "Check that no other app (Zoom, Teams, the Camera app) is "
                "using it, and that Windows camera permission is on for "
                "desktop apps.".format(self.camera_index)
            )
            return

        smoothed_fps = 0.0
        previous_time = time.perf_counter()

        while not self._stop_flag.is_set():
            ok, frame = camera.read()
            if not ok:
                self.error = "The camera stopped sending frames."
                break

            frame = cv2.flip(frame, 1)  # mirror: moving right moves you right
            faces = self.detector.detect(frame)
            draw_overlay(frame, faces)

            # Frames-per-second, smoothed so the number doesn't twitch.
            now = time.perf_counter()
            elapsed = now - previous_time
            previous_time = now
            if elapsed > 0:
                instant = 1.0 / elapsed
                smoothed_fps = instant if smoothed_fps == 0 else (
                    smoothed_fps * 0.9 + instant * 0.1
                )

            with self._lock:
                self._frame = frame
                self._faces = faces
                self._fps = smoothed_fps

        camera.release()


class EmotionApp(ctk.CTk):
    """The main window."""

    def __init__(self):
        super().__init__()

        self.title("Emotion Detector")
        self.geometry("1080x730")
        self.minsize(960, 700)
        self.configure(fg_color=BG_DARK)

        self.detector: EmotionDetector | None = None
        self.camera: CameraThread | None = None
        self.running = False
        self._photo = None  # keeps a reference so Tk does not garbage-collect it

        self._build_layout()

        # Loading the ONNX model takes a moment. Do it just after the window
        # appears, so the user sees the UI immediately instead of a frozen box.
        self.after(80, self._load_detector)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- building the widgets ------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_footer()

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 12))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Emotion Detector",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_MAIN,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Live facial expression analysis  ·  FER+ neural network",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DIM,
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.status_label = ctk.CTkLabel(
            header,
            text="●  Loading model...",   # U+25CF is a filled circle
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DIM,
        )
        self.status_label.grid(row=0, column=2, rowspan=2, sticky="e")

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=24)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # --- video card ----------------------------------------------------
        video_card = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=16)
        video_card.grid(row=0, column=0, sticky="nsew")
        video_card.grid_columnconfigure(0, weight=1)
        video_card.grid_rowconfigure(0, weight=1)

        # A plain tkinter Label, not a CTkLabel, on purpose. CustomTkinter's
        # CTkImage re-scales the picture every time you set it, which measured
        # ~25 ms per frame here versus ~9 ms for ImageTk.PhotoImage. At 30
        # frames a second that difference is the whole frame budget, and it
        # starves the camera thread. Its background is set by hand to match
        # the card so the swap is invisible.
        self.video_label = tk.Label(
            video_card,
            text=PLACEHOLDER_TEXT,
            font=("Segoe UI", 13),
            bg=BG_CARD,
            fg=TEXT_DIM,
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=0, column=0, padx=14, pady=14)

        # --- results panel -------------------------------------------------
        panel = ctk.CTkFrame(body, fg_color=BG_CARD, corner_radius=16, width=330)
        panel.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        panel.grid_propagate(False)  # keep the fixed width even when empty
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="DETECTED EMOTION",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_DIM,
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(22, 6))

        self.emotion_label = ctk.CTkLabel(
            panel,
            text="--",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=TEXT_MAIN,
        )
        self.emotion_label.grid(row=1, column=0, sticky="w", padx=22)

        self.confidence_label = ctk.CTkLabel(
            panel,
            text="waiting for a face",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_DIM,
        )
        self.confidence_label.grid(row=2, column=0, sticky="w", padx=22, pady=(2, 12))

        ctk.CTkFrame(panel, height=1, fg_color=BG_INSET).grid(
            row=3, column=0, sticky="ew", padx=22
        )

        ctk.CTkLabel(
            panel,
            text="ALL SCORES",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_DIM,
        ).grid(row=4, column=0, sticky="w", padx=22, pady=(12, 6))

        # One row per emotion: name on the left, percentage on the right,
        # a coloured bar underneath. Stored in dicts so _update_ui can find them.
        self.bars: dict[str, ctk.CTkProgressBar] = {}
        self.percent_labels: dict[str, ctk.CTkLabel] = {}

        rows = ctk.CTkFrame(panel, fg_color="transparent")
        rows.grid(row=5, column=0, sticky="ew", padx=22, pady=(0, 16))
        rows.grid_columnconfigure(0, weight=1)

        for i, emotion in enumerate(EMOTIONS):
            line = ctk.CTkFrame(rows, fg_color="transparent")
            line.grid(row=i, column=0, sticky="ew", pady=(0, 7))
            line.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                line,
                text=emotion.capitalize(),
                font=ctk.CTkFont(size=12),
                text_color=TEXT_MAIN,
            ).grid(row=0, column=0, sticky="w")

            percent = ctk.CTkLabel(
                line,
                text="0%",
                font=ctk.CTkFont(size=12),
                text_color=TEXT_DIM,
            )
            percent.grid(row=0, column=1, sticky="e")

            bar = ctk.CTkProgressBar(
                line,
                height=6,
                corner_radius=3,
                fg_color=BG_INSET,
                progress_color=EMOTION_COLORS[emotion],
            )
            bar.set(0)
            bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0))

            self.bars[emotion] = bar
            self.percent_labels[emotion] = percent

    def _build_footer(self) -> None:
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.grid(row=2, column=0, sticky="ew", padx=24, pady=(16, 20))
        footer.grid_columnconfigure(2, weight=1)

        self.toggle_button = ctk.CTkButton(
            footer,
            text="Start camera",
            width=150,
            height=40,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color="#4a76e0",
            command=self._toggle_camera,
        )
        self.toggle_button.grid(row=0, column=0)
        self.toggle_button.configure(state="disabled")  # until the model loads

        ctk.CTkLabel(
            footer, text="Camera", font=ctk.CTkFont(size=12), text_color=TEXT_DIM
        ).grid(row=0, column=1, padx=(20, 8))

        self.camera_menu = ctk.CTkOptionMenu(
            footer,
            values=["0", "1", "2"],
            width=70,
            height=36,
            corner_radius=8,
            fg_color=BG_INSET,
            button_color=BG_INSET,
            button_hover_color="#2b3242",
        )
        self.camera_menu.set("0")
        self.camera_menu.grid(row=0, column=2, sticky="w")

        self.fps_label = ctk.CTkLabel(
            footer, text="", font=ctk.CTkFont(size=12), text_color=TEXT_DIM
        )
        self.fps_label.grid(row=0, column=3, sticky="e")

    # -- model loading -------------------------------------------------------

    def _load_detector(self) -> None:
        try:
            self.detector = EmotionDetector()
        except Exception as error:
            self._set_status("●  Model not loaded", "#ef4444")
            self.video_label.configure(text="Could not start:\n\n{}".format(error))
            return

        self._set_status("●  Ready", TEXT_DIM)
        self.toggle_button.configure(state="normal")

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text, text_color=color)

    # -- start / stop --------------------------------------------------------

    def _toggle_camera(self) -> None:
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        if self.detector is None:
            return

        self.detector.reset()
        self.camera = CameraThread(self.detector, int(self.camera_menu.get()))
        self.camera.start()

        self.running = True
        self.toggle_button.configure(text="Stop camera", fg_color="#ef4444",
                                     hover_color="#dc2626")
        self.camera_menu.configure(state="disabled")
        self._set_status("●  Live", "#22c55e")
        self.video_label.configure(text="Starting camera...")

        self._update_ui()

    def _stop_camera(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

        self._photo = None
        self.toggle_button.configure(text="Start camera", fg_color=ACCENT,
                                     hover_color="#4a76e0")
        self.camera_menu.configure(state="normal")
        self._set_status("●  Ready", TEXT_DIM)

        # An empty string is how you clear the image on a plain tk.Label.
        self.video_label.configure(image="", text=PLACEHOLDER_TEXT)
        self.fps_label.configure(text="")
        self.emotion_label.configure(text="--", text_color=TEXT_MAIN)
        self.confidence_label.configure(text="waiting for a face")
        for emotion in EMOTIONS:
            self.bars[emotion].set(0)
            self.percent_labels[emotion].configure(text="0%")

    # -- the repeating GUI refresh ------------------------------------------

    def _update_ui(self) -> None:
        """Runs ~50x a second while the camera is on, then schedules itself."""
        if not self.running or self.camera is None:
            return

        if self.camera.error:
            message = self.camera.error
            self._stop_camera()
            self._set_status("●  Camera error", "#ef4444")
            self.video_label.configure(text=message)
            return

        frame, faces, fps = self.camera.latest()

        if frame is not None:
            # OpenCV gives BGR; PIL and Tk expect RGB, so swap the channels.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Keep the reference on self: Tk does not hold one of its own, so a
            # local variable would be garbage-collected and the video would
            # flicker to blank.
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.video_label.configure(image=self._photo, text="")
            self.fps_label.configure(text="{:.0f} FPS".format(fps))

        if faces:
            main = faces[0]
            self.emotion_label.configure(text=main.label.capitalize(),
                                         text_color=main.color)
            extra = "  ·  {} faces".format(len(faces)) if len(faces) > 1 else ""
            self.confidence_label.configure(
                text="{:.0f}% confident{}".format(main.confidence * 100, extra)
            )
            for i, emotion in enumerate(EMOTIONS):
                value = float(main.probabilities[i])
                self.bars[emotion].set(value)
                self.percent_labels[emotion].configure(
                    text="{:.0f}%".format(value * 100)
                )
        elif frame is not None:
            self.emotion_label.configure(text="--", text_color=TEXT_MAIN)
            self.confidence_label.configure(text="no face detected")
            for emotion in EMOTIONS:
                self.bars[emotion].set(0)
                self.percent_labels[emotion].configure(text="0%")

        # after() asks Tk to call this again in ~33 ms (about 30 times a
        # second), without blocking. Polling faster than the camera can produce
        # frames would only steal CPU from the thread doing the real work.
        self.after(33, self._update_ui)

    # -- shutdown ------------------------------------------------------------

    def _on_close(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.stop()
            self.camera.join(timeout=1.5)  # let the camera release cleanly
        self.destroy()


if __name__ == "__main__":
    EmotionApp().mainloop()
