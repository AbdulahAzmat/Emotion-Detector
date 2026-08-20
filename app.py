"""
app.py
======
The graphical app. Run it with:  python app.py

It works in two modes:

  * LIVE CAMERA -- reads your webcam continuously. When several people are in
    shot, every face gets a box, but the panel reports on the largest (nearest)
    one, because a live readout has to pick someone without being asked.

  * PICTURE -- you choose an image file. Every face found is numbered, and you
    pick which one to analyse, either from the dropdown or by clicking it.
    Nothing is assumed on your behalf.

How it is put together
----------------------
In live mode there are two things happening at once:

  * The CAMERA THREAD grabs frames from the webcam and runs them through the
    EmotionDetector. This is the slow part (tens of milliseconds per frame).
  * The MAIN THREAD runs the GUI and redraws the window.

They must be separate. If you grabbed camera frames inside the GUI thread, the
window would freeze solid between frames -- no button clicks, no dragging. So
the camera thread just keeps a "latest result" in a shared box, and the GUI
peeks into that box about 30 times a second and paints whatever it finds.

`threading.Lock` guards that shared box so the two threads never read and write
it at the same instant.

Picture mode needs none of that: one image, analysed once, on the spot.
"""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from emotion_detector import (
    EMOTIONS,
    EMOTION_COLORS,
    EmotionDetector,
    Face,
    draw_overlay,
    load_image,
)

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
ACCENT = "#15803d"        # dark green: white label text clears 4.5:1 on it,
ACCENT_HOVER = "#166534"  # and it still reads 3.46:1 against the background
DANGER = "#ef4444"
OK_GREEN = "#22c55e"

VIDEO_W, VIDEO_H = 640, 480

PLACEHOLDER_LIVE = "Camera is off\n\nPress “Start camera” below"
PLACEHOLDER_IMAGE = "No picture loaded\n\nPress “Choose picture…” below"

# Still images are analysed at this size at most. Detecting on a full 12
# megapixel phone photo is needlessly slow; shrinking the long side to 1400 px
# keeps every realistic face well above the minimum size while capping the
# work at roughly a tenth of a second.
MAX_ANALYSIS_SIDE = 1400

# How hard to correct for FER+ being trained mostly on neutral and happy
# faces. 0 leaves the model's own bias intact; higher values give the rare
# emotions more of a chance to win. See rebalance() in emotion_detector.py.
BALANCE_LEVELS = {
    "Off": 0.0,
    "Balanced": 0.5,
    "Strong": 0.8,
}


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
            draw_overlay(frame, faces, highlight=0)

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
        self.mode = "live"
        self._photo = None  # keeps a reference so Tk does not garbage-collect it

        # Picture-mode state.
        self.image_original = None      # the picture as loaded, full size
        self.image_display = None       # the shrunk-to-fit copy actually shown
        self.image_faces: list[Face] = []      # boxes in display coordinates
        self.image_name = ""            # filename, for re-analysing
        self.selected_face = 0

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
            text="Facial expression analysis  ·  FER+ neural network",
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

        # --- video / picture card ------------------------------------------
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
            text=PLACEHOLDER_LIVE,
            font=("Segoe UI", 13),
            bg=BG_CARD,
            fg=TEXT_DIM,
            bd=0,
            highlightthickness=0,
        )
        self.video_label.grid(row=0, column=0, padx=14, pady=14)

        # Click a face in a still picture to select it. Bound always, but the
        # handler ignores clicks unless we are in picture mode.
        self.video_label.bind("<Button-1>", self._on_picture_click)

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
        # a coloured bar underneath. Stored in dicts so the update code can
        # find them again by emotion name.
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

            # Square ends, not rounded: corner_radius=0. At this thickness a
            # rounded cap eats most of the bar, so a 2% score renders as a dot
            # rather than a short line.
            bar = ctk.CTkProgressBar(
                line,
                height=4,
                corner_radius=0,
                border_width=0,
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

        # --- mode switch ----------------------------------------------------
        self.mode_switch = ctk.CTkSegmentedButton(
            footer,
            values=["Live camera", "Picture"],
            height=40,
            corner_radius=0,   # squared off; the dropdowns stay rounded
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=BG_INSET,
            unselected_color=BG_INSET,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_hover_color="#2b3242",
            command=self._on_mode_change,
        )
        self.mode_switch.set("Live camera")
        self.mode_switch.grid(row=0, column=0, sticky="w")

        # --- per-mode controls ----------------------------------------------
        # Both frames live in the same grid cell; only one is shown at a time.
        controls = ctk.CTkFrame(footer, fg_color="transparent")
        controls.grid(row=0, column=1, sticky="w", padx=(16, 0))

        # Live-camera controls.
        self.live_controls = ctk.CTkFrame(controls, fg_color="transparent")

        self.toggle_button = ctk.CTkButton(
            self.live_controls,
            text="Start camera",
            width=150,
            height=40,
            corner_radius=0,   # squared off; the dropdowns stay rounded
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._toggle_camera,
        )
        self.toggle_button.grid(row=0, column=0)
        self.toggle_button.configure(state="disabled")  # until the model loads

        ctk.CTkLabel(
            self.live_controls, text="Camera",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
        ).grid(row=0, column=1, padx=(16, 8))

        self.camera_menu = ctk.CTkOptionMenu(
            self.live_controls,
            values=["0", "1", "2"],
            width=70,
            height=36,
            corner_radius=8,
            fg_color=BG_INSET,
            button_color=BG_INSET,
            button_hover_color="#2b3242",
        )
        self.camera_menu.set("0")
        self.camera_menu.grid(row=0, column=2)

        # Picture controls.
        self.picture_controls = ctk.CTkFrame(controls, fg_color="transparent")

        self.choose_button = ctk.CTkButton(
            self.picture_controls,
            text="Choose picture…",
            width=170,
            height=40,
            corner_radius=0,   # squared off; the dropdowns stay rounded
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._choose_picture,
        )
        self.choose_button.grid(row=0, column=0)
        self.choose_button.configure(state="disabled")

        self.face_caption = ctk.CTkLabel(
            self.picture_controls, text="Face",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
        )
        self.face_caption.grid(row=0, column=1, padx=(16, 8))

        self.face_menu = ctk.CTkOptionMenu(
            self.picture_controls,
            values=["--"],
            width=110,
            height=36,
            corner_radius=8,
            fg_color=BG_INSET,
            button_color=BG_INSET,
            button_hover_color="#2b3242",
            command=self._on_face_chosen,
        )
        self.face_menu.set("--")
        self.face_menu.configure(state="disabled")
        self.face_menu.grid(row=0, column=2)

        self.live_controls.grid(row=0, column=0)  # live is the starting mode

        # --- rare-emotion sensitivity ---------------------------------------
        # FER+ was trained on data that is mostly neutral and happy faces, so
        # left alone it rarely reports fear, disgust or contempt. This dials
        # how hard to correct for that. See rebalance() in emotion_detector.py.
        sensitivity = ctk.CTkFrame(footer, fg_color="transparent")
        sensitivity.grid(row=0, column=2, sticky="w", padx=(24, 0))

        ctk.CTkLabel(
            sensitivity, text="Rare emotions",
            font=ctk.CTkFont(size=12), text_color=TEXT_DIM,
        ).grid(row=0, column=0, padx=(0, 8))

        self.balance_menu = ctk.CTkOptionMenu(
            sensitivity,
            values=list(BALANCE_LEVELS),
            width=110,
            height=36,
            corner_radius=8,
            fg_color=BG_INSET,
            button_color=BG_INSET,
            button_hover_color="#2b3242",
            command=self._on_balance_change,
        )
        self.balance_menu.set("Balanced")
        self.balance_menu.grid(row=0, column=1)

        self.info_label = ctk.CTkLabel(
            footer, text="", font=ctk.CTkFont(size=12), text_color=TEXT_DIM
        )
        self.info_label.grid(row=0, column=3, sticky="e", padx=(24, 0))

    # -- model loading -------------------------------------------------------

    def _load_detector(self) -> None:
        try:
            self.detector = EmotionDetector()
        except Exception as error:
            self._set_status("●  Model not loaded", DANGER)
            self.video_label.configure(text="Could not start:\n\n{}".format(error))
            return

        # Pay the one-off model start-up cost now, while the window is
        # already up, instead of letting it stall the first real action.
        self.detector.balance = BALANCE_LEVELS.get(self.balance_menu.get(), 0.5)
        self.detector.warm_up()

        self._set_status("●  Ready", TEXT_DIM)
        self.toggle_button.configure(state="normal")
        self.choose_button.configure(state="normal")

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.configure(text=text, text_color=color)

    # -- switching between live and picture ----------------------------------

    def _on_mode_change(self, value: str) -> None:
        new_mode = "live" if value == "Live camera" else "picture"
        if new_mode == self.mode:
            return

        if self.running:
            self._stop_camera()   # never leave the webcam running unseen

        self.mode = new_mode
        self._photo = None
        self._clear_panel()
        self.info_label.configure(text="")

        if new_mode == "live":
            self.picture_controls.grid_forget()
            self.live_controls.grid(row=0, column=0)
            self.video_label.configure(image="", text=PLACEHOLDER_LIVE,
                                       cursor="")
            self._set_status("●  Ready", TEXT_DIM)
        else:
            self.live_controls.grid_forget()
            self.picture_controls.grid(row=0, column=0)
            self.video_label.configure(image="", text=PLACEHOLDER_IMAGE,
                                       cursor="")
            self._reset_face_menu()
            self._set_status("●  Ready", TEXT_DIM)

    # -- picture mode --------------------------------------------------------

    def _choose_picture(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a picture",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.jfif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return  # the user cancelled the dialog

        image = load_image(path)
        if image is None:
            self._set_status("●  Could not read that file", DANGER)
            self.video_label.configure(
                image="",
                text="That file could not be opened as a picture.\n\n"
                     "Try a .jpg or .png.",
            )
            self._clear_panel()
            self._reset_face_menu()
            return

        self.image_original = image
        self.image_name = os.path.basename(path)
        self._analyse_picture(self.image_name)

    def _analyse_picture(self, filename: str) -> None:
        """Find every face in the loaded picture and show them numbered."""
        if self.detector is None or self.image_original is None:
            return

        self._set_status("●  Analysing...", TEXT_DIM)
        self.update_idletasks()   # let that status actually paint first

        original = self.image_original
        height, width = original.shape[:2]

        # Analyse at a sensible size, and display at a size that fits the card.
        # These are two different scales, so faces get converted between them.
        analysis_scale = min(1.0, MAX_ANALYSIS_SIDE / max(width, height))
        display_scale = min(VIDEO_W / width, VIDEO_H / height, 1.0)

        if analysis_scale < 1.0:
            work = cv2.resize(original, None, fx=analysis_scale, fy=analysis_scale,
                              interpolation=cv2.INTER_AREA)
        else:
            work = original

        # A photo is one fixed moment, so smoothing is switched off. The face
        # limit is raised and the minimum size lowered, because a group photo
        # has more faces and smaller ones than a webcam selfie.
        # scale=0.5 searches a half-size copy: measured 101 ms against 294 ms
        # at full size on a 1100x700 photo, finding exactly the same faces.
        found = self.detector.detect(
            work, max_faces=12, smooth=False, min_face=56, scale=0.5
        )

        # Convert the boxes from analysis coordinates into display coordinates.
        to_display = display_scale / analysis_scale
        self.image_faces = [f.scaled(to_display) for f in found]

        self.image_display = cv2.resize(
            original,
            (max(1, int(width * display_scale)), max(1, int(height * display_scale))),
            interpolation=cv2.INTER_AREA,
        )

        if not self.image_faces:
            self._photo = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(self.image_display, cv2.COLOR_BGR2RGB))
            )
            self.video_label.configure(image=self._photo, text="", cursor="")
            self._clear_panel()
            self.confidence_label.configure(text="no faces found in this picture")
            self._reset_face_menu()
            self._set_status("●  No faces found", DANGER)
            self.info_label.configure(text=filename)
            return

        # Fill the picker, then select the first (largest) face by default --
        # but the choice stays the user's to change.
        names = ["Face {}".format(i + 1) for i in range(len(self.image_faces))]
        self.face_menu.configure(values=names, state="normal")
        self.face_menu.set(names[0])
        self.selected_face = 0

        count = len(self.image_faces)
        self._set_status("●  {} face{} found".format(count, "" if count == 1 else "s"),
                         OK_GREEN)
        self.info_label.configure(
            text="{}   ·   click a face to switch".format(filename)
            if count > 1 else filename
        )

        self._render_picture()

    def _render_picture(self) -> None:
        """Redraw the still image with the current face highlighted."""
        if self.image_display is None:
            return

        canvas = self.image_display.copy()
        draw_overlay(canvas, self.image_faces,
                     highlight=self.selected_face, numbered=True)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(
            image=self._photo, text="",
            cursor="hand2" if len(self.image_faces) > 1 else "",
        )

        face = self.image_faces[self.selected_face]
        extra = "  ·  face {} of {}".format(self.selected_face + 1,
                                            len(self.image_faces))
        self._show_face(face, extra if len(self.image_faces) > 1 else "")

    def _on_face_chosen(self, value: str) -> None:
        """The dropdown changed -- 'Face 3' means index 2."""
        try:
            index = int(value.split()[-1]) - 1
        except (ValueError, IndexError):
            return
        if 0 <= index < len(self.image_faces):
            self.selected_face = index
            self._render_picture()

    def _on_picture_click(self, event) -> None:
        """Clicking directly on a face in the picture selects it."""
        if self.mode != "picture" or not self.image_faces:
            return

        # The label is sized to the image and has no border, so the click
        # coordinates are already image coordinates.
        for i, face in enumerate(self.image_faces):
            if face.contains(event.x, event.y):
                if i != self.selected_face:
                    self.selected_face = i
                    self.face_menu.set("Face {}".format(i + 1))
                    self._render_picture()
                return

    def _on_balance_change(self, value: str) -> None:
        """Rare-emotion sensitivity changed -- apply it and redo the reading."""
        if self.detector is None:
            return
        self.detector.balance = BALANCE_LEVELS.get(value, 0.5)

        # Old smoothed frames were scored under the previous setting, so they
        # would drag the new reading back toward the old one.
        self.detector.reset()

        # Live mode picks the change up on its next frame by itself, but a
        # still picture has to be scored again to show any difference.
        if self.mode == "picture" and self.image_original is not None:
            self._analyse_picture(self.image_name)

    def _reset_face_menu(self) -> None:
        self.image_faces = []
        self.selected_face = 0
        self.face_menu.configure(values=["--"], state="disabled")
        self.face_menu.set("--")

    # -- live mode -----------------------------------------------------------

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
        self.toggle_button.configure(text="Stop camera", fg_color=DANGER,
                                     hover_color="#dc2626")
        self.camera_menu.configure(state="disabled")
        self.mode_switch.configure(state="disabled")
        self._set_status("●  Live", OK_GREEN)
        self.video_label.configure(text="Starting camera...")

        self._update_ui()

    def _stop_camera(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.stop()
            self.camera = None

        self._photo = None
        self.toggle_button.configure(text="Start camera", fg_color=ACCENT,
                                     hover_color=ACCENT_HOVER)
        self.camera_menu.configure(state="normal")
        self.mode_switch.configure(state="normal")
        self._set_status("●  Ready", TEXT_DIM)

        # An empty string is how you clear the image on a plain tk.Label.
        self.video_label.configure(image="", text=PLACEHOLDER_LIVE)
        self.info_label.configure(text="")
        self._clear_panel()

    def _update_ui(self) -> None:
        """Runs ~30x a second while the camera is on, then schedules itself."""
        if not self.running or self.camera is None:
            return

        if self.camera.error:
            message = self.camera.error
            self._stop_camera()
            self._set_status("●  Camera error", DANGER)
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
            self.info_label.configure(text="{:.0f} FPS".format(fps))

        if faces:
            # Live mode has to choose someone unprompted, so it reports on the
            # largest face -- the person nearest the camera. Everyone else
            # still gets a box drawn, just in grey.
            extra = "  ·  {} faces".format(len(faces)) if len(faces) > 1 else ""
            self._show_face(faces[0], extra)
        elif frame is not None:
            self._clear_panel()
            self.confidence_label.configure(text="no face detected")

        # after() asks Tk to call this again in ~33 ms (about 30 times a
        # second), without blocking. Polling faster than the camera can produce
        # frames would only steal CPU from the thread doing the real work.
        self.after(33, self._update_ui)

    # -- the results panel ---------------------------------------------------

    def _show_face(self, face: Face, extra: str = "") -> None:
        """Point the panel at one face."""
        self.emotion_label.configure(text=face.label.capitalize(),
                                     text_color=face.color)
        self.confidence_label.configure(
            text="{:.0f}% confident{}".format(face.confidence * 100, extra)
        )
        for i, emotion in enumerate(EMOTIONS):
            value = float(face.probabilities[i])
            self.bars[emotion].set(value)
            self.percent_labels[emotion].configure(text="{:.0f}%".format(value * 100))

    def _clear_panel(self) -> None:
        self.emotion_label.configure(text="--", text_color=TEXT_MAIN)
        self.confidence_label.configure(text="waiting for a face")
        for emotion in EMOTIONS:
            self.bars[emotion].set(0)
            self.percent_labels[emotion].configure(text="0%")

    # -- shutdown ------------------------------------------------------------

    def _on_close(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.stop()
            self.camera.join(timeout=1.5)  # let the camera release cleanly
        self.destroy()


if __name__ == "__main__":
    EmotionApp().mainloop()
