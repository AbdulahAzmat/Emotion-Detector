# Emotion Detector

A Python desktop app that looks at a face and guesses which emotion it's
showing. It works two ways: live off your webcam, or on a picture you pick.

Everything runs locally. Nothing gets uploaded anywhere.

![Screenshot of the app](docs/screenshot.png)

## What it uses

- **OpenCV** for the camera and for finding faces (Haar cascade)
- **FER+**, a pre-trained neural network from Microsoft Research, for the
  actual emotion classification. It's an ONNX file that runs on the CPU.
- **CustomTkinter** for the GUI

## Setup

You need Python 3.9+ and a webcam.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

If PowerShell complains about scripts being disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

```bash
pip install -r requirements.txt
```

```bash
python download_model.py
```

That last one grabs the 33 MB model file. You only need to do it once — it's
too big to keep in the repo.

Then:

```bash
python app.py
```

### In VS Code

Open the **folder** (not just the file), then hit **Ctrl + Shift + P** →
`Python: Select Interpreter` → pick the one with `.venv` in the path. Skipping
that step is why imports sometimes show red squiggles even though the code
runs fine.

There's a `launch.json` included, so F5 just works.

## The files

| File | What it does |
| --- | --- |
| `app.py` | The GUI. Run this one. |
| `emotion_detector.py` | Face detection + emotion classification. No GUI code. |
| `download_model.py` | Downloads the model. Run once. |

I kept the detection logic completely separate from the GUI, so you can test it
on its own:

```bash
python emotion_detector.py
```

That opens a plain OpenCV window. Press Q to quit. If that works but `app.py`
doesn't, the bug is in the GUI, not the detection.

## How it works

Every frame goes through the same steps:

1. Convert to grayscale — colour doesn't help and it's slower
2. Find faces with a Haar cascade (searched on a half-size copy, which is ~25%
   faster and moves the boxes by a pixel or two at most)
3. Crop each face and resize to 64x64, since that's what FER+ was trained on
4. Run the model, softmax the 8 output scores into percentages
5. Average the last 6 frames so the label doesn't flicker

### Multiple faces

Live mode draws a box on every face but only reports on the biggest one, since
it has to pick someone without asking. Picture mode numbers every face it finds
and lets you choose, either from the dropdown or by clicking the face.

Stills also use different settings: smoothing is off (a photo is one moment, so
there are no recent frames to average with), more faces are allowed, and the
minimum face size is lower because group photos have smaller faces.

### The threading bit

Detection takes 25-35 ms per frame. If that ran in the GUI thread the window
would lock up between frames, so the camera runs on its own thread and drops
its latest result into a variable that the GUI reads ~30 times a second. A
`Lock` keeps them from touching it at the same time.

### Why it kept saying "neutral"

FER+ learned from data that's mostly neutral and happy faces — fear, disgust
and contempt are only a few percent of it. So the model treats neutral as a
safe guess and rarely reports the rare ones.

The **Rare emotions** dropdown fixes this. It divides each score by how common
that emotion was in training, which cancels out the head start the common ones
got. `Balanced` is the default; `Strong` pushes harder but gives more false
alarms. It only affects borderline cases — a clearly happy face still reads as
happy on every setting.

## Things to try changing

- `smoothing=6` in `emotion_detector.py` — set it to 1 to see the raw jittery
  output, or 20 for something very slow
- `minNeighbors=6` in `detect()` — drop it to 3 and it finds more faces, but
  also starts seeing faces in doorknobs
- `RAMP_LIGHT` / `RAMP_DARK` — every bar colour is blended between these two,
  so changing them re-themes the panel. Try `#ffe6c7` and `#c2410c`.
- `OVERLAY_COLORS` — the colours of the boxes drawn on faces. These are kept
  separate from the ramp because pale colours disappear on light photos.

## If it doesn't work

**`module 'cv2' has no attribute 'CascadeClassifier'`**
You have OpenCV 5, which removed it. Run
`pip install "opencv-python>=4.8,<5"`. `requirements.txt` already pins this.

**`Emotion model not found`**
Run `python download_model.py`.

**`Could not open camera 0`**
Something else is using it — Zoom, Teams, the Camera app. Close it. Also check
Settings → Privacy → Camera and allow desktop apps. Try index 1 or 2 if you
have more than one camera.

**Dark video, or no face detected**
Face detection needs light. Check for a privacy shutter on the webcam and face
a window or lamp. A dark webcam also drops to a very low frame rate, which
makes the whole app feel slow — that's the camera, not the code.

**No faces found in a picture that clearly has them**
Haar cascades want fairly front-on faces. Profiles, tilted heads and heavy
shadow all get missed. Lower `min_face` in `_analyse_picture`, or
`minNeighbors` in `detect()`, at the cost of more false positives.

## Limitations

Worth being honest about:

- It reads **expressions, not feelings**. A polite smile and real happiness
  look identical to it.
- Posed emotions are hard. Deliberately "looking sad" at a webcam produces
  something much subtler than real sadness, and fear is nearly impossible to
  fake convincingly — most people just widen their eyes, which reads as
  surprise.
- FER2013, the training data, is small and was labelled by people guessing from
  photos. Accuracy varies with lighting, glasses, and skin tone.
- Contempt and disgust are the weakest classes by far, since there was barely
  any training data for them.

## Credits

- FER+ model by Barsoum et al. at Microsoft Research, from the
  [ONNX Model Zoo](https://github.com/onnx/models) (MIT licence)
- Face detection: Viola-Jones Haar cascade, bundled with OpenCV
