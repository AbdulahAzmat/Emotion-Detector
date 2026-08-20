"""
download_model.py
=================
Downloads the pre-trained emotion model (FER+) into the models/ folder.

You only ever need to run this once:

    python download_model.py

Why a separate file? The model is ~33 MB, which is too big to ship inside a
code project. Keeping the download in its own script means app.py stays about
the app, and you can re-run this if the file ever gets corrupted.

About the model: "FER+" was trained by Microsoft Research on the FER2013 face
dataset, re-labelled by 10 human taggers per image. It is published in the ONNX
format, an open standard that lets one file run under many different runtimes --
here, onnxruntime.
"""

import os
import sys
import urllib.request

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion-ferplus-8.onnx")

# The real file is about 33 MiB. If we end up with far less than that, the
# download was cut short (or we saved an error page instead of the model).
MIN_EXPECTED_BYTES = 30 * 1024 * 1024


def show_progress(block_number: int, block_size: int, total_size: int) -> None:
    """Called repeatedly by urlretrieve so we can print a progress bar."""
    downloaded = block_number * block_size
    if total_size <= 0:
        sys.stdout.write("\r  downloaded {:.1f} MB".format(downloaded / 1e6))
        sys.stdout.flush()
        return

    fraction = min(downloaded / total_size, 1.0)
    filled = int(fraction * 30)
    bar = "#" * filled + "-" * (30 - filled)
    sys.stdout.write("\r  [{}] {:5.1f}%  ({:.1f} / {:.1f} MB)".format(
        bar, fraction * 100, downloaded / 1e6, total_size / 1e6))
    sys.stdout.flush()


def main() -> int:
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_EXPECTED_BYTES:
        size_mb = os.path.getsize(MODEL_PATH) / 1e6
        print("Model is already here, nothing to do.")
        print("  {}  ({:.1f} MB)".format(MODEL_PATH, size_mb))
        return 0

    print("Downloading the FER+ emotion model (~33 MB)...")
    print("  from: " + MODEL_URL)
    print("  to:   " + MODEL_PATH)

    # Download to a temporary name first, then rename. That way an interrupted
    # download can never leave a half-written file that looks valid.
    temp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, temp_path, show_progress)
    except Exception as error:
        print("\n\nDownload failed: {}".format(error))
        print("Check your internet connection and try again.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return 1

    print()  # end the progress-bar line

    if os.path.getsize(temp_path) < MIN_EXPECTED_BYTES:
        print("The downloaded file is too small -- the download was incomplete.")
        print("Please run this script again.")
        os.remove(temp_path)
        return 1

    os.replace(temp_path, MODEL_PATH)  # atomic on Windows and Linux alike
    print("Done. Saved {:.1f} MB.".format(os.path.getsize(MODEL_PATH) / 1e6))
    print("\nNow start the app with:  python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
