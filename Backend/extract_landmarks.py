"""
extract_landmarks.py

Walks a dataset folder organized one subfolder per class:

    dataset_root/
        A/
            img1.jpg
            img2.jpg
        B/
            ...
        0/
            ...

Runs MediaPipe Hands on every image, extracts the 21 (x, y, z) hand
landmarks, normalizes them so the model learns hand *shape* rather than
hand position/size/distance from camera, and writes everything to a
single CSV: landmarks_dataset.csv

Usage:
    python extract_landmarks.py --dataset_dir ./data/asl_hg --out landmarks_dataset.csv
"""
import argparse
import csv
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "model" / "hand_landmarker.task"


def ensure_model_downloaded() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading MediaPipe hand model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
    return MODEL_PATH


def build_landmarker() -> HandLandmarker:
    model_path = ensure_model_downloaded()
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples, in MediaPipe's normalized
    image coordinates (0-1 range).

    Returns a 63-float vector that is translation- and scale-invariant:
    - translated so the wrist (landmark 0) sits at the origin
    - scaled by the wrist -> middle-finger-MCP (landmark 9) distance

    This normalization MUST be mirrored exactly in the browser-side
    inference code (see frontend_integration/predict.js), or the
    trained model will see out-of-distribution inputs at inference time.
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)

    wrist = pts[0].copy()
    pts -= wrist

    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale

    return pts.flatten().tolist()  # 63 floats


def extract_from_dataset(dataset_dir: Path, out_csv: Path, min_detection_confidence: float):
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {dataset_dir}")

    labels = [d.name for d in class_dirs]
    print(f"Found {len(labels)} classes: {labels}")

    rows_written = 0
    skipped = 0

    landmarker = build_landmarker()

    try:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label"] + [f"f{i}" for i in range(63)])

            for class_dir in class_dirs:
                label = class_dir.name
                image_paths = [
                    p for p in class_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                ]
                print(f"[{label}] {len(image_paths)} images")

                for img_path in image_paths:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        skipped += 1
                        continue

                    mp_image = Image(image_format=ImageFormat.SRGB, data=image)
                    results = landmarker.detect(mp_image)

                    if not results.hand_landmarks:
                        skipped += 1
                        continue

                    hand_landmarks = results.hand_landmarks[0]
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                    vector = normalize_landmarks(coords)

                    writer.writerow([label] + vector)
                    rows_written += 1
    finally:
        landmarker.close()

    print(f"\nDone. Wrote {rows_written} rows, skipped {skipped} images "
          f"(no hand detected, or unreadable file).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset root (one subfolder per class)")
    parser.add_argument("--out", default="landmarks_dataset.csv", help="Output CSV path")
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    args = parser.parse_args()

    extract_from_dataset(Path(args.dataset_dir), Path(args.out), args.min_detection_confidence)
