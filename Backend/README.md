# Sign Off — model training pipeline

Trains a lightweight, in-browser hand-sign classifier: MediaPipe extracts
21 hand landmarks per frame, a small dense network classifies the
normalized landmark vector, and the whole thing runs client-side via
TensorFlow.js — no inference server needed.

## 1. Get the dataset

Recommended: **ASL-HG** (Mendeley Data) — 36,000 images across 36 classes,
covering the full alphabet (A-Z) and digits (0-9), which covers Level 1
(letters) and Level 2 (numbers) from one source. It ships in two forms:
raw images, and a MediaPipe-preprocessed version with hand-segmented
crops and an 80/20 train-test split already done.

- Direct link: https://data.mendeley.com/datasets/j4y5w2c8w9/1
- DOI: https://doi.org/10.17632/j4y5w2c8w9.1

Download `ASL_Raw_Images.zip` (or the processed version), unzip it so you
end up with one folder per class:

```
data/asl_hg/
  A/
    img001.jpg
    img002.jpg
    ...
  B/
    ...
  0/
    ...
```

For your 2-3 level MVP, you don't need all 36 classes — just copy the
subfolders for the letters/numbers you're supporting into your working
dataset dir (e.g. only A-E for Level 1).

Alternative datasets if you want more images per class or different
framing conditions:
- "American Sign Language A-Z Dataset + Hand Landmarks" (Kaggle) — comes
  with landmarks pre-annotated: https://www.kaggle.com/datasets/srisahithis/american-sign-language-a-z-dataset-hand-landmarks

## 2. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Extract landmarks

```bash
python scripts/extract_landmarks.py \
  --dataset_dir ./data/asl_hg \
  --out landmarks_dataset.csv
```

This runs MediaPipe Hands on every image and writes a CSV where each row
is a 63-number landmark vector (21 points × x,y,z) plus its label.
Images where no hand is detected are skipped and logged.

## 4. Train the classifier

```bash
python scripts/train_classifier.py \
  --csv landmarks_dataset.csv \
  --out_dir ./model
```

Trains a small dense network (128 → 64 → softmax) on the landmark
vectors. Outputs:
- `model/saved_model/` — Keras SavedModel
- `model/labels.json` — class index → sign name mapping

## 5. Convert to TensorFlow.js

```bash
chmod +x scripts/convert_to_tfjs.sh
./scripts/convert_to_tfjs.sh ./model/saved_model ./model/tfjs_model
```

Copy `model/tfjs_model/` and `model/labels.json` into your frontend's
`public/model/` directory.

## 6. Wire it into the frontend

`frontend_integration/predict.js` loads the TF.js model and classifies
a single frame's landmarks — its normalization logic mirrors
`extract_landmarks.py` exactly, so keep them in sync if you ever change
one.

`frontend_integration/useSignRecognition.js` is a Vue composable that
wires webcam → MediaPipe HandLandmarker → `predict.js` → a reactive
`prediction` ref, ready to drop into your v0.dev components:

```vue
<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { useSignRecognition } from './useSignRecognition.js';

const videoRef = ref(null);
const { prediction, isReady, start, stop } = useSignRecognition(videoRef);

onMounted(start);
onUnmounted(stop);
</script>

<template>
  <video ref="videoRef" autoplay playsinline muted />
  <p v-if="prediction">Detected: {{ prediction.label }} ({{ (prediction.confidence * 100).toFixed(0) }}%)</p>
</template>
```

Install the frontend deps it needs:

```bash
npm install @tensorflow/tfjs @mediapipe/tasks-vision
```

## Notes

- Public datasets are captured from a narrow set of signers — accuracy
  can dip for hand shapes/skin tones/angles outside the training
  distribution. Worth a calibration step or at least a disclaimer.
- Static poses only (this pipeline doesn't handle motion). Signs needing
  movement (J, Z, dynamic words) would need a sequence model over a
  buffered window of frames — treat as a stretch goal.
