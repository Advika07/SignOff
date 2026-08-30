// predict.js
//
// Loads the trained TF.js model + label map, and classifies a single
// frame's hand landmarks (from MediaPipe Hands, running in-browser)
// into a sign class.
//
// IMPORTANT: normalizeLandmarks() here mirrors extract_landmarks.py's
// normalize_landmarks() exactly (translate by wrist, scale by wrist->
// middle-finger-MCP distance). If you change one, change the other,
// or predictions will silently be wrong.

import * as tf from '@tensorflow/tfjs';

let model = null;
let labels = null;

export async function loadModel(
  modelUrl = '/model/tfjs_model/model.json',
  labelsUrl = '/model/labels.json'
) {
  model = await tf.loadLayersModel(modelUrl);
  labels = await fetch(labelsUrl).then((r) => r.json());
  return { model, labels };
}

// landmarks: array of 21 {x, y, z} objects, as returned by
// MediaPipe Hands (@mediapipe/tasks-vision HandLandmarker results).
export function normalizeLandmarks(landmarks) {
  const pts = landmarks.map((p) => [p.x, p.y, p.z]);
  const wrist = pts[0];

  const translated = pts.map(([x, y, z]) => [
    x - wrist[0],
    y - wrist[1],
    z - wrist[2],
  ]);

  const [mx, my, mz] = translated[9]; // middle finger MCP
  const scale = Math.max(Math.sqrt(mx * mx + my * my + mz * mz), 1e-6);

  return translated.flatMap(([x, y, z]) => [x / scale, y / scale, z / scale]); // 63 numbers
}

// Returns { label, confidence, allProbs } for a single hand's landmarks.
export function predict(landmarks) {
  if (!model || !labels) {
    throw new Error('Model not loaded yet — call loadModel() first');
  }

  const vector = normalizeLandmarks(landmarks);

  return tf.tidy(() => {
    const input = tf.tensor2d([vector]);
    const output = model.predict(input);
    const probs = output.dataSync();

    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    return {
      label: labels[maxIdx],
      confidence: probs[maxIdx],
      allProbs: labels.map((label, i) => ({ label, confidence: probs[i] })),
    };
  });
}

export function isModelLoaded() {
  return model !== null && labels !== null;
}
