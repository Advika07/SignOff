// useSignRecognition.js
//
// Vue composable that wires together: webcam -> MediaPipe HandLandmarker
// -> predict.js classifier -> a reactive `prediction` you can render or
// compare against the level's target sign.
//
// Usage in a component:
//
//   const videoRef = ref(null);
//   const { prediction, start, stop } = useSignRecognition(videoRef);
//   onMounted(start);
//   onUnmounted(stop);
//
// Then in the template: <video ref="videoRef" autoplay playsinline muted />

import { ref, shallowRef } from 'vue';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { loadModel, predict, isModelLoaded } from './predict.js';

export function useSignRecognition(videoRef, options = {}) {
  const {
    modelUrl = '/model/tfjs_model/model.json',
    labelsUrl = '/model/labels.json',
    minConfidence = 0.7,
  } = options;

  const prediction = ref(null); // { label, confidence } | null
  const isReady = ref(false);
  const error = ref(null);

  const handLandmarker = shallowRef(null);
  let stream = null;
  let rafId = null;

  async function start() {
    try {
      if (!isModelLoaded()) {
        await loadModel(modelUrl, labelsUrl);
      }

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      handLandmarker.value = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        },
        runningMode: 'VIDEO',
        numHands: 1,
      });

      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.value.srcObject = stream;
      await videoRef.value.play();

      isReady.value = true;
      loop();
    } catch (err) {
      error.value = err;
    }
  }

  function loop() {
    const video = videoRef.value;
    if (!video || !handLandmarker.value) return;

    const results = handLandmarker.value.detectForVideo(video, performance.now());

    if (results.landmarks && results.landmarks.length > 0) {
      const result = predict(results.landmarks[0]);
      prediction.value = result.confidence >= minConfidence ? result : null;
    } else {
      prediction.value = null;
    }

    rafId = requestAnimationFrame(loop);
  }

  function stop() {
    if (rafId) cancelAnimationFrame(rafId);
    if (stream) stream.getTracks().forEach((track) => track.stop());
    handLandmarker.value?.close();
    isReady.value = false;
  }

  return { prediction, isReady, error, start, stop };
}
