// useSignRecognition.ts
//
// React hook that wires together: webcam -> MediaPipe HandLandmarker ->
// predict.ts classifier -> a `prediction` state you render or compare
// against the current level's target sign.
//
// Usage inside a component (must be a Client Component — add
// 'use client' at the top of the file that uses this hook):
//
//   const videoRef = useRef<HTMLVideoElement>(null);
//   const { prediction, isReady, error, start, stop } = useSignRecognition(videoRef);
//
//   useEffect(() => {
//     start();
//     return () => stop();
//   }, [start, stop]);
//
//   return <video ref={videoRef} autoPlay playsInline muted />;

'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { loadModel, predict, isModelLoaded, type Prediction } from './predict';

interface UseSignRecognitionOptions {
  modelUrl?: string;
  labelsUrl?: string;
  minConfidence?: number;
}

export function useSignRecognition(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  options: UseSignRecognitionOptions = {}
) {
  const {
    modelUrl = '/model/tfjs_model/model.json',
    labelsUrl = '/model/labels.json',
    minConfidence = 0.7,
  } = options;

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafIdRef = useRef<number | null>(null);

  const loop = useCallback(() => {
    const video = videoRef.current;
    const landmarker = handLandmarkerRef.current;
    if (!video || !landmarker) return;

    // Only process if video is ready and has frames
    if (video.readyState !== HTMLMediaElement.HAVE_FUTURE_DATA && 
        video.readyState !== HTMLMediaElement.HAVE_ENOUGH_DATA) {
      rafIdRef.current = requestAnimationFrame(loop);
      return;
    }

    try {
      const results = landmarker.detectForVideo(video, performance.now());

      if (results.landmarks && results.landmarks.length > 0) {
        const result = predict(results.landmarks[0]);
        setPrediction(result.confidence >= minConfidence ? result : null);
      } else {
        setPrediction(null);
      }
    } catch (err) {
      // Silently ignore detection errors from MediaPipe WASM layer
      console.debug('MediaPipe detection error (non-critical):', err);
    }

    rafIdRef.current = requestAnimationFrame(loop);
  }, [videoRef, minConfidence]);

  const start = useCallback(async () => {
    try {
      if (!isModelLoaded()) {
        await loadModel(modelUrl, labelsUrl);
      }

      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        },
        runningMode: 'VIDEO',
        numHands: 1,
      });

      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsReady(true);
      rafIdRef.current = requestAnimationFrame(loop);
    } catch (err) {
      setError(err as Error);
    }
  }, [videoRef, modelUrl, labelsUrl, loop]);

  const stop = useCallback(() => {
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    handLandmarkerRef.current?.close();
    setIsReady(false);
  }, []);

  // safety net: stop the camera if the component unmounts without
  // calling stop() explicitly
  useEffect(() => {
    return () => stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { prediction, isReady, error, start, stop };
}
