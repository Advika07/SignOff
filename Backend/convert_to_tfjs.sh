#!/usr/bin/env bash
# Converts the Keras SavedModel produced by train_classifier.py into a
# TensorFlow.js layers model (model.json + weight shards) that the
# frontend loads directly in-browser.
#
# Usage:
#   ./convert_to_tfjs.sh ./model/saved_model ./model/tfjs_model
set -euo pipefail

SAVED_MODEL_DIR="${1:-./model/saved_model}"
TFJS_OUT_DIR="${2:-./model/tfjs_model}"

tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_layers_model \
  "$SAVED_MODEL_DIR" \
  "$TFJS_OUT_DIR"

echo "TF.js model written to $TFJS_OUT_DIR"
echo "Copy $TFJS_OUT_DIR and model/labels.json into your frontend's public/ dir."
