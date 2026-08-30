import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAVED_MODEL_DIR = MODEL_DIR / "saved_model"
OUT_DIR = MODEL_DIR / "tfjs_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topology():
    return {
        "class_name": "Sequential",
        "config": {
            "name": "sequential",
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {
                        "batch_input_shape": [None, 63],
                        "dtype": "float32",
                        "sparse": False,
                        "name": "dense_input",
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 128,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "VarianceScaling",
                            "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_1",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 64,
                        "activation": "relu",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "VarianceScaling",
                            "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_2",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 24,
                        "activation": "softmax",
                        "use_bias": True,
                        "kernel_initializer": {
                            "class_name": "VarianceScaling",
                            "config": {"scale": 1.0, "mode": "fan_avg", "distribution": "uniform", "seed": None},
                        },
                        "bias_initializer": {"class_name": "Zeros", "config": {}},
                        "kernel_regularizer": None,
                        "bias_regularizer": None,
                        "activity_regularizer": None,
                        "kernel_constraint": None,
                        "bias_constraint": None,
                    },
                },
            ],
        },
        "keras_version": tf.keras.__version__,
        "backend": "tensorflow",
    }


def export_tfjs_manual():
    if not SAVED_MODEL_DIR.exists():
        raise FileNotFoundError(f"SavedModel not found at {SAVED_MODEL_DIR}")

    loaded = tf.saved_model.load(str(SAVED_MODEL_DIR))
    vars_by_name = {v.name.replace(':0', ''): v.numpy() for v in loaded.trainable_variables}

    manifest_weights = []
    arrays = []
    for layer_name in ["sequential/dense/kernel", "sequential/dense/bias", "sequential/dense_1/kernel", "sequential/dense_1/bias", "sequential/dense_2/kernel", "sequential/dense_2/bias"]:
        arr = np.asarray(vars_by_name[layer_name], dtype=np.float32)
        manifest_weights.append({
            "name": layer_name,
            "shape": list(arr.shape),
            "dtype": "float32",
        })
        arrays.append(arr.reshape(-1))

    weights_bytes = np.concatenate(arrays).astype(np.float32).tobytes()
    bin_path = OUT_DIR / "group1-shard1of1.bin"
    bin_path.write_bytes(weights_bytes)

    model_json = {
        "format": "layers-model",
        "generatedBy": "manual-export-script",
        "convertedBy": "SignOff backend",
        "modelTopology": build_topology(),
        "weightsManifest": [{
            "paths": [bin_path.name],
            "weights": manifest_weights,
        }],
    }

    (OUT_DIR / "model.json").write_text(json.dumps(model_json, separators=(',', ':')))
    print(f"Exported tfjs model to: {OUT_DIR}")
    print(f"Weights file: {bin_path}")
    print(f"Model file: {OUT_DIR / 'model.json'}")


if __name__ == "__main__":
    export_tfjs_manual()
