"""
train_classifier.py

Trains a small dense neural network on the landmark vectors produced by
extract_landmarks.py, and exports a Keras SavedModel ready for
conversion to TensorFlow.js (see convert_to_tfjs.sh).

Usage:
    python train_classifier.py --csv landmarks_dataset.csv --out_dir ./model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(csv_path: Path, out_dir: Path, epochs: int, batch_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], num_classes=len(encoder.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f} (loss: {val_loss:.4f})")

    # Keras SavedModel — this is the input the tfjs converter expects
    saved_model_path = out_dir / "saved_model"
    model.export(str(saved_model_path))

    # Label mapping so the frontend can turn a predicted class index
    # back into a sign name ("A", "0", etc.) — order matches the
    # softmax output index.
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(list(encoder.classes_), f, indent=2)

    print(f"Saved model to: {saved_model_path}")
    print(f"Saved label map to: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default="./model")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(Path(args.csv), Path(args.out_dir), args.epochs, args.batch_size)
