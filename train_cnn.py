#!/usr/bin/env python3
"""Train the optional CNN eye-state classifier (open vs closed) and export TensorFlow Lite.

Dataset: Kaggle "MRL Eye Dataset" (or any folder tree of eye crops). Labels are read from
  * parent folder names  open/ closed/  (case-insensitive), or
  * the MRL file name  s0001_00001_0_0_0_0_0_01.png  (5th field: 0 = closed, 1 = open).
The split is done BY SUBJECT (s0001 ...) so the test set contains people the model never saw.

  python train_cnn.py --data /path/to/mrlEyes --out models --max-per-class 20000
"""
import argparse
import glob
import os
import random
import re

import cv2
import numpy as np

SIZE = 64


def collect(root):
    items = []
    for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True):
        if not p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        name, parent = os.path.basename(p), os.path.basename(os.path.dirname(p)).lower()
        parts = name.split("_")
        m = re.match(r"(s\d+)_", name)
        label = None
        if parent in ("closed", "close", "closedeyes", "closed_eyes"):
            label = 1
        elif parent in ("open", "opened", "openeyes", "open_eyes"):
            label = 0
        elif m and len(parts) >= 5 and parts[4] in ("0", "1"):
            label = 1 if parts[4] == "0" else 0
        if label is None:
            continue
        items.append((p, label, m.group(1) if m else None))
    return items


def split_by_subject(items, seed=0):
    rnd = random.Random(seed)
    subjects = sorted({s for _, _, s in items if s})
    if len(subjects) < 6:
        print("WARNING: no subject ids in file names -> random split per image (test score will be optimistic).")
        rnd.shuffle(items)
        n = len(items)
        return items[: int(.7 * n)], items[int(.7 * n): int(.85 * n)], items[int(.85 * n):]
    rnd.shuffle(subjects)
    n = len(subjects)
    tr, va = set(subjects[: int(.7 * n)]), set(subjects[int(.7 * n): int(.85 * n)])
    out = ([], [], [])
    for it in items:
        out[0 if it[2] in tr else 1 if it[2] in va else 2].append(it)
    return out


def load(items, cap_per_class, seed=0):
    rnd = random.Random(seed)
    by = {0: [], 1: []}
    for it in items:
        by[it[1]].append(it)
    chosen = []
    for k in by:
        rnd.shuffle(by[k])
        chosen += by[k][:cap_per_class]
    rnd.shuffle(chosen)
    X, y = [], []
    for p, l, _ in chosen:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(cv2.resize(img, (SIZE, SIZE)))
        y.append(l)
    return np.asarray(X, np.uint8)[..., None], np.asarray(y, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-per-class", type=int, default=20000)
    a = ap.parse_args()

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow.keras import layers, models

    items = collect(a.data)
    print(f"found {len(items)} labelled images ({sum(l for _, l, _ in items)} closed)")
    if not items:
        raise SystemExit("No labelled images found - check --data and the folder/file naming.")
    tr, va, te = split_by_subject(items)
    Xtr, ytr = load(tr, a.max_per_class)
    Xva, yva = load(va, a.max_per_class // 4)
    Xte, yte = load(te, a.max_per_class // 4)
    print("train/val/test:", len(Xtr), len(Xva), len(Xte))

    def aug(x, y):
        x = tf.image.random_brightness(x, 30.0)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        x = tf.image.random_flip_left_right(x)
        return tf.clip_by_value(x, 0.0, 255.0), y

    ds_tr = (tf.data.Dataset.from_tensor_slices((Xtr.astype("float32"), ytr)).shuffle(4096)
             .map(aug, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(2))
    ds_va = tf.data.Dataset.from_tensor_slices((Xva.astype("float32"), yva)).batch(a.batch)

    inp = layers.Input((SIZE, SIZE, 1))
    x = layers.Rescaling(1 / 255.0)(inp)               # model takes raw 0..255 pixels
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)     # P(closed)
    model = models.Model(inp, out)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(ds_tr, validation_data=ds_va, epochs=a.epochs,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

    pred = (model.predict(Xte.astype("float32"), batch_size=256).ravel() >= 0.5).astype(int)
    print("\nTEST (unseen subjects)\n", classification_report(yte, pred, target_names=["open", "closed"]))
    print("confusion matrix [rows=true open/closed]:\n", confusion_matrix(yte, pred))

    os.makedirs(a.out, exist_ok=True)
    model.save(os.path.join(a.out, "eye_cnn.keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(os.path.join(a.out, "eye_cnn.tflite"), "wb") as f:
        f.write(conv.convert())
    print("saved", os.path.join(a.out, "eye_cnn.tflite"))


if __name__ == "__main__":
    main()
