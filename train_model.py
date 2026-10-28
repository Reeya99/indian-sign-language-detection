"""Milestone 2 (+ STEP A) - Train / retrain the ISL classifier.

Data:
  - keypoint.csv            : base dataset (Milestone 1).
  - my_keypoints.csv        : your own recorded samples (optional). Combined
                              automatically if present, and weighted higher.

Accuracy improvements over the first version:
  - Mirror augmentation: each TRAIN row gets a left/right-mirrored copy
    (negate the x features, which are wrist-relative), so both hands work.
  - Light augmentation: small random rotation (+/-10 deg), scale (+/-10%),
    and tiny noise on TRAIN rows, so the model stops memorizing the near-
    duplicate dataset frames. Augmented rows are re-normalized by max-abs to
    stay identical in form to what preprocessing.py produces at inference.
  - Mirror + augmentation are applied to the TRAIN split only; val/test stay
    original.

Run / retrain (venv active):
    python train_model.py
    python train_model.py --my-boost 8 --aug-copies 3   # optional knobs
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import config
from preprocessing import FEATURE_COLUMNS, NUM_FEATURES, PER_HAND

SEED = 42
EPOCHS = 40
BATCH_SIZE = 64


def set_seeds():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def _read_csv(path, source_flag):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = df["label"].astype(str).to_numpy()
    src = np.full(len(df), source_flag, dtype=np.int8)
    return X, labels, src


def load_data():
    if not os.path.isfile(config.KEYPOINT_CSV):
        raise SystemExit(
            f"{config.KEYPOINT_CSV} not found. Run "
            "dataset_keypoint_generation.py first (Milestone 1)."
        )
    X, labels, src = _read_csv(config.KEYPOINT_CSV, 0)
    n_mine = 0
    if os.path.isfile(config.MY_KEYPOINT_CSV):
        Xm, lm, sm = _read_csv(config.MY_KEYPOINT_CSV, 1)
        X = np.concatenate([X, Xm])
        labels = np.concatenate([labels, lm])
        src = np.concatenate([src, sm])
        n_mine = len(Xm)

    classes = sorted(np.unique(labels).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[l] for l in labels], dtype=np.int64)
    return X, y, src, classes, n_mine


# ---- Augmentation --------------------------------------------------------
NUM_HANDS = 2
NUM_LANDMARKS = 21


def mirror_x(X):
    """Left/right mirror: negate x, and swap the two hand blocks.

    Negating x reflects the whole configuration; the hands then swap which is
    leftmost, so (where a real second hand exists) we swap the two 42-blocks to
    keep the "leftmost hand first" convention. One-handed rows (second block all
    zeros) keep their hand in block 0. Max-abs is unchanged by negation.
    """
    pts = X.reshape(-1, NUM_HANDS, NUM_LANDMARKS, 2).copy()
    pts[..., 0] *= -1.0
    present1 = np.any(pts[:, 1] != 0, axis=(1, 2))   # real second hand present
    pts[present1] = pts[present1][:, ::-1]           # swap hand0 <-> hand1
    return pts.reshape(-1, NUM_FEATURES)


def augment(X, rng, rot_deg=10.0, scale_pct=0.10, noise=0.01):
    """Random rotation about the shared origin, scale, and noise; re-normalize.

    Rotates/scales both hands together (shared origin), so inter-hand geometry
    is preserved. Zero-padded (absent) hands are kept zero - we never add noise
    to a hand that isn't there.
    """
    n = X.shape[0]
    pts = X.reshape(n, NUM_HANDS, NUM_LANDMARKS, 2).astype(np.float32).copy()

    # Remember which hand-blocks are absent (all zero) per sample.
    absent = ~np.any(pts != 0, axis=(2, 3))          # (n, NUM_HANDS) bool

    ang = np.radians(rng.uniform(-rot_deg, rot_deg, n)).astype(np.float32)
    cos, sin = np.cos(ang)[:, None, None], np.sin(ang)[:, None, None]
    x, y = pts[..., 0].copy(), pts[..., 1].copy()
    pts[..., 0] = x * cos - y * sin
    pts[..., 1] = x * sin + y * cos

    scale = rng.uniform(1 - scale_pct, 1 + scale_pct, n)[:, None, None, None]
    pts *= scale.astype(np.float32)
    pts += rng.normal(0.0, noise, pts.shape).astype(np.float32)

    pts[absent] = 0.0                                 # restore absent hands

    flat = pts.reshape(n, NUM_FEATURES)
    max_abs = np.max(np.abs(flat), axis=1, keepdims=True)
    max_abs[max_abs == 0] = 1.0
    return (flat / max_abs).astype(np.float32)


def build_training_set(X, y, src, aug_copies, rng):
    """Original + mirror, then `aug_copies` augmented copies of that set."""
    base_X = np.concatenate([X, mirror_x(X)])
    base_y = np.concatenate([y, y])
    base_src = np.concatenate([src, src])

    parts_X, parts_y, parts_src = [base_X], [base_y], [base_src]
    for _ in range(aug_copies):
        parts_X.append(augment(base_X, rng))
        parts_y.append(base_y)
        parts_src.append(base_src)

    return (
        np.concatenate(parts_X),
        np.concatenate(parts_y),
        np.concatenate(parts_src),
    )


def compute_hand_counts(X, y, classes):
    """Required hand count (1 or 2) per class, from the training data.

    A row has a real second hand when its second 42-feature block is non-zero.
    If the majority of a class's rows use two hands, that class needs two.
    """
    has_two = np.any(X[:, PER_HAND:] != 0, axis=1)
    counts = {}
    for i, c in enumerate(classes):
        mask = (y == i)
        frac = float(has_two[mask].mean()) if mask.any() else 0.0
        counts[c] = 2 if frac >= 0.5 else 1
    return counts


def filter_to_canonical_hands(X, y, src, classes):
    """Keep only rows whose hand count matches the class's canonical count.

    Two-handed classes (e.g. B) keep only their two-hand rows, dropping the
    occlusion rows where MediaPipe found one hand (those overlap with genuine
    one-handed signs and blur the boundary). One-handed classes keep only their
    one-hand rows, dropping spurious two-hand detections. This makes each
    class's feature distribution consistent -> fewer wrong predictions.
    """
    has_two = np.any(X[:, PER_HAND:] != 0, axis=1)
    counts = compute_hand_counts(X, y, classes)
    keep = np.zeros(len(X), dtype=bool)
    print("Cleaning data to canonical hand counts:")
    for i, c in enumerate(classes):
        mask = (y == i)
        canon_two = counts[c] == 2
        row_keep = mask & (has_two == canon_two)
        dropped = int(mask.sum() - row_keep.sum())
        keep |= row_keep
        if dropped:
            print(f"  {c} [{counts[c]}-hand]: kept {int(row_keep.sum())}, "
                  f"dropped {dropped} mismatched rows")
    return X[keep], y[keep], src[keep], counts


def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel="True label", xlabel="Predicted label",
        title="ISL confusion matrix (test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=90)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confused_pairs(cm, classes, top_n=15):
    pairs = [
        (classes[i], classes[j], int(cm[i, j]))
        for i in range(len(classes)) for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my-boost", type=float, default=5.0,
                    help="sample-weight multiplier for your recorded samples")
    ap.add_argument("--aug-copies", type=int, default=2,
                    help="augmented copies of the (orig+mirror) training set")
    args = ap.parse_args()

    set_seeds()
    rng = np.random.default_rng(SEED)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    X, y, src, classes, n_mine = load_data()
    num_classes = len(classes)
    print(f"Loaded {len(X)} samples ({len(X) - n_mine} base + {n_mine} yours), "
          f"{num_classes} classes.")

    # NOTE: filter_to_canonical_hands() (dropping rows whose hand count differs
    # from the class's canonical count) was tested and REDUCED real accuracy
    # 96.9% -> 87.3% - it removed the one-hand occlusion frames that teach mixed
    # signs (P, S, F, Z...). We keep all rows; the one/two-hand requirement is
    # enforced at inference by the reject-on-mismatch gate in app.py instead.

    # Stratified split (carry src alongside). Val/test stay original.
    Xtrf, Xtest, ytrf, ytest, strf, _ = train_test_split(
        X, y, src, test_size=0.20, stratify=y, random_state=SEED)
    Xtr, Xval, ytr, yval, strn, _ = train_test_split(
        Xtrf, ytrf, strf, test_size=0.10, stratify=ytrf, random_state=SEED)
    print(f"Train: {len(Xtr)}  Val: {len(Xval)}  Test: {len(Xtest)}")

    # Balanced class weights from the original train labels.
    cls_w = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=ytr)

    # Augment the training split.
    Xtr_a, ytr_a, strn_a = build_training_set(
        Xtr, ytr, strn, args.aug_copies, rng)
    print(f"Augmented train: {len(Xtr_a)} rows "
          f"(orig+mirror x {1 + args.aug_copies})")

    # Per-sample weights: balanced class weight * boost for your samples.
    sw = np.array([cls_w[c] for c in ytr_a], dtype=np.float32)
    sw[strn_a == 1] *= args.my_boost
    if n_mine:
        print(f"Your samples weighted x{args.my_boost}.")

    model = build_model(num_classes)
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, min_lr=1e-5),
    ]
    model.fit(
        Xtr_a, ytr_a,
        validation_data=(Xval, yval),
        sample_weight=sw,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cbs, verbose=2,
    )

    # ---- Evaluate on original (non-augmented) test set ------------------
    y_pred = np.argmax(model.predict(Xtest, verbose=0), axis=1)
    test_acc = accuracy_score(ytest, y_pred)
    print(f"\n==== TEST ACCURACY: {test_acc * 100:.2f}% ====\n")

    report_txt = classification_report(
        ytest, y_pred, target_names=classes, digits=3)
    print(report_txt)

    report_path = os.path.join(config.REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report_txt)

    cm = confusion_matrix(ytest, y_pred, labels=np.arange(num_classes))
    cm_path = os.path.join(config.REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path)

    report_dict = classification_report(
        ytest, y_pred, target_names=classes, output_dict=True)
    recalls = sorted(((c, report_dict[c]["recall"]) for c in classes),
                     key=lambda x: x[1])
    print("Worst 8 classes by recall:")
    for c, r in recalls[:8]:
        print(f"  {c}: recall {r:.3f}")

    print("\nTop confused pairs (true -> predicted, count):")
    for true_c, pred_c, cnt in top_confused_pairs(cm, classes):
        print(f"  {true_c} -> {pred_c}: {cnt}")

    # Required hand count per class (for live hand-count gating).
    hand_counts = compute_hand_counts(X, y, classes)
    two = [c for c in classes if hand_counts[c] == 2]
    one = [c for c in classes if hand_counts[c] == 1]
    print(f"\nTwo-handed classes: {', '.join(two)}")
    print(f"One-handed classes: {', '.join(one)}")

    model.save(config.MODEL_PATH)
    # Write classes.json atomically so a concurrent reader (e.g. the running
    # Flask app) can never observe a half-written file.
    tmp = config.CLASSES_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(classes, f)
    os.replace(tmp, config.CLASSES_JSON)
    tmp = config.HAND_COUNTS_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hand_counts, f)
    os.replace(tmp, config.HAND_COUNTS_JSON)
    print(f"\nSaved model  -> {config.MODEL_PATH}")
    print(f"Saved classes-> {config.CLASSES_JSON}")
    print(f"Saved hands  -> {config.HAND_COUNTS_JSON}")
    print(f"Saved report -> {report_path}")
    print(f"Saved matrix -> {cm_path}")


if __name__ == "__main__":
    main()
