"""Milestone 1 - Generate keypoint.csv from the ISL image dataset.

Reads every image under config.DATA_DIR (one subfolder per class), runs
MediaPipe Hands in static-image mode, applies the shared preprocessing, and
writes one row per detected hand to keypoint.csv as: label, f1 .. f42.

Mirroring: to match the live webcam view (selfie / mirrored), we flip each
image horizontally before detection when config.MIRROR_FRAMES is True. The
live app flips its frames the same way, so the features line up.

Run (venv active):
    python dataset_keypoint_generation.py
"""
import csv
import os

import cv2
import mediapipe as mp
from tqdm import tqdm

import config
from preprocessing import (
    CSV_COLUMNS,
    landmarks_to_xy,
    preprocess_hands,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_classes(data_dir):
    """Return sorted class folder names under data_dir."""
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def list_images(class_dir):
    """Return sorted image file paths in a class folder."""
    return sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )


def main():
    if not os.path.isdir(config.DATA_DIR):
        raise SystemExit(
            f"Dataset folder not found: {config.DATA_DIR}\n"
            "Fix DATA_DIR in config.py."
        )

    classes = list_classes(config.DATA_DIR)
    if not classes:
        raise SystemExit(f"No class folders found in {config.DATA_DIR}")

    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    print(f"Mirroring images horizontally: {config.MIRROR_FRAMES}")
    print(f"Writing: {config.KEYPOINT_CSV}\n")

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,               # two-handed ISL signs need both hands
        min_detection_confidence=0.5,
    )

    # Per-class tallies for the end-of-run report.
    stats = {}  # class -> [read, detected, skipped, two_hand]
    total_written = 0

    with open(config.KEYPOINT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)

        for label in classes:
            class_dir = os.path.join(config.DATA_DIR, label)
            images = list_images(class_dir)
            read = detected = skipped = two_hand = 0

            for img_path in tqdm(images, desc=f"Class {label:>2}", unit="img"):
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                read += 1

                if config.MIRROR_FRAMES:
                    img = cv2.flip(img, 1)

                # MediaPipe expects RGB.
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if not result.multi_hand_landmarks:
                    skipped += 1
                    continue

                hand_list = result.multi_hand_landmarks[:2]
                hands_xy = [landmarks_to_xy(hl, w, h) for hl in hand_list]
                features = preprocess_hands(hands_xy)
                writer.writerow([label] + features.tolist())
                detected += 1
                if len(hand_list) == 2:
                    two_hand += 1
                total_written += 1

            stats[label] = [read, detected, skipped, two_hand]

    hands.close()

    # ---- Report ----------------------------------------------------------
    print("\n" + "=" * 64)
    print(f"{'Class':>6} {'Read':>8} {'Detected':>10} {'Skipped':>9} "
          f"{'Skip%':>7} {'2-hand%':>9}")
    print("-" * 64)
    heavy_skip = []
    for label in classes:
        read, detected, skipped, two_hand = stats[label]
        attempted = detected + skipped
        skip_pct = (skipped / attempted * 100) if attempted else 0.0
        two_pct = (two_hand / detected * 100) if detected else 0.0
        print(f"{label:>6} {read:>8} {detected:>10} {skipped:>9} "
              f"{skip_pct:>6.1f}% {two_pct:>8.1f}%")
        if skip_pct >= 30.0:
            heavy_skip.append((label, skip_pct))

    print("=" * 64)
    print(f"Total rows written to keypoint.csv: {total_written}")

    if heavy_skip:
        print("\nWARNING - classes with heavy skips (>=30% no hand detected):")
        for label, pct in heavy_skip:
            print(f"  {label}: {pct:.1f}% skipped")
        print("Even with no hand detected on some frames, remaining frames "
              "(now with both hands when present) are used.")
    else:
        print("\nNo classes exceeded the 30% skip threshold.")


if __name__ == "__main__":
    main()
