"""Shared landmark preprocessing for the ISL translator (TWO-HAND version).

SINGLE source of truth for turning MediaPipe hand landmarks into the feature
vector the model expects. Imported by keypoint generation, training, and live
prediction so the transform is identical everywhere. Do not duplicate it.

Many ISL signs use two hands, so we support up to two:
  21 landmarks x 2 coords x 2 hands = 84 features.

Pipeline:
  1. For each detected hand, take its 21 landmarks as PIXEL (x, y) pairs
     (normalized MediaPipe coords * image width/height). Pixel coords keep the
     true hand proportions regardless of image aspect ratio (square training
     images vs a 4:3 webcam) - see the aspect-ratio note below.
  2. Order the hands left-to-right by wrist (landmark 0) x, so the layout is
     deterministic.
  3. Reference ALL points to the MEAN of the present wrists (a single shared
     origin). Sharing one origin preserves the geometry BETWEEN the two hands,
     which is what distinguishes many two-handed signs. For one hand this is
     just that hand's wrist.
  4. Flatten to 84: [hand0 (42), hand1 (42)]. A missing second hand -> 42 zeros,
     so genuinely one-handed signs (numbers, etc.) still work.
  5. Divide everything by the largest absolute value (scale normalization).

Aspect ratio: MediaPipe returns x in [0,1] (÷width) and y in [0,1] (÷height).
Using those directly squashes x vs y on a non-square image. Multiplying by
(width, height) first gives aspect-correct PIXEL coords; max-abs then removes
overall scale.
"""
import numpy as np

NUM_HANDS = 2
NUM_LANDMARKS = 21
PER_HAND = NUM_LANDMARKS * 2            # 42
NUM_FEATURES = NUM_HANDS * PER_HAND    # 84


def landmarks_to_xy(hand_landmarks, width, height):
    """21 (x, y) PIXEL pairs from one MediaPipe hand_landmarks object.

    Multiplying normalized coords by (width, height) makes them aspect-ratio
    correct. z is ignored; preprocess_hands() re-normalizes the scale.
    """
    return [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]


def preprocess_hands(hands_xy):
    """Convert 1 or 2 hands of pixel (x, y) landmarks into the 84-feature vector.

    Args:
        hands_xy: list of up to NUM_HANDS items, each a sequence of 21 (x, y)
            pixel pairs. May be empty.

    Returns:
        np.ndarray shape (84,), float32. All zeros if no hands given.
    """
    hands = [np.asarray(h, dtype=np.float32) for h in hands_xy if h is not None]
    hands = [h for h in hands if h.shape == (NUM_LANDMARKS, 2)]
    if not hands:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    hands = hands[:NUM_HANDS]
    # 2. Order left-to-right by wrist x.
    hands = sorted(hands, key=lambda h: float(h[0][0]))

    # 3. Shared reference = mean of present wrists.
    ref = np.mean([h[0] for h in hands], axis=0)

    # 4. Flatten per hand; zero-pad a missing hand.
    blocks = [(h - ref).flatten() for h in hands]
    while len(blocks) < NUM_HANDS:
        blocks.append(np.zeros(PER_HAND, dtype=np.float32))
    flat = np.concatenate(blocks).astype(np.float32)

    # 5. Scale-normalize by largest absolute value.
    max_abs = np.max(np.abs(flat))
    if max_abs > 0:
        flat = flat / max_abs
    return flat.astype(np.float32)


# Column names for the keypoint CSV: label, f1 .. f84.
FEATURE_COLUMNS = [f"f{i}" for i in range(1, NUM_FEATURES + 1)]
CSV_COLUMNS = ["label"] + FEATURE_COLUMNS
