"""Shared project configuration for the ISL Real-Time Translator.

Central place for paths and constants used across keypoint generation,
training, and the live Flask app. Edit DATA_DIR here if you move the dataset.
"""
import os

# Project root = directory containing this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The Kaggle ISL dataset extracted to E:\cv_isl_project\Indian\<CLASS>\*.jpg
# (one extra nesting level vs. the originally-planned images/data/). Each class
# folder (1-9, A-Z) holds the images for that sign.
DATA_DIR = os.path.join(BASE_DIR, "Indian")

# Output artifacts.
KEYPOINT_CSV = os.path.join(BASE_DIR, "keypoint.csv")
# Samples you record from your own webcam via the website's "Collect samples"
# mode. Combined with keypoint.csv at training time and weighted higher.
MY_KEYPOINT_CSV = os.path.join(BASE_DIR, "my_keypoints.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
CLASSES_JSON = os.path.join(BASE_DIR, "classes.json")
# How many hands each class needs (1 or 2), derived from the training data.
# Used to gate live predictions: a two-handed sign is only recognized when two
# hands are visible, a one-handed sign only when one hand is visible.
HAND_COUNTS_JSON = os.path.join(BASE_DIR, "hand_counts.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Recognition settings.
CONFIDENCE_THRESHOLD = 0.60

# Whether frames are horizontally mirrored. The live webcam view is mirrored
# (selfie view), so we mirror training images the same way for consistency.
MIRROR_FRAMES = True
