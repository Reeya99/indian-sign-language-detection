# Indian Sign Language (ISL) Real-Time Translator

A deep-learning-based computer vision application that recognizes Indian Sign Language (ISL) gestures in real time using a webcam and translates them into text.

## Features
- **Real-Time Landmark Detection**: Uses Google MediaPipe to extract hand landmarks at high frame rates.
- **One-Handed & Two-Handed Sign Recognition**: Intelligent gating to handle both single-hand and dual-hand signs.
- **Flask Web Interface**: Clean web UI with live video overlay and prediction smoothing.
- **Trained Deep Learning Model**: Accurate classification across 35 classes (numbers 1-9 and letters A-Z).

## Project Structure
```text
├── app.py                         # Flask web application & live prediction server
├── config.py                      # Central configuration & paths
├── preprocessing.py               # Hand landmarks normalization & feature extraction
├── train_model.py                 # Neural network architecture & training pipeline
├── dataset_keypoint_generation.py # Keypoint extraction script
├── classes.json                   # Class labels
├── hand_counts.json               # Required hand count per sign
├── model.h5                       # Trained Keras model
├── requirements.txt               # Dependencies
├── static/                        # CSS and JavaScript
├── templates/                     # HTML templates
└── reports/                       # Training & evaluation metrics
```

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd cv_isl_project
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```powershell
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5050`.

