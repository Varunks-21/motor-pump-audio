"""
Central configuration for the Motor Pump Audio Event Detection project (Phase 1).

Every stage -- dataset generation, feature extraction, training, testing and the
live web UI -- imports its settings from this single file so they can never drift
out of sync. If you want to change the sample rate, the mel settings or the list
of classes, change it HERE and nowhere else.
"""
import os

# ---------------------------------------------------------------------------
# Paths (all relative to this file, so the project runs from any directory)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")            # per-class single-event clips
MIXED_DIR = os.path.join(DATA_DIR, "mixed")        # mixed multi-event test clips
HISTORY_DB = os.path.join(DATA_DIR, "history.db")  # live prediction log (SQLite)

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# trained-model artefacts
YOHO_MODEL_PATH = os.path.join(MODEL_DIR, "yoho_best.pt")
AE_MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_best.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
RUN_CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

METADATA_CSV = os.path.join(RAW_DIR, "metadata.csv")

# ---------------------------------------------------------------------------
# Audio settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000           # Hz - motor-pump signatures sit well below 8 kHz
CLIP_DURATION = 10.0          # seconds - every dataset clip is exactly this long
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# ---------------------------------------------------------------------------
# Mel spectrogram - the SHARED feature representation for BOTH models
# ---------------------------------------------------------------------------
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 64
FMIN = 20
FMAX = SAMPLE_RATE // 2       # 8000 Hz
# number of spectrogram frames produced for a full 10 s clip
FRAMES_PER_CLIP = 1 + CLIP_SAMPLES // HOP_LENGTH

# ---------------------------------------------------------------------------
# YOHO time grid
# ---------------------------------------------------------------------------
N_BINS = 20                          # 20 bins over 10 s
BIN_DUR = CLIP_DURATION / N_BINS     # 0.5 s per bin
PRESENCE_THRESHOLD = 0.5             # presence prob above which a class is "active"

# ---------------------------------------------------------------------------
# Classes
# "background_noise" removed.
# "motor_off" replaces silence (no motor running at all).
# "unknown_sound" is a catch-all for unrecognised audio -- not trained explicitly;
#   it is assigned at inference time when the autoencoder flags an anomaly AND
#   YOHO confidence is low (no strong known-class match).
# ---------------------------------------------------------------------------
CLASSES = [
    "motor_off",          # replaces old "silence" -- no motor running
    "normal",
    "bearing_fault",
    "cavitation",
    "impeller_damage",
    "pipe_leak",
    "motor_overload",
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# "unknown_sound" is NOT a trained class; it's a display-only label that
# live_app.py assigns when neither YOHO nor the AE recognises the audio.
UNKNOWN_CLASS = "unknown_sound"

# classes treated as abnormal -> RED warning in the live UI
ANOMALY_CLASSES = [
    "bearing_fault",
    "cavitation",
    "impeller_damage",
    "pipe_leak",
    "motor_overload",
]
NORMAL_CLASSES = ["normal"]                   # healthy operation -> GREEN
NEUTRAL_CLASSES = ["motor_off"]               # motor off -> GREY/YELLOW

# a fixed colour per class (includes the synthetic unknown_sound label)
CLASS_COLORS = {
    "motor_off":        "#9e9e9e",
    "normal":           "#2e7d32",
    "bearing_fault":    "#c62828",
    "cavitation":       "#ad1457",
    "impeller_damage":  "#6a1b9a",
    "pipe_leak":        "#1565c0",
    "motor_overload":   "#e65100",
    "unknown_sound":    "#ff6f00",   # amber -- unrecognised
}

# Confidence below which a YOHO prediction is downgraded
LOW_CONF_THRESHOLD = 0.40

# AE score multiplier above the stored threshold that triggers "unknown_sound"
# rather than the YOHO class. Set to 1.0 to use the raw threshold.
UNKNOWN_AE_MULTIPLIER = 1.0


def status_for_class(class_name):
    """Map a predicted class to the traffic-light status used by the web UI."""
    if class_name in ANOMALY_CLASSES:
        return "red"
    if class_name == UNKNOWN_CLASS:
        return "red"          # unknown also triggers red -- needs investigation
    if class_name in NORMAL_CLASSES:
        return "green"
    return "yellow"           # motor_off -> yellow


def is_anomaly(class_name):
    """True if the class is one of the abnormal motor-pump fault states."""
    return class_name in ANOMALY_CLASSES or class_name == UNKNOWN_CLASS


# ---------------------------------------------------------------------------
# Dataset generation defaults
# ---------------------------------------------------------------------------
SAMPLES_PER_CLASS = 300       # increased from 250 for better class separation
VAL_SPLIT = 0.15
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create every project directory if it does not already exist."""
    for d in [DATA_DIR, RAW_DIR, MIXED_DIR, MODEL_DIR,
              OUTPUT_DIR, PRED_DIR, VIZ_DIR]:
        os.makedirs(d, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)
