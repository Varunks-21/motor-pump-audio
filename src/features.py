"""
features.py
===========
Shared feature layer used IDENTICALLY by training, testing and the live UI.

Two responsibilities:
  1. Audio  ->  log-mel spectrogram  (the input both models consume)
  2. Annotation timeline  <->  YOHO per-bin targets  (encode / decode)

The encode/decode logic is pure numpy so it is fully unit-testable without torch
or librosa. `logmel` / `load_audio` use librosa at call time (lazy import) so this
module imports cleanly even before librosa is installed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

_EPS = 1e-10


# ===========================================================================
# Audio -> log-mel spectrogram
# ===========================================================================
def load_audio(path, sr=config.SAMPLE_RATE, fix_to=None):
    """Load a mono waveform at `sr`. Optionally pad/trim to `fix_to` samples."""
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if fix_to is not None:
        y = fix_length(y, fix_to)
    return y.astype(np.float32)


def fix_length(y, n):
    if len(y) == n:
        return y
    if len(y) > n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def logmel(y):
    """
    Log-mel spectrogram, shape (N_MELS, T).

    Uses an ABSOLUTE power reference (not per-clip max) so that quiet clips stay
    quiet -- this is what lets the model tell `silence` from a running pump.
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=np.asarray(y, dtype=np.float32),
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        win_length=config.WIN_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0,
    )
    return (10.0 * np.log10(mel + _EPS)).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisation (stats computed once on the training set, saved to config.json)
# ---------------------------------------------------------------------------
def compute_norm_stats(feat_list):
    """Per-mel-bin mean and std across a list of (N_MELS, T) features."""
    allcols = np.concatenate([f for f in feat_list], axis=1)  # (N_MELS, sum_T)
    mean = allcols.mean(axis=1, keepdims=True)                # (N_MELS, 1)
    std = allcols.std(axis=1, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(feat, mean, std):
    """Standardise a (N_MELS, T) feature with per-bin mean/std."""
    mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(-1, 1)
    return ((feat - mean) / std).astype(np.float32)


# ===========================================================================
# YOHO targets:  events  <->  (N_BINS, NUM_CLASSES, 3) array of (presence,start,end)
# ===========================================================================
def events_to_targets(events, n_bins=config.N_BINS, bin_dur=config.BIN_DUR):
    """
    Encode a list of events into YOHO targets.

    Each event is a dict with keys: event_class, start_time, end_time (seconds).
    For every bin a class overlaps, presence=1 and (start,end) are the normalised
    onset/offset of the overlap *within that bin* (0..1).
    """
    tgt = np.zeros((n_bins, config.NUM_CLASSES, 3), dtype=np.float32)
    for ev in events:
        c = config.CLASS_TO_IDX[ev["event_class"]]
        s, e = float(ev["start_time"]), float(ev["end_time"])
        for b in range(n_bins):
            b0 = b * bin_dur
            b1 = b0 + bin_dur
            ov0, ov1 = max(s, b0), min(e, b1)
            if ov1 > ov0:                       # event overlaps this bin
                tgt[b, c, 0] = 1.0
                tgt[b, c, 1] = (ov0 - b0) / bin_dur
                tgt[b, c, 2] = (ov1 - b0) / bin_dur
    return tgt


def decode_events(pred, threshold=config.PRESENCE_THRESHOLD,
                  n_bins=config.N_BINS, bin_dur=config.BIN_DUR, min_dur=0.15):
    """
    Decode YOHO predictions into a list of events (per-class, multi-label).

    `pred` is (N_BINS, NUM_CLASSES, 3) with sigmoid-activated values. Consecutive
    active bins of the same class are merged; the merged event's edges use the
    regressed start of the first bin and regressed end of the last bin.
    """
    events = []
    pred = np.asarray(pred)
    for c in range(config.NUM_CLASSES):
        active = pred[:, c, 0] > threshold
        b = 0
        while b < n_bins:
            if not active[b]:
                b += 1
                continue
            b0 = b
            while b < n_bins and active[b]:
                b += 1
            b1 = b - 1
            start = b0 * bin_dur + float(pred[b0, c, 1]) * bin_dur
            end = b1 * bin_dur + float(pred[b1, c, 2]) * bin_dur
            start = max(0.0, start)
            end = min(n_bins * bin_dur, end)
            if end - start >= min_dur:
                events.append({
                    "event_class": config.IDX_TO_CLASS[c],
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                })
    events.sort(key=lambda r: r["start_time"])
    return events


def dominant_timeline(pred, n_bins=config.N_BINS, bin_dur=config.BIN_DUR):
    """
    Per-bin top class (by presence), merged into contiguous segments.
    Returns list of dicts {event_class, start_time, end_time} that tile [0, clip].
    Used by the spectrogram visualiser to colour each region.
    """
    pred = np.asarray(pred)
    top = np.argmax(pred[:, :, 0], axis=1)      # (N_BINS,)
    segments = []
    b = 0
    while b < n_bins:
        c = top[b]
        b0 = b
        while b < n_bins and top[b] == c:
            b += 1
        segments.append({
            "event_class": config.IDX_TO_CLASS[int(c)],
            "start_time": round(b0 * bin_dur, 3),
            "end_time": round(b * bin_dur, 3),
        })
    return segments


def bin_edges(n_bins=config.N_BINS, bin_dur=config.BIN_DUR):
    """Return the [start, end] time of every bin (for plotting/inspection)."""
    return [(round(b * bin_dur, 3), round((b + 1) * bin_dur, 3)) for b in range(n_bins)]