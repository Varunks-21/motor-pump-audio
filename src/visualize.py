"""
visualize.py
============
Render a mel spectrogram with the model's predicted class regions highlighted.

Produces, per test clip:
  - mel spectrogram background (dB)
  - translucent colour-coded overlays for each predicted class interval, labelled
  - a timeline strip comparing GROUND TRUTH vs PREDICTED class over time
  - time axis in seconds, frequency (mel) axis in kHz, and a class-colour legend

The core plotting function takes a mel array + segment lists, so it has no librosa
dependency (matplotlib only). `visualize_file` is the convenience wrapper that
computes the spectrogram from an audio file via features.py.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless-safe PNG rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Mel-frequency helpers (pure numpy; used only for y-axis tick labels)
# ---------------------------------------------------------------------------
def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_frequencies(n_mels, fmin, fmax):
    m = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels)
    return _mel_to_hz(m)


def _color(cls):
    return config.CLASS_COLORS.get(cls, "#888888")


# ---------------------------------------------------------------------------
# Core plot
# ---------------------------------------------------------------------------
def plot_prediction(mel_db, pred_segments, duration, out_path,
                    gt_segments=None, title=None, ae_score=None, ae_flag=None):
    """
    mel_db        : (N_MELS, T) spectrogram in dB (for display)
    pred_segments : list of {event_class, start_time, end_time} (predicted)
    gt_segments   : optional list of the same shape (ground truth)
    """
    n_mels = mel_db.shape[0]
    fig, (ax_spec, ax_tl) = plt.subplots(
        2, 1, figsize=(12, 6), height_ratios=[4, 1],
        sharex=True, gridspec_kw={"hspace": 0.08})

    # --- spectrogram background ---
    ax_spec.imshow(mel_db, aspect="auto", origin="lower",
                   extent=[0.0, duration, 0.0, n_mels], cmap="magma")

    # frequency (mel) axis labelled in kHz
    mel_freqs = _mel_frequencies(n_mels, config.FMIN, config.FMAX)
    tick_idx = np.linspace(0, n_mels - 1, 6).astype(int)
    ax_spec.set_yticks(tick_idx)
    ax_spec.set_yticklabels([f"{mel_freqs[i] / 1000:.1f}" for i in tick_idx])
    ax_spec.set_ylabel("Frequency (kHz)")

    # --- predicted overlays on the spectrogram ---
    present = []
    for s in pred_segments:
        ax_spec.axvspan(s["start_time"], s["end_time"],
                        color=_color(s["event_class"]), alpha=0.30, lw=0)
        ax_spec.axvline(s["start_time"], color="white", lw=0.6, alpha=0.5)
        mid = 0.5 * (s["start_time"] + s["end_time"])
        ax_spec.text(mid, n_mels * 0.92, s["event_class"],
                     ha="center", va="top", fontsize=8, color="white",
                     bbox=dict(boxstyle="round,pad=0.2",
                               fc=_color(s["event_class"]), ec="none", alpha=0.85))
        present.append(s["event_class"])

    ttl = title or "Predicted motor-pump events"
    if ae_score is not None:
        state = "ANOMALY" if ae_flag else "normal"
        ttl += f"   |   AE score {ae_score:.3f} ({state})"
    ax_spec.set_title(ttl)

    # --- timeline strip: GT (top) vs Pred (bottom) ---
    def _strip(ax, segments, y, h):
        for s in segments:
            ax.broken_barh([(s["start_time"], s["end_time"] - s["start_time"])],
                           (y, h), facecolors=_color(s["event_class"]))

    rows = []
    if gt_segments is not None:
        _strip(ax_tl, gt_segments, 1.1, 0.8)
        rows.append((1.5, "Ground truth"))
        for s in gt_segments:
            present.append(s["event_class"])
    _strip(ax_tl, pred_segments, 0.1, 0.8)
    rows.append((0.5, "Predicted"))

    ax_tl.set_ylim(0, 2.0 if gt_segments is not None else 1.0)
    ax_tl.set_yticks([r[0] for r in rows])
    ax_tl.set_yticklabels([r[1] for r in rows])
    ax_tl.set_xlim(0, duration)
    ax_tl.set_xlabel("Time (s)")

    # --- legend (classes actually shown) ---
    seen = [c for c in config.CLASSES if c in set(present)]
    handles = [mpatches.Patch(color=_color(c), label=c) for c in seen]
    if handles:
        fig.legend(handles=handles, loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_file(audio_path, pred_segments, out_path,
                   gt_segments=None, ae_score=None, ae_flag=None):
    """Compute the spectrogram from an audio file, then render the prediction."""
    import features as F  # lazy (pulls librosa)
    y = F.load_audio(audio_path, fix_to=config.CLIP_SAMPLES)
    mel_db = F.logmel(y)
    duration = len(y) / config.SAMPLE_RATE
    title = f"{os.path.basename(audio_path)}"
    return plot_prediction(mel_db, pred_segments, duration, out_path,
                           gt_segments=gt_segments, title=title,
                           ae_score=ae_score, ae_flag=ae_flag)