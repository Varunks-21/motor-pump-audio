"""
test_model.py
=============
Evaluate the trained models on the mixed multi-event test clips.

For each data/mixed/mix_XXX.wav this:
  - runs YOHO -> decodes the predicted event timeline (class + start/end)
  - runs the autoencoder -> anomaly score + flag (score > learned threshold)
  - compares the predicted timeline against the ground-truth annotation
    (frame-level class accuracy + anomaly accuracy)
  - prints the predicted timeline
  - saves per-file results to outputs/predictions/<file>.json
  - saves a highlighted mel spectrogram to outputs/visualizations/<file>.png

Run (after training):
    python src/test_model.py
"""
import os
import sys
import csv
import json
import glob
import argparse

import numpy as np
import torch

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)
import config                              # noqa: E402
import features as F                       # noqa: E402
import visualize as V                      # noqa: E402
from models import YOHO, ConvAutoencoder   # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_run_config():
    if not os.path.exists(config.RUN_CONFIG_PATH):
        sys.exit("ERROR: models/config.json not found. Run `python src/train.py` first.")
    with open(config.RUN_CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["feat_mean"] = np.array(cfg["feat_mean"], dtype=np.float32).reshape(-1, 1)
    cfg["feat_std"] = np.array(cfg["feat_std"], dtype=np.float32).reshape(-1, 1)
    return cfg


def load_models(device):
    for p in (config.YOHO_MODEL_PATH, config.AE_MODEL_PATH):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run `python src/train.py` first.")
    yoho = YOHO().to(device)
    yoho.load_state_dict(torch.load(config.YOHO_MODEL_PATH, map_location=device))
    yoho.eval()
    ae = ConvAutoencoder().to(device)
    ae.load_state_dict(torch.load(config.AE_MODEL_PATH, map_location=device))
    ae.eval()
    return yoho, ae


def load_annotations():
    """Return {filename: [ {event_class,start_time,end_time}, ... ]}."""
    path = os.path.join(config.MIXED_DIR, "annotations.csv")
    if not os.path.exists(path):
        return {}
    gt = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            gt.setdefault(r["filename"], []).append({
                "event_class": r["event_class"],
                "start_time": float(r["start_time"]),
                "end_time": float(r["end_time"]),
            })
    return gt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _pred_class_at(pred, t, bin_dur):
    b = min(config.N_BINS - 1, max(0, int(t / bin_dur)))
    return config.IDX_TO_CLASS[int(np.argmax(pred[b, :, 0]))]


def _gt_class_at(gt_events, t):
    for e in gt_events:
        if e["start_time"] <= t < e["end_time"]:
            return e["event_class"]
    return None


def evaluate(pred, gt_events, duration, bin_dur, step=0.1):
    if not gt_events:
        return None
    ts = np.arange(step / 2, duration, step)
    n = correct = anom_correct = 0
    for t in ts:
        gt_c = _gt_class_at(gt_events, t)
        if gt_c is None:
            continue
        pr_c = _pred_class_at(pred, t, bin_dur)
        n += 1
        correct += int(gt_c == pr_c)
        anom_correct += int(config.is_anomaly(gt_c) == config.is_anomaly(pr_c))
    if n == 0:
        return None
    return {"frame_accuracy": correct / n, "anomaly_accuracy": anom_correct / n,
            "n_points": n}


def _fmt_timeline(segments):
    return " | ".join(f"{s['start_time']:.1f}-{s['end_time']:.1f} {s['event_class']}"
                      for s in segments)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Test trained models on mixed clips.")
    ap.add_argument("--mixed-dir", default=config.MIXED_DIR)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the presence threshold for YOHO decode")
    args = ap.parse_args()

    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_run_config()
    yoho, ae = load_models(device)
    annotations = load_annotations()

    mean, std = cfg["feat_mean"], cfg["feat_std"]
    bin_dur = cfg.get("bin_dur", config.BIN_DUR)
    ae_thr = cfg.get("ae_threshold", 0.0)
    pres_thr = args.threshold if args.threshold is not None else cfg.get(
        "presence_threshold", config.PRESENCE_THRESHOLD)

    wavs = sorted(glob.glob(os.path.join(args.mixed_dir, "*.wav")))
    if not wavs:
        sys.exit(f"No .wav files in {args.mixed_dir}. "
                 f"Run `python src/mixed_audio_generation.py` first.")

    print(f"Device: {device}   files: {len(wavs)}   AE threshold: {ae_thr:.4f}\n")
    summary = []

    for wav in wavs:
        fname = os.path.basename(wav)
        y = F.load_audio(wav, fix_to=config.CLIP_SAMPLES)
        duration = len(y) / config.SAMPLE_RATE
        raw = F.logmel(y)
        feat = F.normalize(raw, mean, std)

        pred = yoho.predict(feat)
        events = F.decode_events(pred, threshold=pres_thr)
        dom = F.dominant_timeline(pred)
        ae_score = ae.anomaly_score(feat)
        ae_flag = bool(ae_score > ae_thr)

        gt = annotations.get(fname, [])
        metrics = evaluate(pred, gt, duration, bin_dur)

        # ---- print ----
        head = fname
        if metrics:
            head += (f"   frame acc {metrics['frame_accuracy'] * 100:5.1f}%"
                     f"   anomaly acc {metrics['anomaly_accuracy'] * 100:5.1f}%")
        print(head)
        if gt:
            print("  GT  :", _fmt_timeline(gt))
        print("  Pred:", _fmt_timeline(dom))
        print(f"  AE score {ae_score:.4f} -> {'ANOMALY' if ae_flag else 'normal'}\n")

        # ---- save json ----
        result = {
            "filename": fname,
            "duration": round(duration, 3),
            "predicted_events": events,
            "dominant_timeline": dom,
            "ae_score": round(ae_score, 6),
            "ae_threshold": round(ae_thr, 6),
            "ae_flag": ae_flag,
            "ground_truth": gt,
            "metrics": metrics,
        }
        with open(os.path.join(config.PRED_DIR, fname.replace(".wav", ".json")), "w") as f:
            json.dump(result, f, indent=2)

        # ---- save visualization ----
        out_png = os.path.join(config.VIZ_DIR, fname.replace(".wav", ".png"))
        V.plot_prediction(raw, dom, duration, out_png,
                          gt_segments=gt or None, title=fname,
                          ae_score=ae_score, ae_flag=ae_flag)

        summary.append({
            "filename": fname,
            "frame_accuracy": round(metrics["frame_accuracy"], 4) if metrics else "",
            "anomaly_accuracy": round(metrics["anomaly_accuracy"], 4) if metrics else "",
            "n_pred_events": len(events),
            "n_gt_events": len(gt),
            "ae_score": round(ae_score, 6),
            "ae_flag": ae_flag,
        })

    # ---- summary ----
    with open(os.path.join(config.PRED_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    accs = [s["frame_accuracy"] for s in summary if s["frame_accuracy"] != ""]
    if accs:
        print(f"Mean frame accuracy across {len(accs)} files: "
              f"{100 * np.mean(accs):.1f}%")
    print(f"Predictions  -> {config.PRED_DIR}")
    print(f"Visualizations -> {config.VIZ_DIR}")


if __name__ == "__main__":
    main()