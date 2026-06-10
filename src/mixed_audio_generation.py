"""
mixed_audio_generation.py
=========================
Build mixed TEST clips with updated class list (no background_noise, uses motor_off).

Run:
    python src/mixed_audio_generation.py
    python src/mixed_audio_generation.py --num-files 20 --seed 3
"""
import os
import sys
import csv
import argparse

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)
import config                                   # noqa: E402
from dataset_generation import SYNTH, write_wav, _normalize_peak  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed scenarios (updated -- no background_noise, silence → motor_off)
# ---------------------------------------------------------------------------
FIXED_SCENARIOS = [
    [("normal", 3.0), ("cavitation", 2.0), ("motor_off", 2.0), ("bearing_fault", 3.0)],
    [("motor_off", 2.0), ("normal", 4.0), ("motor_overload", 4.0)],
    [("motor_off", 2.0), ("normal", 3.0), ("pipe_leak", 3.0), ("normal", 2.0)],
    [("normal", 2.5), ("impeller_damage", 3.0), ("normal", 2.0), ("cavitation", 2.5)],
    [("motor_off", 1.5), ("normal", 3.0), ("bearing_fault", 3.0), ("normal", 2.5)],
    [("normal", 4.0), ("motor_overload", 3.0), ("pipe_leak", 3.0)],
]


def _random_scenario(rng, total=None, min_dur=1.5):
    total = total or config.CLIP_DURATION
    n = int(rng.integers(3, 6))
    raw = rng.dirichlet(np.ones(n))
    durs = min_dur + raw * (total - n * min_dur)
    durs = np.round(durs, 1)
    durs[-1] = round(total - float(np.sum(durs[:-1])), 1)

    classes, prev = [], None
    for _ in range(n):
        choices = [c for c in config.CLASSES if c != prev]
        c = str(rng.choice(choices))
        classes.append(c)
        prev = c
    return list(zip(classes, [float(d) for d in durs]))


def _segment_audio(class_name, n_samples, rng):
    full = SYNTH[class_name](rng)
    if len(full) <= n_samples:
        seg = np.pad(full, (0, n_samples - len(full)))
    else:
        start = int(rng.integers(0, len(full) - n_samples))
        seg = full[start:start + n_samples]
    return seg.astype(np.float64)


def _fade(seg, fade_len):
    fade_len = min(fade_len, len(seg) // 2)
    if fade_len <= 0:
        return seg
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade_len)))
    seg[:fade_len] *= ramp
    seg[-fade_len:] *= ramp[::-1]
    return seg


def build_mixed(scenario, rng):
    sr = config.SAMPLE_RATE
    total_samples = config.CLIP_SAMPLES
    audio = np.zeros(total_samples, dtype=np.float64)
    fade_len = int(0.03 * sr)

    rows = []
    cursor = 0
    n_seg = len(scenario)
    for i, (cls, dur) in enumerate(scenario):
        if i == n_seg - 1:
            end = total_samples
        else:
            end = min(total_samples, cursor + int(round(dur * sr)))
        seg_len = end - cursor
        if seg_len <= 0:
            continue
        seg = _segment_audio(cls, seg_len, rng)
        seg = _fade(seg, fade_len)
        audio[cursor:end] += seg
        rows.append({
            "event_class": cls,
            "start_time": round(cursor / sr, 3),
            "end_time": round(end / sr, 3),
        })
        cursor = end
        if cursor >= total_samples:
            break

    audio = _normalize_peak(audio, 0.9)
    return audio.astype(np.float32), rows


def generate(num_files, seed):
    config.ensure_dirs()
    rng = np.random.default_rng(seed)

    scenarios = list(FIXED_SCENARIOS)
    while len(scenarios) < num_files:
        scenarios.append(_random_scenario(rng))
    scenarios = scenarios[:num_files]

    master_rows = []
    for idx, scenario in enumerate(scenarios):
        fname = f"mix_{idx:03d}.wav"
        fpath = os.path.join(config.MIXED_DIR, fname)
        audio, rows = build_mixed(scenario, rng)
        write_wav(fpath, audio, config.SAMPLE_RATE)

        per_file = os.path.join(config.MIXED_DIR, f"mix_{idx:03d}.csv")
        with open(per_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "event_class", "start_time", "end_time"])
            w.writeheader()
            for r in rows:
                w.writerow({"filename": fname, **r})
                master_rows.append({"filename": fname, **r})

        timeline = "  ".join(f"{r['start_time']:.1f}-{r['end_time']:.1f}:{r['event_class']}"
                             for r in rows)
        print(f"  {fname}: {timeline}")

    master = os.path.join(config.MIXED_DIR, "annotations.csv")
    with open(master, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "event_class", "start_time", "end_time"])
        w.writeheader()
        w.writerows(master_rows)

    print(f"\nDone. {num_files} mixed clips, {len(master_rows)} events total.")
    print(f"Master annotation: {master}")


def main():
    ap = argparse.ArgumentParser(description="Generate mixed multi-event test audio.")
    ap.add_argument("--num-files", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    print(f"Generating {args.num_files} mixed test clips (seed={args.seed})...")
    generate(args.num_files, args.seed)


if __name__ == "__main__":
    main()
