"""
dataset_generation.py
=====================
Generate a synthetic motor-pump audio dataset.

Each clip is exactly CLIP_DURATION seconds long (config.py) and belongs to ONE of
the 7 classes. Every class has a physically-motivated synthetic signature designed
so each class is clearly distinguishable in the mel-spectrogram domain, and a
randomised augmentation pipeline (gain, noise, speed/pitch, time-shift) is applied
so no two clips are identical.

Classes
-------
  motor_off       -- no motor running: very low amplitude broadband floor only
  normal          -- healthy motor: shaft harmonics + blade pass, clean and periodic
  bearing_fault   -- periodic high-freq impulse bursts at defect frequency, modulated
  cavitation      -- random wideband crackle bursts (2–7 kHz) over the motor base
  impeller_damage -- strong 1× imbalance tone + per-revolution knock, wobble AM
  pipe_leak       -- sustained mid-high turbulent hiss (1–6 kHz), NOT the base motor
  motor_overload  -- boosted low-freq electrical harmonics (50/60 Hz) + slow swell

Key design decisions for separation
-------------------------------------
- pipe_leak and motor_overload now use ORTHOGONAL frequency bands so they cannot
  be confused. Pipe leak energy lives 1–6 kHz (turbulent hiss); motor overload
  energy is dominated by 50/100/150/200 Hz electrical lines and shaft harmonics.
- motor_off is near-silence (RMS < 0.01) – the model learns the energy floor.
- Each class has a characteristic that survives augmentation: impulse timing for
  bearing_fault, burst burstiness for cavitation, imbalance-tone level for
  impeller_damage, frequency band for pipe_leak, harmonic envelope for overload.

Run:
    python src/dataset_generation.py
    python src/dataset_generation.py --samples-per-class 300
"""
import os
import sys
import csv
import argparse

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# ===========================================================================
# Low-level signal helpers
# ===========================================================================
def _bandlimited_noise(low, high, n, sr, rng):
    low = max(1.0, float(low))
    high = min(float(high), sr / 2.0 - 1.0)
    white = rng.standard_normal(n)
    sos = sps.butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, white)


def _lowpass_noise(cut, n, sr, rng):
    white = rng.standard_normal(n)
    sos = sps.butter(4, min(cut, sr / 2.0 - 1.0), btype="low", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, white)


def _highpass_noise(cut, n, sr, rng):
    white = rng.standard_normal(n)
    sos = sps.butter(4, max(cut, 10.0), btype="high", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, white)


def _harmonic_tone(f0, n_harm, n, sr, rng, decay=0.7, jitter=0.0):
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k in range(1, n_harm + 1):
        fk = f0 * k * (1.0 + jitter * rng.standard_normal())
        if fk >= sr / 2.0:
            break
        phase = rng.uniform(0, 2 * np.pi)
        out += (decay ** (k - 1)) * np.sin(2 * np.pi * fk * t + phase)
    return out


def _amp_modulate(x, mod_hz, depth, sr, rng):
    t = np.arange(len(x)) / sr
    env = 1.0 + depth * np.sin(2 * np.pi * mod_hz * t + rng.uniform(0, 2 * np.pi))
    return x * env


def _impulse_train(rate_hz, n, sr, rng, res_freq=3500.0, decay_ms=5.0, jitter=0.04):
    """Periodic decaying clicks -- the bearing-fault signature."""
    out = np.zeros(n)
    period = sr / rate_hz
    click_len = int(sr * decay_ms / 1000.0)
    tt = np.arange(click_len) / sr
    env = np.exp(-tt / (decay_ms / 1000.0 / 3.0))
    click = env * np.sin(2 * np.pi * res_freq * tt)
    pos = rng.uniform(0, period)
    while pos < n:
        idx = int(pos)
        length = min(click_len, n - idx)
        out[idx:idx + length] += click[:length] * rng.uniform(0.8, 1.2)
        pos += period * (1.0 + jitter * rng.standard_normal())
    return out


def _normalize_peak(x, peak=0.9):
    m = np.max(np.abs(x))
    if m < 1e-9:
        return x
    return x / m * peak


# ===========================================================================
# Per-class synthesis
# ===========================================================================

def synth_motor_off(rng):
    """
    Motor is completely off: only a very faint broadband noise floor.
    RMS is kept < 0.005 so it is clearly separable from any running-motor class.
    """
    n = config.CLIP_SAMPLES
    floor = rng.uniform(0.001, 0.004)
    return rng.standard_normal(n) * floor


def _motor_base(rng, shaft_hz=None):
    """
    Healthy-motor core: shaft harmonics (25–30 Hz) + blade-pass tone + low-freq
    flow noise. Kept deliberately CLEAN so faults layer onto a consistent base.
    Returns (signal, shaft_hz) at ~0.75 peak amplitude.
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE
    if shaft_hz is None:
        shaft_hz = rng.uniform(24.0, 30.0)        # 1440–1800 rpm
    blades = int(rng.integers(5, 8))

    # Shaft harmonics (strong in 25–240 Hz range)
    body = _harmonic_tone(shaft_hz, 8, n, sr, rng, decay=0.65, jitter=0.001)
    # Blade-pass frequency (5–8× shaft)
    blade = _harmonic_tone(shaft_hz * blades, 3, n, sr, rng, decay=0.5) * 0.35
    # Mains hum (faint)
    line_f = float(rng.choice([50.0, 60.0]))
    line = _harmonic_tone(line_f, 2, n, sr, rng, decay=0.4) * 0.08
    # Low-freq flow noise (below 800 Hz)
    flow = _lowpass_noise(800, n, sr, rng) * 0.08

    sig = body + blade + line + flow
    # Gentle once-per-rev amplitude breathing
    sig = _amp_modulate(sig, shaft_hz, 0.04, sr, rng)
    return _normalize_peak(sig, 0.75), shaft_hz


def synth_normal(rng):
    """Healthy motor: clean shaft harmonics, minimal noise."""
    sig, _ = _motor_base(rng)
    return sig


def synth_bearing_fault(rng):
    """
    Periodic high-freq impulse bursts at defect frequency (3–6× shaft),
    load-modulated, ringing 2.5–4.5 kHz. Clearly separated from pipe_leak
    (which is sustained hiss) and motor_overload (which is low-freq).
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE
    base, shaft_hz = _motor_base(rng)
    defect_hz = shaft_hz * rng.uniform(3.0, 6.0)
    res = rng.uniform(2500, 4500)
    impacts = _impulse_train(defect_hz, n, sr, rng,
                             res_freq=res,
                             decay_ms=rng.uniform(4, 8))
    # Load modulation: impacts wax and wane at shaft rate
    impacts = _amp_modulate(impacts, shaft_hz, 0.45, sr, rng)
    return _normalize_peak(base * 0.7 + 1.1 * impacts, 0.90)


def synth_cavitation(rng):
    """
    Random wideband crackle bursts (collapsing bubbles: 2–7 kHz).
    Bursts are random in time (Poisson-like via low-pass envelope) which
    distinguishes them from the periodic impulses of bearing_fault.
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE
    base, _ = _motor_base(rng)
    hiss = _bandlimited_noise(2000, 7000, n, sr, rng)
    # Random burst envelope (slow modulation < 30 Hz clips to positive)
    burst_env = np.clip(_lowpass_noise(30, n, sr, rng), 0, None)
    burst_env = burst_env / (np.max(burst_env) + 1e-9)
    crackle = hiss * (0.4 + 1.6 * burst_env)
    return _normalize_peak(base * 0.6 + 0.9 * crackle, 0.90)


def synth_impeller_damage(rng):
    """
    Strong 1× shaft imbalance tone + once-per-rev knock (low-mid resonance
    800–1600 Hz) + amplitude wobble. The dominant signature is the large 1×
    shaft sine and the repetitive knock -- very different from pipe_leak hiss
    or the electrical signature of motor_overload.
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE
    base, shaft_hz = _motor_base(rng)
    t = np.arange(n) / sr

    # Large 1× imbalance sine (dominant feature)
    imbalance = 0.60 * np.sin(2 * np.pi * shaft_hz * t + rng.uniform(0, 2 * np.pi))

    # Once-per-rev mechanical knock (low-mid resonance, longer decay)
    knock = _impulse_train(shaft_hz, n, sr, rng,
                           res_freq=rng.uniform(800, 1600),
                           decay_ms=rng.uniform(10, 18))

    sig = base * 0.6 + imbalance + 0.55 * knock
    # Slow rotational wobble
    sig = _amp_modulate(sig, shaft_hz, 0.30, sr, rng)
    return _normalize_peak(sig, 0.90)


def synth_pipe_leak(rng):
    """
    Sustained turbulent broadband hiss (1–6 kHz) WITHOUT the normal motor base.
    This is the key change: pipe_leak is now a HISS-dominant signal in the
    mid-high band, clearly distinguishable from motor_overload (low-freq
    electrical) and normal (clean shaft harmonics).
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE

    # Turbulent hiss is the PRIMARY feature -- strong and sustained in 1–6 kHz
    hiss = _bandlimited_noise(1000, 6000, n, sr, rng)
    # Slow pressure fluctuation amplitude-modulates the hiss
    slow_mod = np.clip(_lowpass_noise(2.0, n, sr, rng), 0.5, None)
    slow_mod = slow_mod / np.max(slow_mod)
    hiss = hiss * (0.6 + 0.4 * slow_mod)

    # Optional very faint motor hum in the background (the pump is still running)
    if rng.random() < 0.6:
        motor_hum, _ = _motor_base(rng)
        hiss = hiss + motor_hum * rng.uniform(0.08, 0.18)

    return _normalize_peak(hiss, 0.90)


def synth_motor_overload(rng):
    """
    Boosted LOW-FREQUENCY electrical harmonics: mains frequency (50/60 Hz) and
    its harmonics are dominant, plus a slow overload swell and slip wobble.
    Frequency content is concentrated below ~500 Hz, the OPPOSITE of pipe_leak
    which is concentrated above 1 kHz. This is the main fix for the confusion.
    """
    n, sr = config.CLIP_SAMPLES, config.SAMPLE_RATE
    base, shaft_hz = _motor_base(rng)
    line = float(rng.choice([50.0, 60.0]))

    # Heavy electrical harmonics -- 2× line is the most prominent overload signal
    elec = _harmonic_tone(line, 6, n, sr, rng, decay=0.60) * 0.80
    elec += _harmonic_tone(2 * line, 4, n, sr, rng, decay=0.55) * 0.70
    elec += _harmonic_tone(3 * line, 3, n, sr, rng, decay=0.45) * 0.40

    # Low-freq saturation noise (motor core flux distortion: < 300 Hz)
    sat_noise = _lowpass_noise(300, n, sr, rng) * 0.25

    sig = base + elec + sat_noise

    # Slow overload swell (0.1–0.4 Hz) -- the motor struggles under load
    t = np.arange(n) / sr
    swell = 1.0 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
    sig = sig * swell

    # Slip wobble near shaft frequency (overloaded motor loses a little speed)
    sig = _amp_modulate(sig, shaft_hz * rng.uniform(0.95, 0.99), 0.12, sr, rng)
    return _normalize_peak(sig, 0.90)


SYNTH = {
    "motor_off":       synth_motor_off,
    "normal":          synth_normal,
    "bearing_fault":   synth_bearing_fault,
    "cavitation":      synth_cavitation,
    "impeller_damage": synth_impeller_damage,
    "pipe_leak":       synth_pipe_leak,
    "motor_overload":  synth_motor_overload,
}


# ===========================================================================
# Augmentation
# ===========================================================================
def _speed_perturb(x, rate, n_target):
    if abs(rate - 1.0) < 1e-3:
        y = x
    else:
        new_len = max(1, int(round(len(x) / rate)))
        y = sps.resample(x, new_len)
    if len(y) >= n_target:
        return y[:n_target]
    return np.pad(y, (0, n_target - len(y)))


def _add_noise_snr(x, snr_db, rng):
    sig_power = np.mean(x ** 2) + 1e-12
    noise = rng.standard_normal(len(x))
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / noise_power)
    return x + noise


def _time_shift(x, rng, max_frac=0.3):
    shift = int(rng.uniform(-max_frac, max_frac) * len(x))
    return np.roll(x, shift)


def augment(x, class_name, rng):
    """Apply a randomised augmentation chain that preserves class-defining features."""
    n = config.CLIP_SAMPLES

    if class_name == "motor_off":
        # Keep it near-silent: vary the floor amplitude only slightly
        return _normalize_peak(x, rng.uniform(0.001, 0.005))

    # Speed / pitch variation (mild -- large changes smear freq features)
    if rng.random() < 0.6:
        x = _speed_perturb(x, rng.uniform(0.94, 1.06), n)

    # Time shift
    if rng.random() < 0.7:
        x = _time_shift(x, rng)

    # Additive white noise at controlled SNR
    # pipe_leak and motor_overload get a tighter SNR range to preserve their
    # characteristic frequency bands
    if rng.random() < 0.8:
        if class_name in ("pipe_leak", "motor_overload"):
            snr = rng.uniform(14.0, 35.0)   # less noise contamination
        else:
            snr = rng.uniform(8.0, 35.0)
        x = _add_noise_snr(x, snr, rng)

    # Final gain variation
    x = _normalize_peak(x, rng.uniform(0.55, 0.95))
    return x


# ===========================================================================
# WAV writing
# ===========================================================================
def write_wav(path, x, sr):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    try:
        import soundfile as sf
        sf.write(path, x, sr, subtype="PCM_16")
    except Exception:
        from scipy.io import wavfile
        wavfile.write(path, sr, (x * 32767.0).astype(np.int16))


# ===========================================================================
# Main generation routine
# ===========================================================================
def generate(samples_per_class, seed, classes=None):
    config.ensure_dirs()
    classes = classes or config.CLASSES
    rng = np.random.default_rng(seed)

    rows = []
    for cls in classes:
        synth_fn = SYNTH[cls]
        n_val = int(round(samples_per_class * config.VAL_SPLIT))
        order = list(range(samples_per_class))
        rng.shuffle(order)
        val_set = set(order[:n_val])

        for i in range(samples_per_class):
            clip = synth_fn(rng)
            clip = augment(clip, cls, rng)
            if len(clip) != config.CLIP_SAMPLES:
                clip = _speed_perturb(clip, 1.0, config.CLIP_SAMPLES)

            fname = f"{cls}_{i:04d}.wav"
            fpath = os.path.join(config.RAW_DIR, cls, fname)
            write_wav(fpath, clip, config.SAMPLE_RATE)
            rows.append({
                "filepath": os.path.relpath(fpath, config.BASE_DIR).replace("\\", "/"),
                "filename": fname,
                "class": cls,
                "class_idx": config.CLASS_TO_IDX[cls],
                "split": "val" if i in val_set else "train",
            })
        print(f"  [{cls:16s}] {samples_per_class} clips "
              f"({samples_per_class - n_val} train / {n_val} val)")

    with open(config.METADATA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "filename", "class", "class_idx", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} clips total.")
    print(f"Metadata written to: {config.METADATA_CSV}")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic motor-pump dataset.")
    ap.add_argument("--samples-per-class", type=int, default=config.SAMPLES_PER_CLASS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    print(f"Generating {args.samples_per_class} clips per class "
          f"({config.NUM_CLASSES} classes, seed={args.seed})...")
    generate(args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
