# MotorPump1 — Real-Time Acoustic Fault Detection

A complete audio-based predictive maintenance system for motor pumps. It listens to
the pump's acoustic signature via microphone, classifies what is happening each
second, and pushes live traffic-light alerts to a browser dashboard.

---

## What changed in this revision

| Area | Change |
|---|---|
| `background_noise` class | **Removed.** It was acoustically too similar to other classes and caused false positives. |
| `silence` class | **Renamed to `motor_off`**. Now specifically means the motor is completely switched off. Detected by an RMS energy gate before any model runs — no model inference is wasted on silence. |
| `unknown_sound` | **New label** (not a trained class). Assigned by `live_app.py` when the autoencoder flags an anomaly AND YOHO confidence is low. Means "something is wrong but it does not match any known fault". Shown in amber. |
| `pipe_leak` synthesis | **Reworked**: now a sustained mid-high turbulent hiss concentrated in **1–6 kHz**. No longer shares the motor-base harmonic signature with `normal` or `motor_overload`. |
| `motor_overload` synthesis | **Reworked**: dominated by boosted **50/100/150 Hz electrical harmonics** plus a slow load swell. Energy is concentrated **below 500 Hz** — the opposite of `pipe_leak`. |
| `impeller_damage` synthesis | Strengthened 1× shaft imbalance tone and per-revolution knock for clear separation. |
| Augmentation | Tighter SNR for `pipe_leak` and `motor_overload` to preserve their characteristic frequency bands through augmentation. |
| Samples per class | Increased to **300** (from 250) for better generalisation. |

---

## Classes

| Class | Colour | Status | Description |
|---|---|---|---|
| `motor_off` | grey | 🟡 Yellow | Motor is switched off — near-silence, RMS-gated |
| `normal` | green | 🟢 Green | Healthy pump: clean shaft harmonics |
| `bearing_fault` | red | 🔴 Red | Periodic high-freq impulse bursts at defect frequency |
| `cavitation` | red | 🔴 Red | Random wideband crackle bursts (2–7 kHz) |
| `impeller_damage` | purple | 🔴 Red | Strong 1× imbalance + per-rev knock |
| `pipe_leak` | blue | 🔴 Red | Sustained turbulent hiss (1–6 kHz) |
| `motor_overload` | orange | 🔴 Red | Boosted 50/100/150 Hz electrical lines + slow swell |
| `unknown_sound` | amber | 🔴 Red | AE anomaly but no confident YOHO match |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (≈ 5–10 min on CPU)
python src/dataset_generation.py

# 3. Generate mixed test clips
python src/mixed_audio_generation.py

# 4. Train both models (≈ 10–30 min on CPU)
python src/train.py

# 5. Evaluate on the mixed clips
python src/test_model.py

# 6. Launch the live dashboard
python src/live_app.py
# → open http://127.0.0.1:5000
```

---

## Live classification decision logic

```
waveform → RMS check
  │
  ├─ RMS < 0.005 → motor_off (yellow)
  │
  └─ RMS OK → YOHO + Autoencoder
               │
               ├─ AE normal  + YOHO confident → YOHO class (green/red)
               ├─ AE anomaly + YOHO confident → YOHO class, forced RED
               ├─ AE anomaly + YOHO uncertain → unknown_sound (red)
               └─ AE normal  + YOHO uncertain → YOHO class, yellow
```

---

## Tuning `SILENCE_RMS_THRESHOLD`

The default value (`0.005` in `live_app.py`) suits a laptop mic at arm's length.
If `motor_off` is triggered while the pump IS running, lower the value. If it
triggers too rarely in a quiet room, raise it. Print the RMS to calibrate:

```python
import sounddevice as sd, numpy as np
while True:
    audio = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    print(f"RMS: {np.sqrt(np.mean(audio**2)):.5f}")
```

---

## Project structure

```
MotorPump1/
├── config.py                    ← single source of truth for all settings
├── requirements.txt
├── src/
│   ├── dataset_generation.py    ← synthetic audio synthesis per class
│   ├── mixed_audio_generation.py← build multi-event test clips
│   ├── features.py              ← log-mel extraction + YOHO encode/decode
│   ├── models.py                ← YOHO + ConvAutoencoder (PyTorch)
│   ├── train.py                 ← training loop + artifact saving
│   ├── test_model.py            ← evaluation on mixed clips
│   ├── visualize.py             ← spectrogram plotting
│   └── live_app.py              ← Flask + Socket.IO real-time dashboard
├── data/
│   ├── raw/<class>/             ← generated WAV clips
│   ├── mixed/                   ← test clips + annotations
│   └── history.db               ← SQLite prediction log
├── models/                      ← saved weights + config.json + labels.json
├── outputs/
│   ├── predictions/             ← per-file JSON + summary CSV
│   └── visualizations/          ← annotated spectrogram PNGs
├── static/                      ← app.js, style.css, socket.io.min.js
└── templates/index.html         ← dashboard UI
```
# redeploy
