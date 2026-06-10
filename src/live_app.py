"""
live_app.py
===========
Real-time motor-pump monitoring web app.

Captures the laptop microphone with sounddevice, runs YOHO + the autoencoder on a
rolling window roughly once per second, and pushes results to the browser over
Socket.IO: current class, confidence, traffic-light status, anomaly warning, and a
live mel spectrogram. Every prediction is logged to SQLite (data/history.db) and is
searchable by time range.

Classification logic (priority order)
--------------------------------------
1. Energy gate: if the RMS of the raw waveform is below SILENCE_RMS_THRESHOLD,
   the result is immediately set to "motor_off" -- no model inference needed.
2. YOHO predicts the dominant class. If confidence >= LOW_CONF_THRESHOLD and the
   autoencoder score is within the trained threshold, the YOHO class is returned.
3. If the autoencoder score EXCEEDS the threshold AND YOHO confidence is LOW
   (the model has no strong match), the result is "unknown_sound".
4. If the autoencoder flags an anomaly but YOHO is confident, the AE vote overrides
   the status to RED (known fault confirmed by two independent signals).

Run (after training):
    python src/live_app.py
    open http://127.0.0.1:5000 in your browser
"""
import os
import sys
import io
import json
import base64
import sqlite3
import threading
from datetime import datetime, timedelta

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)
import config                              # noqa: E402
import features as F                       # noqa: E402
from models import YOHO, ConvAutoencoder  # noqa: E402

# ---------------------------------------------------------------------------
# Live settings
# ---------------------------------------------------------------------------
WINDOW_SEC = 2.0            # audio context fed to the model each prediction
BUFFER_SEC = 3.0            # how much recent audio we keep
PREDICT_INTERVAL = 1.0      # seconds between predictions

# RMS below this → motor is off (no model inference needed).
# Calibrate to your microphone: 0.005 works well for a laptop mic at arm's length.
SILENCE_RMS_THRESHOLD = 0.005

# YOHO confidence below this (mean presence over bins) → high uncertainty
LOW_CONF = config.LOW_CONF_THRESHOLD

app = Flask(__name__,
            template_folder=os.path.join(_ROOT, "templates"),
            static_folder=os.path.join(_ROOT, "static"))
app.config["SECRET_KEY"] = "motor-pump-phase1"
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.listening = False
        self.worker_running = False
        self.stream = None
        self.models_ready = False
        self.yoho = None
        self.ae = None
        self.mean = None
        self.std = None
        self.ae_thr = 0.0


S = State()


def load_models():
    """Load models + run config; tolerate absence so the UI still serves."""
    try:
        if not (os.path.exists(config.RUN_CONFIG_PATH)
                and os.path.exists(config.YOHO_MODEL_PATH)
                and os.path.exists(config.AE_MODEL_PATH)):
            print("WARNING: trained model artifacts not found - run src/train.py. "
                  "UI will load but detection is disabled.")
            return
        with open(config.RUN_CONFIG_PATH) as f:
            cfg = json.load(f)
        S.mean = np.array(cfg["feat_mean"], dtype=np.float32).reshape(-1, 1)
        S.std = np.array(cfg["feat_std"], dtype=np.float32).reshape(-1, 1)
        S.ae_thr = float(cfg.get("ae_threshold", 0.0))
        S.yoho = YOHO()
        S.yoho.load_state_dict(torch.load(config.YOHO_MODEL_PATH, map_location="cpu"))
        S.yoho.eval()
        S.ae = ConvAutoencoder()
        S.ae.load_state_dict(torch.load(config.AE_MODEL_PATH, map_location="cpu"))
        S.ae.eval()
        S.models_ready = True
        print("Models loaded. Detection enabled.")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: failed to load models ({e}). Detection disabled.")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def db_conn():
    return sqlite3.connect(config.HISTORY_DB, check_same_thread=False)


def init_db():
    config.ensure_dirs()
    with db_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, epoch REAL,
                predicted_class TEXT, confidence REAL,
                status TEXT, anomaly INTEGER, ae_score REAL,
                start_time TEXT, end_time TEXT, duration REAL
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_epoch ON predictions(epoch)")


def db_insert(row):
    with db_conn() as c:
        c.execute("""INSERT INTO predictions
            (ts, epoch, predicted_class, confidence, status, anomaly, ae_score,
             start_time, end_time, duration)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                  (row["ts"], row["epoch"], row["predicted_class"], row["confidence"],
                   row["status"], int(row["anomaly"]), row["ae_score"],
                   row["start_time"], row["end_time"], row["duration"]))


def db_query(start_epoch, end_epoch, limit=500):
    with db_conn() as c:
        cur = c.execute("""SELECT ts, predicted_class, confidence, status, anomaly,
                                  ae_score, start_time, end_time, duration
                           FROM predictions
                           WHERE epoch BETWEEN ? AND ?
                           ORDER BY epoch DESC LIMIT ?""",
                        (start_epoch, end_epoch, limit))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Audio capture + inference
# ---------------------------------------------------------------------------
def audio_callback(indata, frames, time_info, status):  # noqa: ARG001
    chunk = np.asarray(indata[:, 0], dtype=np.float32)
    with S.lock:
        S.buffer = np.concatenate([S.buffer, chunk])
        maxlen = int(BUFFER_SEC * config.SAMPLE_RATE)
        if len(S.buffer) > maxlen:
            S.buffer = S.buffer[-maxlen:]


def render_spectrogram_png(mel_db):
    """Small mel spectrogram as a base64 PNG (no axes) for the live display."""
    fig = plt.figure(figsize=(5.0, 1.7), dpi=80)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def process_window(win):
    """
    Run inference on a waveform window and return a prediction payload.

    Decision logic:
      1. RMS gate → motor_off  (no model needed)
      2. YOHO → dominant class + confidence
      3. Autoencoder → anomaly score
      4. If AE score > threshold AND YOHO confidence is low → unknown_sound
      5. If AE score > threshold AND YOHO confidence is high → keep YOHO class
         but force status to red (confirmed fault)
    """
    # --- Step 1: energy gate ---
    rms = float(np.sqrt(np.mean(win ** 2)))
    if rms < SILENCE_RMS_THRESHOLD:
        cls = "motor_off"
        confidence = 1.0
        ae_score = 0.0
        ae_flag = False
        status = "yellow"
        anomaly = False
    else:
        # --- Step 2: feature extraction ---
        raw = F.logmel(win)
        feat = F.normalize(raw, S.mean, S.std)

        # --- Step 3: YOHO prediction ---
        pred = S.yoho.predict(feat)                       # (N_BINS, C, 3)
        presence = pred[:, :, 0].mean(axis=0)             # mean presence per class
        cls_idx = int(np.argmax(presence))
        cls = config.IDX_TO_CLASS[cls_idx]
        confidence = float(presence[cls_idx])

        # --- Step 4: autoencoder anomaly score ---
        ae_score = float(S.ae.anomaly_score(feat))
        ae_flag = ae_score > S.ae_thr

        # --- Step 5: decision tree ---
        if ae_flag and confidence < LOW_CONF:
            # Model is uncertain AND the AE says it's anomalous → unknown
            cls = config.UNKNOWN_CLASS
            confidence = 0.0
            status = "red"
            anomaly = True
        elif ae_flag:
            # AE confirms an anomaly even though YOHO has a match
            status = "red"
            anomaly = True
        else:
            status = config.status_for_class(cls)
            if status == "green" and confidence < LOW_CONF:
                status = "yellow"
            anomaly = status == "red"

    now = datetime.now()
    start = now - timedelta(seconds=WINDOW_SEC)
    payload = {
        "ts": now.strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": now.timestamp(),
        "predicted_class": cls,
        "confidence": round(confidence, 3),
        "status": status,
        "anomaly": anomaly,
        "ae_score": round(ae_score, 5),
        "ae_threshold": round(S.ae_thr, 5),
        "start_time": start.strftime("%H:%M:%S"),
        "end_time": now.strftime("%H:%M:%S"),
        "duration": round(WINDOW_SEC, 2),
    }
    # Add spectrogram only when the motor is running (skip for motor_off)
    if cls != "motor_off":
        raw_for_spec = F.logmel(win)
        payload["spectrogram"] = render_spectrogram_png(raw_for_spec)
    return payload


def worker():
    S.worker_running = True
    need = int(WINDOW_SEC * config.SAMPLE_RATE)
    while S.listening:
        socketio.sleep(PREDICT_INTERVAL)
        with S.lock:
            buf = S.buffer.copy()
        if len(buf) < need:
            continue
        try:
            payload = process_window(buf[-need:])
        except Exception as e:  # noqa: BLE001
            socketio.emit("error", {"message": f"inference error: {e}"})
            continue
        db_insert(payload)
        socketio.emit("prediction", payload)
    S.worker_running = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           classes=config.CLASSES + [config.UNKNOWN_CLASS],
                           colors=config.CLASS_COLORS,
                           anomaly_classes=config.ANOMALY_CLASSES + [config.UNKNOWN_CLASS],
                           models_ready=S.models_ready)


@app.route("/api/devices")
def devices():
    try:
        import sounddevice as sd
        out = [{"index": i, "name": d["name"]}
               for i, d in enumerate(sd.query_devices())
               if d["max_input_channels"] > 0]
        return jsonify({"devices": out})
    except Exception as e:  # noqa: BLE001
        return jsonify({"devices": [], "error": str(e)})


def _parse_dt(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        if "T" in s or "-" in s:
            return datetime.fromisoformat(s).timestamp()
        today = datetime.now().strftime("%Y-%m-%d")
        return datetime.fromisoformat(f"{today}T{s}").timestamp()
    except ValueError:
        return None


@app.route("/api/search")
def search():
    start = _parse_dt(request.args.get("start"))
    end = _parse_dt(request.args.get("end"))
    if start is None:
        start = 0.0
    if end is None:
        end = datetime.now().timestamp()
    rows = db_query(start, end)
    anomalies = sum(1 for r in rows if r["anomaly"])
    return jsonify({"count": len(rows), "anomalies": anomalies, "rows": rows})


@app.route("/api/history")
def history():
    limit = int(request.args.get("limit", 20))
    rows = db_query(0.0, datetime.now().timestamp(), limit=limit)
    return jsonify({"rows": rows})


# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------
@socketio.on("connect")
def on_connect():
    socketio.emit("status", {"listening": S.listening,
                             "models_ready": S.models_ready})


@socketio.on("start")
def on_start(data=None):
    if not S.models_ready:
        socketio.emit("error", {"message": "Models not loaded. Run src/train.py first."})
        return
    if S.listening:
        return
    device = (data or {}).get("device")
    device = int(device) if device not in (None, "", "default") else None
    try:
        import sounddevice as sd
        with S.lock:
            S.buffer = np.zeros(0, dtype=np.float32)
        S.stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1,
                                  dtype="float32", device=device,
                                  callback=audio_callback)
        S.stream.start()
        S.listening = True
        if not S.worker_running:
            socketio.start_background_task(worker)
        socketio.emit("status", {"listening": True, "models_ready": True})
    except Exception as e:  # noqa: BLE001
        S.listening = False
        socketio.emit("error", {"message": f"Could not open microphone: {e}"})


@socketio.on("stop")
def on_stop(data=None):  # noqa: ARG001
    S.listening = False
    try:
        if S.stream is not None:
            S.stream.stop()
            S.stream.close()
            S.stream = None
    except Exception:  # noqa: BLE001
        pass
    socketio.emit("status", {"listening": False, "models_ready": S.models_ready})


if __name__ == "__main__":
    init_db()
    load_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://0.0.0.0:{port} in your browser.")
    socketio.run(app, host="0.0.0.0", port=port,
                 debug=False, allow_unsafe_werkzeug=True)


# ---------------------------------------------------------------------------
# CLOUD API — POST /predict
# ---------------------------------------------------------------------------
# Accepts a WAV file from any remote device and returns the fault prediction.
# The remote device sends:
#   curl -X POST http://<your-railway-url>/predict -F "audio=@chunk.wav"
# Response JSON:
#   {"predicted_class": "bearing_fault", "confidence": 0.87,
#    "status": "red", "anomaly": true, "ae_score": 0.031}
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_api():
    if not S.models_ready:
        return jsonify({"error": "Models not loaded"}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file. Send as multipart field 'audio'"}), 400

    import tempfile, soundfile as sf

    f = request.files["audio"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        f.save(tmp_path)

    try:
        wav, sr = sf.read(tmp_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav[:, 0]                          # stereo → mono
        if sr != config.SAMPLE_RATE:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        need = int(WINDOW_SEC * config.SAMPLE_RATE)
        if len(wav) < need:
            # pad if clip is shorter than WINDOW_SEC
            wav = np.pad(wav, (0, need - len(wav)))
        win = wav[-need:]                            # take last WINDOW_SEC seconds

        payload = process_window(win)
        os.unlink(tmp_path)

        # Save to DB + push to all open browser dashboards in real time
        db_insert(payload)
        socketio.emit("prediction", payload)

        # Return only the fields relevant to the API caller
        return jsonify({
            "predicted_class": payload["predicted_class"],
            "confidence":       payload["confidence"],
            "status":           payload["status"],
            "anomaly":          payload["anomaly"],
            "ae_score":         payload["ae_score"],
            "ae_threshold":     payload["ae_threshold"],
            "ts":               payload["ts"],
        })
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return jsonify({"error": str(e)}), 500
