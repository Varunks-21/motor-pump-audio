"""
train.py
========
Train both Phase 1 models and save everything the later stages need.

Pipeline:
  1. Read data/raw/metadata.csv and load each clip's log-mel (cached once).
  2. Compute per-mel-bin normalisation stats on the TRAIN split (saved to config.json).
  3. Train YOHO on all 8 classes  (whole clip = one event of that class).
       - masked presence + boundary loss, Adam, EARLY STOPPING on val loss.
  4. Train the autoencoder on `normal` clips only, then set the anomaly threshold
     from the normal-clip reconstruction-error distribution.
  5. Save: models/yoho_best.pt, models/autoencoder_best.pt, labels.json, config.json.

Run:
    python src/train.py                       # full training
    python src/train.py --epochs 40 --batch-size 16
    python src/train.py --quick               # tiny run to verify the pipeline

Notes for CPU / Windows:
  - Features are cached in RAM once (~160 KB per clip). 2000 clips ~= 320 MB.
    Generate fewer clips (e.g. --samples-per-class 150) if RAM is tight.
  - DataLoader uses num_workers=0 (most reliable on Windows).
"""
import os
import sys
import csv
import json
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SRC)
import config                                            # noqa: E402
import features as F                                     # noqa: E402
from models import YOHO, YohoLoss, ConvAutoencoder, count_params  # noqa: E402


# ===========================================================================
# Utilities
# ===========================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    """
    Stop training when validation loss has not improved for `patience` epochs.
    Keeps a copy of the best model weights so we always save the best, not the last.
    """

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None
        self.best_epoch = -1

    def step(self, val_loss, model, epoch):
        improved = val_loss < self.best - self.min_delta
        if improved:
            self.best = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return improved

    @property
    def stop(self):
        return self.counter >= self.patience


def read_metadata():
    if not os.path.exists(config.METADATA_CSV):
        sys.exit("ERROR: data/raw/metadata.csv not found. "
                 "Run `python src/dataset_generation.py` first.")
    with open(config.METADATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    return train, val


def build_feature_cache(rows):
    """Compute the raw (un-normalised) log-mel for every clip, once."""
    cache = {}
    n = len(rows)
    for i, r in enumerate(rows):
        path = os.path.join(config.BASE_DIR, r["filepath"])
        y = F.load_audio(path, fix_to=config.CLIP_SAMPLES)
        cache[r["filepath"]] = F.logmel(y)
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"    features {i + 1}/{n}", end="\r")
    print()
    return cache


# ===========================================================================
# Datasets
# ===========================================================================
class YohoDataset(Dataset):
    def __init__(self, rows, cache, mean, std):
        self.rows, self.cache, self.mean, self.std = rows, cache, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feat = F.normalize(self.cache[r["filepath"]], self.mean, self.std)
        # a whole single-class clip = one event spanning the entire clip
        events = [{"event_class": r["class"],
                   "start_time": 0.0, "end_time": config.CLIP_DURATION}]
        tgt = F.events_to_targets(events)
        return torch.from_numpy(feat), torch.from_numpy(tgt)


class AEDataset(Dataset):
    def __init__(self, rows, cache, mean, std):
        self.rows, self.cache, self.mean, self.std = rows, cache, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feat = F.normalize(self.cache[r["filepath"]], self.mean, self.std)
        return torch.from_numpy(feat)


# ===========================================================================
# Training loops
# ===========================================================================
def train_yoho(train_rows, val_rows, cache, mean, std, args, device):
    print("\n=== Training YOHO (event detection) ===")
    tr = DataLoader(YohoDataset(train_rows, cache, mean, std),
                    batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(YohoDataset(val_rows, cache, mean, std),
                    batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = YOHO().to(device)
    print(f"    parameters: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = YohoLoss(coord_weight=1.0)
    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, t in tr:
            x, t = x.to(device), t.to(device)
            opt.zero_grad()
            loss = crit(model(x), t)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, t in va:
                x, t = x.to(device), t.to(device)
                va_loss += crit(model(x), t).item() * x.size(0)
        va_loss /= max(1, len(va.dataset))

        improved = stopper.step(va_loss, model, epoch)
        flag = " *" if improved else ""
        print(f"  epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}{flag}")
        if stopper.stop:
            print(f"  early stopping (no val improvement for {args.patience} epochs)")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), config.YOHO_MODEL_PATH)
    print(f"  best val loss {stopper.best:.4f} @ epoch {stopper.best_epoch} "
          f"-> {config.YOHO_MODEL_PATH}")
    return model


def train_autoencoder(train_rows, val_rows, cache, mean, std, args, device):
    print("\n=== Training Autoencoder (anomaly detection, normal-only) ===")
    norm_tr = [r for r in train_rows if r["class"] == "normal"]
    norm_va = [r for r in val_rows if r["class"] == "normal"]
    print(f"    normal clips: {len(norm_tr)} train / {len(norm_va)} val")

    tr = DataLoader(AEDataset(norm_tr, cache, mean, std),
                    batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(AEDataset(norm_va, cache, mean, std),
                    batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ConvAutoencoder().to(device)
    print(f"    parameters: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    stopper = EarlyStopping(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x in tr:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = crit(recon, x.unsqueeze(1))
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(tr.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x in va:
                x = x.to(device)
                va_loss += crit(model(x), x.unsqueeze(1)).item() * x.size(0)
        va_loss /= max(1, len(va.dataset))

        improved = stopper.step(va_loss, model, epoch)
        flag = " *" if improved else ""
        print(f"  epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}{flag}")
        if stopper.stop:
            print(f"  early stopping (no val improvement for {args.patience} epochs)")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), config.AE_MODEL_PATH)

    # set the anomaly threshold from the normal-clip error distribution
    model.eval()
    errs = []
    with torch.no_grad():
        for r in norm_tr:
            feat = F.normalize(cache[r["filepath"]], mean, std)
            errs.append(model.anomaly_score(feat))
    errs = np.array(errs)
    threshold = float(np.percentile(errs, 99))
    print(f"  normal recon error: mean {errs.mean():.4f}  std {errs.std():.4f}  "
          f"p99 {threshold:.4f}")
    print(f"  best val loss {stopper.best:.4f} @ epoch {stopper.best_epoch} "
          f"-> {config.AE_MODEL_PATH}")
    return model, threshold


# ===========================================================================
# Artifact saving
# ===========================================================================
def save_artifacts(mean, std, ae_threshold):
    labels = {
        "classes": config.CLASSES,
        "class_to_idx": config.CLASS_TO_IDX,
        "anomaly_classes": config.ANOMALY_CLASSES,
        "normal_classes": config.NORMAL_CLASSES,
        "neutral_classes": config.NEUTRAL_CLASSES,
        "class_colors": config.CLASS_COLORS,
    }
    with open(config.LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    run_cfg = {
        "sample_rate": config.SAMPLE_RATE,
        "clip_duration": config.CLIP_DURATION,
        "clip_samples": config.CLIP_SAMPLES,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "win_length": config.WIN_LENGTH,
        "n_mels": config.N_MELS,
        "fmin": config.FMIN,
        "fmax": config.FMAX,
        "n_bins": config.N_BINS,
        "bin_dur": config.BIN_DUR,
        "presence_threshold": config.PRESENCE_THRESHOLD,
        "feat_mean": mean.reshape(-1).tolist(),
        "feat_std": std.reshape(-1).tolist(),
        "ae_threshold": ae_threshold,
    }
    with open(config.RUN_CONFIG_PATH, "w") as f:
        json.dump(run_cfg, f, indent=2)

    print(f"\nSaved labels -> {config.LABELS_PATH}")
    print(f"Saved run config -> {config.RUN_CONFIG_PATH}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Train YOHO + autoencoder.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--quick", action="store_true",
                    help="tiny run (5 epochs, patience 3) to verify the pipeline")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.patience = 5, 3

    set_seed(args.seed)
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_rows, val_rows = read_metadata()
    print(f"Clips: {len(train_rows)} train / {len(val_rows)} val")

    t0 = time.time()
    print("\nExtracting features (cached once)...")
    print("  train:")
    cache = build_feature_cache(train_rows)
    print("  val:")
    cache.update(build_feature_cache(val_rows))

    # normalisation stats from TRAIN features only
    mean, std = F.compute_norm_stats([cache[r["filepath"]] for r in train_rows])
    print(f"Norm stats computed (per-mel-bin). mean range "
          f"[{mean.min():.1f}, {mean.max():.1f}] dB")

    train_yoho(train_rows, val_rows, cache, mean, std, args, device)
    _, ae_threshold = train_autoencoder(train_rows, val_rows, cache, mean, std, args, device)

    save_artifacts(mean, std, ae_threshold)
    print(f"\nAll done in {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()