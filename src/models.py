"""
models.py
=========
The two PyTorch networks for Phase 1, both CPU-friendly:

  1. YOHO  -- supervised audio event detection / segmentation.
     A small conv backbone collapses frequency and pools time to a fixed N_BINS,
     then predicts (presence, start, end) per bin per class -- the YOHO format.

  2. ConvAutoencoder -- unsupervised anomaly detection.
     Trained ONLY on `normal` clips to reconstruct their log-mel. Reconstruction
     error is the anomaly score: sounds it never learned reconstruct poorly.

Both consume the same normalised log-mel feature (N_MELS x T) from features.py.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# ===========================================================================
# YOHO
# ===========================================================================
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU with configurable (freq, time) pooling."""

    def __init__(self, c_in, c_out, pool=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(pool) if pool else None

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class YOHO(nn.Module):
    """
    Input : (B, 1, N_MELS, T)  normalised log-mel
    Output: (B, N_BINS, NUM_CLASSES, 3)  RAW values
            channel 0 = presence logit (apply sigmoid for probability)
            channel 1 = start offset logit  (sigmoid -> 0..1 within bin)
            channel 2 = end   offset logit  (sigmoid -> 0..1 within bin)
    """

    def __init__(self, n_mels=config.N_MELS, n_classes=config.NUM_CLASSES,
                 n_bins=config.N_BINS):
        super().__init__()
        self.n_classes = n_classes
        self.n_bins = n_bins

        # pool frequency aggressively, time more gently
        self.block1 = ConvBlock(1, 16, pool=(2, 2))
        self.block2 = ConvBlock(16, 32, pool=(2, 2))
        self.block3 = ConvBlock(32, 64, pool=(2, 2))
        self.block4 = ConvBlock(64, 64, pool=(2, 1))
        self.block5 = ConvBlock(64, 128, pool=(2, 1))

        # collapse frequency to 1 and pin the time axis to exactly N_BINS
        self.pool_to_bins = nn.AdaptiveAvgPool2d((1, n_bins))

        # 1x1 conv over time -> (presence, start, end) per class
        self.head = nn.Conv1d(128, n_classes * 3, kernel_size=1)

    def forward(self, x):
        if x.dim() == 3:                         # (B, N_MELS, T) -> add channel
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool_to_bins(x)                 # (B, 128, 1, N_BINS)
        x = x.squeeze(2)                         # (B, 128, N_BINS)
        x = self.head(x)                         # (B, C*3, N_BINS)
        x = x.transpose(1, 2)                    # (B, N_BINS, C*3)
        return x.reshape(x.size(0), self.n_bins, self.n_classes, 3)

    @torch.no_grad()
    def predict(self, feat):
        """
        feat: numpy or tensor (N_MELS, T) OR (B, N_MELS, T).
        Returns numpy (N_BINS, C, 3) [single] or (B, N_BINS, C, 3) with sigmoids
        applied to all three channels.
        """
        self.eval()
        t = torch.as_tensor(feat, dtype=torch.float32)
        single = (t.dim() == 2)
        if single:
            t = t.unsqueeze(0)
        out = torch.sigmoid(self.forward(t))
        out = out.cpu().numpy()
        return out[0] if single else out


class YohoLoss(nn.Module):
    """
    presence -> BCEWithLogits over all bins x classes.
    start/end -> MSE, counted only where the GROUND-TRUTH class is present.
    """

    def __init__(self, coord_weight=1.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # pred raw logits, target in [0,1]; shapes (B, N_BINS, C, 3)
        pres_logit = pred[..., 0]
        pres_tgt = target[..., 0]
        loss_p = self.bce(pres_logit, pres_tgt)

        coords = torch.sigmoid(pred[..., 1:3])
        coord_tgt = target[..., 1:3]
        mask = pres_tgt.unsqueeze(-1)            # (B, N_BINS, C, 1)
        denom = mask.sum() * 2 + 1e-6            # 2 coords per active cell
        loss_c = (((coords - coord_tgt) ** 2) * mask).sum() / denom

        return loss_p + self.coord_weight * loss_c


# ===========================================================================
# Convolutional Autoencoder (anomaly detection on `normal` clips)
# ===========================================================================
class ConvAutoencoder(nn.Module):
    """
    Reconstructs the normalised log-mel. Trained on `normal` only.
    Input/output (B, 1, N_MELS, T). Handles arbitrary T by padding H and W up to
    a multiple of 16 internally, then cropping the reconstruction back to size.
    """

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        _, _, h, w = x.shape
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        xp = F.pad(x, (0, pw, 0, ph))            # pad W then H to multiples of 16
        z = self.enc(xp)
        out = self.dec(z)
        return out[:, :, :h, :w]                 # crop back to original size

    @torch.no_grad()
    def anomaly_score(self, feat):
        """Mean squared reconstruction error for one clip (or batch)."""
        self.eval()
        t = torch.as_tensor(feat, dtype=torch.float32)
        single = (t.dim() == 2)
        if single:
            t = t.unsqueeze(0)
        recon = self.forward(t)
        x = t.unsqueeze(1) if t.dim() == 3 else t
        err = ((recon - x) ** 2).mean(dim=[1, 2, 3])
        err = err.cpu().numpy()
        return float(err[0]) if single else err


# ===========================================================================
# Small helpers
# ===========================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick shape smoke-test (requires torch installed)
    T = config.FRAMES_PER_CLIP
    x = torch.randn(2, config.N_MELS, T)

    yoho = YOHO()
    y = yoho(x)
    print("YOHO output:", tuple(y.shape),
          "| expected:", (2, config.N_BINS, config.NUM_CLASSES, 3),
          "| params:", count_params(yoho))

    ae = ConvAutoencoder()
    r = ae(x)
    print("AE output:", tuple(r.shape),
          "| expected:", (2, 1, config.N_MELS, T),
          "| params:", count_params(ae))
    print("AE anomaly score (batch):", ae.anomaly_score(x))