"""
models.py — All four model architectures
  1. EfficientNetModel  — spatial CNN
  2. SwinModel          — transformer
  3. FreqCNN            — SRM filter bank + CNN
  4. EnsembleMLP        — learned combination of base models 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import CFG


# ─────────────────────────────────────────────────────────────
# MODEL 1: EfficientNet-B4
# ─────────────────────────────────────────────────────────────

class EfficientNetModel(nn.Module):
    """
    Compound-scaled CNN for spatial artifact detection.
    Detects: GAN fingerprints, blending artifacts, unnatural textures.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 1792
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_embedding(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
# MODEL 2: Swin-Tiny Transformer
# ─────────────────────────────────────────────────────────────

class SwinModel(nn.Module):
    """
    Shifted-window transformer for global inconsistency detection.
    Detects: mismatched lighting, geometric incoherence,
             cross-region artifacts CNNs miss.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        feat_dim = self.backbone.num_features  # 768
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_embedding(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
# MODEL 3: FreqCNN with SRM Filter Bank
# ─────────────────────────────────────────────────────────────

# Steganalysis Rich Model kernels — fixed, non-trainable
SRM_KERNELS = np.array([
    [[-1,  2, -1],[ 2, -4,  2],[-1,  2, -1]],   # Square-3x3
    [[ 0,  0,  0],[ 0, -1,  1],[ 0,  0,  0]],   # H-Edge
    [[ 1, -2,  1],[-2,  4, -2],[ 1, -2,  1]],   # Laplacian
    [[ 0,  0, -1],[ 0,  1,  0],[ 0,  0,  0]],   # Diagonal
    [[ 0, -1,  0],[-1,  4, -1],[ 0, -1,  0]],   # Cross-deriv
], dtype=np.float32)

SRM_OUT_CHANNELS = len(SRM_KERNELS) * 3   # 15


class SRMFilterBank(nn.Module):
    """
    Fixed (frozen) SRM filter bank.
    Input : (B, 3, H, W)  — raw pixels [0, 1]
    Output: (B, 15, H, W) — noise residual maps in [-1, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            3, SRM_OUT_CHANNELS, 3, padding=1, bias=False)
        weight = np.zeros(
            (SRM_OUT_CHANNELS, 3, 3, 3), dtype=np.float32)
        for k_idx, kernel in enumerate(SRM_KERNELS):
            for c_idx in range(3):
                weight[k_idx * 3 + c_idx, c_idx] = kernel / 4.0
        self.conv.weight = nn.Parameter(
            torch.from_numpy(weight), requires_grad=False)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class FreqCNN(nn.Module):
    """
    Frequency-domain CNN.
    Detects: GAN upsampling grid artifacts, periodic noise,
             high-frequency residuals invisible spatially.
    Most robust model across unseen generators.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.srm = SRMFilterBank()

        def conv_block(in_c, out_c, pool=True):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(SRM_OUT_CHANNELS, 32),   # 112
            conv_block(32, 64),                  # 56
            conv_block(64, 128),                 # 28
            conv_block(128, 256, pool=False),
            nn.AdaptiveAvgPool2d((4, 4))         # 4×4
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.features(self.srm(x)).flatten(1))

    def get_embedding(self, x):
        return self.features(self.srm(x)).flatten(1)

    def get_residuals(self, x):
        with torch.no_grad():
            return self.srm(x)


# ─────────────────────────────────────────────────────────────
# MODEL 4: Ensemble MLP
# ─────────────────────────────────────────────────────────────

class EnsembleMLP(nn.Module):
    """
    Learns when to trust each base model.
    Input : 6-dim concatenated softmax from all 3 base models
    Output: 2-class final prediction
    """
    def __init__(self, in_dim=6, hidden=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────

def save_model(model, name):
    path = f"{CFG.CKPT_DIR}/{name}.pth"
    torch.save(model.state_dict(), path)
    print(f"  Saved → {path}")


def load_all_models(device=None):
    """Load all four trained models from saved checkpoints."""
    if device is None:
        device = CFG.DEVICE

    eff = EfficientNetModel(pretrained=False).to(device)
    eff.load_state_dict(torch.load(
        f"{CFG.CKPT_DIR}/EfficientNet-B4.pth",
        map_location=device))
    eff.eval()

    swin = SwinModel(pretrained=False).to(device)
    swin.load_state_dict(torch.load(
        f"{CFG.CKPT_DIR}/Swin-Tiny.pth",
        map_location=device))
    swin.eval()

    freq = FreqCNN().to(device)
    freq.load_state_dict(torch.load(
        f"{CFG.CKPT_DIR}/FreqCNN.pth",
        map_location=device))
    freq.eval()

    ens = EnsembleMLP(in_dim=6).to(device)
    ens.load_state_dict(torch.load(
        f"{CFG.CKPT_DIR}/ensemble.pth",
        map_location=device))
    ens.eval()

    print("✓ All models loaded from checkpoints")
    return eff, swin, freq, ens