"""
config.py — Central configuration for Deepfake Detection Project ai
Edit the paths and hyperparameters here before running anything.
"""

import os
import torch

class CFG:
    # ── Dataset paths ─────────────────────────────────────────
    # Your primary training dataset (must have real/ and fake/ subfolders)
    DATA_ROOT = "archive/real_and_fake_face"          # Windows
    # DATA_ROOT = "/home/user/deepfake_project/dataset"  # Linux/Mac

    # Where model checkpoints and plots are saved
    CKPT_DIR  = "checkpoints"      # Windows
    # CKPT_DIR  = "/home/user/deepfake_project/checkpoints" # Linux/Mac

    # ── Cross-dataset paths ───────────────────────────────────
    # Add paths to additional datasets for generalization testing
    # Each path must also have real/ and fake/ subfolders
    CROSS_DATASETS = {
        "Dataset2_CipLab" : "Dataset"
        #"140K"          : r"D:\Gr 2\140k"
        # "FaceForensics"   : r"C:\deepfake_project\faceforensics",
        # "StyleGAN_140k"   : r"C:\deepfake_project\140k_faces",
    }

    # ── Multi-dataset training paths ──────────────────────────
    # Datasets to combine for multi-dataset training experiment
    MULTI_TRAIN_DATASETS = [
        "Dataset",
        "archive/real_and_fake_face"
        #r"D:\Gr 2\140k"
    ]

    # ── Training hyperparameters ──────────────────────────────
    SEED         = 42
    IMG_SIZE     = 224
    BATCH_SIZE   = 32        # reduce to 8 if you get out-of-memory errors
    NUM_WORKERS  = 0         # 0 = main process only (safest on Windows)
    EPOCHS_BASE  = 20        # EfficientNet and Swin
    EPOCHS_FREQ  = 15        # FreqCNN
    EPOCHS_ENS   = 10        # Ensemble MLP
    LR           = 3e-4
    LR_ENS       = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT    = 0.15
    TEST_SPLIT   = 0.10

    # ── Device ───────────────────────────────────────────────
    # Auto-detect: uses CUDA GPU if available, else CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Class labels ─────────────────────────────────────────
    CLASSES = ["real", "fake"]   # index 0 = real, index 1 = fake

    # ── Gradio demo port ─────────────────────────────────────
    GRADIO_PORT   = 7860
    GRADIO_SHARE  = False   # set True to get a public URL

# Create checkpoint directory if it doesn't exist
os.makedirs(CFG.CKPT_DIR, exist_ok=True)