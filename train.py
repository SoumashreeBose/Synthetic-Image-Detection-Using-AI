"""
train.py — Main entry point for the full training pipeline
Run: python train.py

Steps executed:
  1. Load data
  2. Train EfficientNet-B4
  3. Train Swin-Tiny
  4. Train FreqCNN
  5. Train Ensemble MLP
  6. Full evaluation + plots
  7. Grad-CAM visualizations
  8. Cross-dataset evaluation (if paths set in config.py)
  9. Multi-dataset training experiment (if paths set)
"""
import os
os.environ.pop("SSL_CERT_FILE", None)   # remove broken cert path
import random
import numpy as np
import torch

from config import CFG
from dataset import make_loaders, make_combined_loader
from models import EfficientNetModel, SwinModel, FreqCNN, load_all_models
from trainer import (
    train_base_model, train_ensemble,
    full_evaluation, cross_dataset_evaluation
)
from visualize import (
    plot_training_curves, plot_evaluation,
    visualize_ensemble_gradcam, build_gradcam_hooks,
    plot_cross_dataset_results, plot_generalization_drop
)


def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    seed_everything()
    print(f"\n{'='*60}")
    print("  DEEPFAKE DETECTION — MULTI-MODEL ENSEMBLE PIPELINE")
    print(f"  Device : {CFG.DEVICE}")
    print(f"  Data   : {CFG.DATA_ROOT}")
    print(f"  Output : {CFG.CKPT_DIR}")
    print(f"{'='*60}\n")

    # ── Step 1: Load data ──────────────────────────────────────
    (train_loader, val_loader, test_loader,
     train_freq_loader, val_freq_loader, test_freq_loader,
     test_samples) = make_loaders()

    # ── Step 2: Train EfficientNet-B4 ─────────────────────────
    efficientnet = EfficientNetModel(pretrained=True)
    efficientnet, hist_eff = train_base_model(
        efficientnet, "EfficientNet-B4",
        train_loader, val_loader,
        epochs=CFG.EPOCHS_BASE,
        lr=CFG.LR,
        use_warmup=False
    )

    # ── Step 3: Train Swin-Tiny ───────────────────────────────
    swin = SwinModel(pretrained=True)
    swin, hist_swin = train_base_model(
        swin, "Swin-Tiny",
        train_loader, val_loader,
        epochs=CFG.EPOCHS_BASE,
        lr=2e-4,
        use_warmup=True    # warmup is critical for transformers
    )

    # ── Step 4: Train FreqCNN ─────────────────────────────────
    freqcnn = FreqCNN()
    freqcnn, hist_freq = train_base_model(
        freqcnn, "FreqCNN",
        train_freq_loader, val_freq_loader,
        epochs=CFG.EPOCHS_FREQ,
        lr=6e-4,
        use_warmup=False
    )

    # ── Step 5: Plot training curves ──────────────────────────
    plot_training_curves({
        "EfficientNet-B4": hist_eff,
        "Swin-Tiny":       hist_swin,
        "FreqCNN":         hist_freq,
    })

    # ── Step 6: Train Ensemble MLP ────────────────────────────
    ensemble = train_ensemble(
        efficientnet, swin, freqcnn,
        train_loader, train_freq_loader,
        val_loader,   val_freq_loader
    )

    # ── Step 7: Full evaluation ───────────────────────────────
    results = full_evaluation(
        efficientnet, swin, freqcnn, ensemble,
        test_loader, test_freq_loader
    )
    plot_evaluation(results)

    # ── Step 8: Grad-CAM visualizations ──────────────────────
    cam_eff, swin_attn = build_gradcam_hooks(efficientnet, swin)
    visualize_ensemble_gradcam(
        efficientnet, swin, freqcnn, ensemble,
        cam_eff, swin_attn,
        test_loader, test_freq_loader,
        n_images=6
    )

    # ── Step 9: Cross-dataset evaluation  ─────────────────────
    if CFG.CROSS_DATASETS:
        cross_results = cross_dataset_evaluation(
            efficientnet, swin, freqcnn, ensemble,
            CFG.CROSS_DATASETS
        )
        plot_cross_dataset_results(cross_results)
        plot_generalization_drop(results, cross_results)
    else:
        print("\n[Info] No cross-datasets configured in config.py")
        print("       Add paths to CFG.CROSS_DATASETS to run this.")

    # ── Step 10: Multi-dataset training (optional) ────────────
    def find_val_dir(root):
        for name in ["val", "valid", "validation"]:
            if os.path.isdir(os.path.join(root, name)):
                return name
        return None 
    
    if len(CFG.MULTI_TRAIN_DATASETS) > 1:
        print(f"\n{'='*60}")
        print("  MULTI-DATASET TRAINING EXPERIMENT")
        print(f"{'='*60}")

        combined_train = make_combined_loader(
            CFG.MULTI_TRAIN_DATASETS, split="train")
        combined_val   = make_combined_loader(
            CFG.MULTI_TRAIN_DATASETS, split="val")

        eff_multi = EfficientNetModel(pretrained=True)
        eff_multi, _ = train_base_model(
            eff_multi, "EfficientNet-B4-Multi",
            combined_train, combined_val,
            epochs=CFG.EPOCHS_BASE, lr=CFG.LR
        )

        swin_multi = SwinModel(pretrained=True)
        swin_multi, _ = train_base_model(
            swin_multi, "Swin-Tiny-Multi",
            combined_train, combined_val,
            epochs=CFG.EPOCHS_BASE, lr=2e-4, use_warmup=True
        )

        freq_multi = FreqCNN()
        freq_multi, _ = train_base_model(
            freq_multi, "FreqCNN-Multi",
            combined_train, combined_val,
            epochs=CFG.EPOCHS_FREQ, lr=6e-4
        )
        print("\n  Multi-dataset models saved to checkpoints/")
    else:
        print("\n[Info] Multi-dataset training skipped.")
        print("       Add paths to CFG.MULTI_TRAIN_DATASETS to run.")

    print(f"\n{'='*60}")
    print("  ✓ Pipeline complete")
    print(f"  All outputs saved to: {CFG.CKPT_DIR}")
    print(f"{'='*60}")
    print("\nNext step: python demo.py  ← launch Gradio demo")


if __name__ == "__main__":
    main()