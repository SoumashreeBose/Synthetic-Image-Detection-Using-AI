# 🔍 Synthetic Image Detection Using AI

A deep learning pipeline that detects AI-generated / manipulated face images by combining three complementary architectures — a spatial CNN, a vision transformer, and a frequency-domain CNN — through a learned ensemble. Includes a Grad-CAM powered Gradio demo for interactive, explainable predictions.

---

## ✨ Key Features

- **Three specialized base models**, each catching different types of forgery artifacts:
  - **EfficientNet-B4** — spatial CNN tuned for GAN fingerprints, blending seams, and unnatural textures.
  - **Swin-Tiny** — shifted-window transformer for global inconsistencies (mismatched lighting, geometric incoherence) that CNNs tend to miss.
  - **FreqCNN** — a custom CNN built on a fixed **SRM (Steganalysis Rich Model) filter bank** that operates on noise residuals to catch GAN upsampling grid artifacts and periodic noise patterns. Typically the most robust model on unseen generators.
- **Learned Ensemble MLP** that combines the softmax outputs of all three base models rather than simple averaging/voting.
- **Explainability built in**: Grad-CAM heatmaps for EfficientNet, attention maps for Swin, and noise-residual visualizations for FreqCNN.
- **Cross-dataset generalization testing** — evaluate trained models on entirely unseen datasets to measure real-world robustness.
- **Multi-dataset training experiment** — optionally combine multiple datasets for a single training run to improve generalization.
- **Interactive Gradio demo** with a live 6-panel visualization dashboard (original image, Grad-CAM overlay, heatmap, frequency residual, attention map, ensemble verdict card).

---

## 🗂️ Project Structure

```
.
├── config.py       # Central configuration (paths, hyperparameters, device)
├── dataset.py      # Dataset discovery, splitting, transforms, DataLoaders
├── models.py       # EfficientNet-B4, Swin-Tiny, FreqCNN, EnsembleMLP definitions
├── trainer.py       # Training loop, evaluation, cross-dataset evaluation
├── visualize.py    # Grad-CAM, training curves, evaluation plots
├── train.py       # Main training pipeline entry point
├── demo.py        # Gradio web demo with live Grad-CAM
└── checkpoints/    # Saved model weights & generated plots (created at runtime)
```

---

## 🏗️ Architecture Overview

```
                ┌─────────────────┐
   Image ─────► │  EfficientNet-B4 │──┐
                └─────────────────┘  │
                ┌─────────────────┐  │      ┌──────────────┐
   Image ─────► │    Swin-Tiny     │──┼───► │ Ensemble MLP │──► Real / Fake
                └─────────────────┘  │      └──────────────┘
                ┌─────────────────┐  │
   Image ─► SRM │     FreqCNN      │──┘
           filters└────────────────┘
```

Each base model independently outputs a 2-class softmax (real / fake probability). Their concatenated outputs (6 values) feed into a small MLP that learns which model to trust for a given input, producing the final verdict.

---

## 📦 Requirements

- Python 3.9+
- PyTorch (with CUDA for GPU training, optional but recommended)
- `timm` (pretrained EfficientNet / Swin backbones)
- `torchvision`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `gradio`
- `tqdm`
- `numpy`, `Pillow`

Install everything with:

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn gradio tqdm numpy pillow
```

> 💡 For GPU training, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) that matches your CUDA version.

---

## 📁 Dataset Setup

Each dataset must contain **`real/`** and **`fake/`** subfolders (case-insensitive), either directly at the root or inside `train/`, `val/`, and `test/` split folders. Supported layouts are auto-detected:

- `full_split` — `train/`, `val/`, `test/` folders each with `real/` and `fake/`
- `train_test_only` — `train/` and `test/` only (val is carved out of train)
- `single_folder` — all images directly under `real/` and `fake/` (auto-split into train/val/test)

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

Edit the paths in **`config.py`**:

```python
DATA_ROOT = "path/to/your/dataset"        # primary training dataset

CROSS_DATASETS = {
    "Dataset2_CipLab": "path/to/other/dataset",   # for generalization testing
}

MULTI_TRAIN_DATASETS = [
    "path/to/dataset_a",
    "path/to/dataset_b",
]
```

---

## ⚙️ Configuration

All hyperparameters live in `config.py`:

| Parameter | Description | Default |
|---|---|---|
| `IMG_SIZE` | Input resolution | 224 |
| `BATCH_SIZE` | Training batch size | 32 |
| `EPOCHS_BASE` | Epochs for EfficientNet / Swin | 2 |
| `EPOCHS_FREQ` | Epochs for FreqCNN | 2 |
| `EPOCHS_ENS` | Epochs for the Ensemble MLP | 2 |
| `LR` / `LR_ENS` | Learning rates | 3e-4 / 1e-3 |
| `VAL_SPLIT` / `TEST_SPLIT` | Split ratios | 0.15 / 0.10 |
| `DEVICE` | Auto-detects CUDA, falls back to CPU | — |

> Default epoch counts are low for quick smoke-testing — increase `EPOCHS_BASE`, `EPOCHS_FREQ`, and `EPOCHS_ENS` for real training runs.

---

## 🚀 Usage

### 1. Train the full pipeline

```bash
python train.py
```

This runs the entire pipeline end to end:

1. Load and split the dataset
2. Train EfficientNet-B4
3. Train Swin-Tiny (with warmup + cosine LR schedule)
4. Train FreqCNN
5. Plot training curves
6. Train the Ensemble MLP on the base models' outputs
7. Run full test-set evaluation (accuracy, AUC-ROC, F1, confusion matrices, ROC curves)
8. Generate Grad-CAM visualizations on sample test images
9. Run cross-dataset evaluation (if `CROSS_DATASETS` is configured)
10. Run the multi-dataset training experiment (if `MULTI_TRAIN_DATASETS` has more than one path)

All checkpoints and plots are saved to `CFG.CKPT_DIR` (default: `checkpoints/`).

### 2. Launch the interactive demo

Once models are trained (checkpoints exist), launch the Gradio app:

```bash
python demo.py
```

Open the printed local URL (default `http://localhost:7860`) and upload any face image. The demo returns:

- A text verdict with per-model confidence breakdown
- A 6-panel visualization: original image, EfficientNet Grad-CAM overlay, raw heatmap, FreqCNN noise residual, Swin attention map, and an ensemble verdict card showing per-model agreement

---

## 📊 Outputs

Training produces the following artifacts in `checkpoints/`:

- `EfficientNet-B4.pth`, `Swin-Tiny.pth`, `FreqCNN.pth`, `ensemble.pth` — best model weights (selected by validation AUC)
- `training_curves.png` — loss / accuracy / AUC curves per model
- `evaluation_results.png` — accuracy, AUC, F1 bar charts, ROC curves, and confusion matrices
- `ensemble_gradcam.png` — combined Grad-CAM visualization on sample test images
- `cross_dataset_results.png` / `generalization_drop.png` — cross-dataset generalization comparisons (if configured)

---

## 🧠 How It Works

- **EfficientNet-B4** and **Swin-Tiny** use standard ImageNet-normalized RGB inputs and are fine-tuned from ImageNet-pretrained weights (via `timm`).
- **FreqCNN** first passes the raw image through a frozen **SRM filter bank** (5 hand-crafted high-pass kernels applied per channel) to extract noise residuals, then runs a lightweight CNN on top — this makes it sensitive to generator-specific frequency artifacts that are invisible in the spatial domain.
- The **Ensemble MLP** is trained *after* the three base models are frozen, using their concatenated softmax outputs as input features, and learns which model's predictions to weight more heavily.
- **Class imbalance** is handled via a `WeightedRandomSampler` during training.

---

## ⚠️ Notes & Limitations

- This project is intended for **research and educational purposes** — synthetic media detection is an evolving field and no model generalizes perfectly to unseen generators.
- Default epoch counts in `config.py` are intentionally small for quick pipeline verification; increase them for meaningful accuracy.
- `NUM_WORKERS = 0` is set for Windows compatibility; increase it on Linux/Mac for faster data loading.
- Grad-CAM and attention visualizations are diagnostic aids, not proof of manipulation — always interpret results with appropriate caution.



