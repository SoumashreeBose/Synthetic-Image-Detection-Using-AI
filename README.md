# Synthetic-Image-Detection-Using-AI

## 📌 Overview
This project implements a **deepfake / AI-generated image detection system** using a **multi-model ensemble approach**. It combines CNN, Transformer, and Frequency-domain models to achieve high accuracy and robustness.

The system detects fake images by analyzing:
- Visual patterns (CNN)
- Global context (Transformer)
- Hidden noise artifacts (Frequency analysis)

---

## 🚀 Key Idea

Deepfake images may look real but contain:
- Subtle spatial inconsistencies
- Global structural anomalies
- High-frequency noise patterns

This project combines all three using:
- **EfficientNet-B4** → Spatial features  
- **Swin Transformer** → Global + local attention  
- **FreqCNN (SRM filters)** → Noise residual detection  

---

## 🏗️ Architecture

### 🔹 Base Models

| Model | Type | Role |
|------|------|------|
| EfficientNet-B4 | CNN | Extract visual features |
| Swin-Tiny | Transformer | Capture global dependencies |
| FreqCNN | Frequency CNN | Detect hidden artifacts |

---

### 🔹 Ensemble Model

- Input: Probabilities from all base models (6 features)
- Model: Shallow MLP
- Output: Final prediction (Real / Fake)

---

## 🧪 Pipeline

1. **Data Loading**
   - Folder structure:
     ```
     dataset/
       real/
       fake/
     ```

2. **Preprocessing**
   - Resize, normalize
   - Data augmentation (flip, crop, color jitter)

3. **Training Base Models**
   - EfficientNet-B4
   - Swin-Tiny
   - FreqCNN

4. **Ensemble Training**
   - Combine predictions from all models
   - Train MLP for final output

5. **Evaluation**
   - Accuracy
   - AUC-ROC
   - F1 Score
   - Confusion Matrix

6. **Visualization**
   - Training curves
   - Performance graphs
   - Grad-CAM heatmaps

---

## 📂 Project Structure

```bash
project/
│
├── dataset/
│   ├── real/
│   └── fake/
│
├── checkpoints/
│   ├── EfficientNet-B4.pth
│   ├── Swin-Tiny.pth
│   ├── FreqCNN.pth
│   └── ensemble.pth
│
├── evaluation_results.png
├── gradcam_results.png
├── training_curves.png
│
└── main.py
```

---

## ▶️ How to Run

```bash
pip install timm torch torchvision scikit-learn matplotlib seaborn gradio
python main.py
```
