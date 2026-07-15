"""
visualize.py — All plotting and Grad-CAM visualization utilities
"""

import io
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for VS Code
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve

from config import CFG


# ─────────────────────────────────────────────────────────────
# HELPERS 
# ─────────────────────────────────────────────────────────────

def unnormalize(tensor):
    """Convert normalized tensor back to displayable numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)\
                                      .permute(1, 2, 0).numpy()


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL image (for Gradio)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def save_fig(fig, filename):
    path = os.path.join(CFG.CKPT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x):
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        pred_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, pred_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu(
            (weights * self.activations).sum(1)).squeeze()
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_idx


def build_gradcam_hooks(eff_model, swin_model):
    """Attach Grad-CAM and attention hooks to models."""
    cam_eff = GradCAM(
        eff_model, eff_model.backbone.blocks[-1][-1])

    swin_attn = [None]
    def swin_hook(module, inp, out):
        swin_attn[0] = out.detach()
    try:
        swin_model.backbone.layers[-1].blocks[-1].attn\
            .register_forward_hook(swin_hook)
        print("  ✓ Swin attention hook attached")
    except Exception as e:
        print(f"  Swin hook unavailable: {e}")

    return cam_eff, swin_attn


# ─────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────

def plot_training_curves(histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d0d0d")
    colors = {"EfficientNet-B4": "#00e5ff",
              "Swin-Tiny":       "#e040fb",
              "FreqCNN":         "#69f0ae"}

    for name, hist in histories.items():
        c = colors.get(name, "white")
        ep = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(ep, hist["train_loss"], "--", color=c, alpha=0.5)
        axes[0].plot(ep, hist["val_loss"],   "-",  color=c, label=name)
        axes[1].plot(ep, hist["train_acc"],  "--", color=c, alpha=0.5)
        axes[1].plot(ep, hist["val_acc"],    "-",  color=c, label=name)
        axes[2].plot(ep, hist["val_auc"],    "-",  color=c, label=name)

    for ax, title in zip(axes,
        ["Loss (dashed=train)", "Accuracy", "Val AUC-ROC"]):
        ax.set_title(title, color="white")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(facecolor="#1a1a1a", labelcolor="white")

    fig.suptitle("Training Curves", color="white", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "training_curves.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────

def plot_evaluation(results):
    bar_colors = {"EfficientNet-B4": "#00e5ff",
                  "Swin-Tiny":       "#e040fb",
                  "FreqCNN":         "#69f0ae",
                  "Ensemble":        "#ff6e40"}

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0d0d0d")
    names  = list(results.keys())
    colors = [bar_colors[n] for n in names]

    def bar_ax(ax, vals, title, ylim=(0.5, 1.0)):
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylim(*ylim)
        ax.set_title(title, color="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom",
                    color="white", fontsize=9)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white", labelrotation=15)
        ax.spines[:].set_color("#333")

    bar_ax(axes[0,0], [r["acc"] for r in results.values()],
           "Accuracy")
    bar_ax(axes[0,1], [r["auc"] for r in results.values()],
           "AUC-ROC")
    bar_ax(axes[0,2], [r["f1"]  for r in results.values()],
           "F1-Score", ylim=(0, 1))

    ax_roc = axes[1, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"][:, 1])
        ax_roc.plot(fpr, tpr, color=bar_colors[name], lw=2,
                    label=f"{name} ({res['auc']:.3f})")
    ax_roc.plot([0,1],[0,1], color="#555", ls="--")
    ax_roc.set_title("ROC Curves", color="white")
    ax_roc.legend(facecolor="#1a1a1a", labelcolor="white",
                  fontsize=8)
    ax_roc.set_facecolor("#1a1a1a")
    ax_roc.tick_params(colors="white")
    ax_roc.spines[:].set_color("#333")

    for ax, (name, res) in zip(axes[1, 1:],
        [("Ensemble", results["Ensemble"]),
         ("EfficientNet-B4", results["EfficientNet-B4"])]):
        import seaborn as sns
        cm = __import__("sklearn.metrics",
                        fromlist=["confusion_matrix"]
                        ).confusion_matrix(
            res["labels"], res["probs"].argmax(1))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    ax=ax, xticklabels=CFG.CLASSES,
                    yticklabels=CFG.CLASSES)
        ax.set_title(f"CM — {name}", color="white")
        ax.tick_params(colors="white")
        ax.set_facecolor("#1a1a1a")

    fig.suptitle("Full Evaluation — All Models",
                 color="white", fontsize=15)
    plt.tight_layout()
    save_fig(fig, "evaluation_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# ENSEMBLE GRAD-CAM (static test set)
# ─────────────────────────────────────────────────────────────

def visualize_ensemble_gradcam(eff, swin, freq, ensemble,
                                cam_eff, swin_attn,
                                test_sp, test_fr,
                                n_images=6):
    device = CFG.DEVICE
    imgs_sp, imgs_fr, labels = [], [], []
    for (isp, lbl), (ifr, _) in zip(test_sp, test_fr):
        imgs_sp.append(isp); imgs_fr.append(ifr)
        labels.append(lbl)
        if sum(len(x) for x in imgs_sp) >= n_images:
            break

    imgs_sp = torch.cat(imgs_sp)[:n_images]
    imgs_fr = torch.cat(imgs_fr)[:n_images]
    labels  = torch.cat(labels)[:n_images]

    fig, axes = plt.subplots(n_images, 5,
                             figsize=(22, n_images * 4.5))
    fig.patch.set_facecolor("#0d0d0d")

    for col, title in enumerate(
        ["Original", "EfficientNet\nGrad-CAM",
         "Heatmap", "FreqCNN\nResidual", "Ensemble\nVerdict"]):
        axes[0, col].set_title(title, color="white",
                               fontsize=11, fontweight="bold")

    for i in range(n_images):
        true_lbl = CFG.CLASSES[labels[i].item()]
        isp      = imgs_sp[i].unsqueeze(0).to(device)
        ifr      = imgs_fr[i].unsqueeze(0).to(device)
        orig     = unnormalize(imgs_sp[i])

        cam_map, eff_pred = cam_eff.generate(isp)
        cam_r = np.array(
            Image.fromarray((cam_map * 255).astype(np.uint8)
                            ).resize((224, 224), Image.BILINEAR)
        ) / 255.0
        hmap    = plt.cm.jet(cam_r)[..., :3]
        overlay = (orig * 0.45 + hmap * 0.55).clip(0, 1)

        with torch.no_grad():
            p_eff  = F.softmax(eff(isp),  dim=1).cpu().numpy()[0]
            p_swin = F.softmax(swin(isp), dim=1).cpu().numpy()[0]
            p_freq = F.softmax(freq(ifr), dim=1).cpu().numpy()[0]
            resids = freq.get_residuals(ifr)
            X_ens  = torch.FloatTensor(
                np.concatenate([p_eff, p_swin, p_freq])
            ).unsqueeze(0).to(device)
            p_ens  = F.softmax(
                ensemble(X_ens), dim=1).cpu().numpy()[0]

        ens_pred = CFG.CLASSES[p_ens.argmax()]
        ens_conf = p_ens.max() * 100
        correct  = ens_pred == true_lbl
        vc       = "#69f0ae" if correct else "#ff5252"

        axes[i,0].imshow(orig)
        axes[i,0].set_ylabel(
            f"Img {i+1} | True: {true_lbl}",
            color="white", fontsize=9)

        axes[i,1].imshow(overlay)
        c1 = ("#69f0ae"
              if CFG.CLASSES[eff_pred] == true_lbl
              else "#ff5252")
        axes[i,1].set_xlabel(
            f"Eff: {CFG.CLASSES[eff_pred]} "
            f"({p_eff.max()*100:.0f}%)",
            color=c1, fontsize=9)

        axes[i,2].imshow(cam_r, cmap="jet")
        py, px = np.unravel_index(
            np.argmax(cam_r), cam_r.shape)
        axes[i,2].annotate(
            "peak", xy=(px, py),
            xytext=(px+25, py-25),
            color="white", fontsize=7,
            arrowprops=dict(
                arrowstyle="->", color="white", lw=1))

        resid_avg  = resids[0, :3].mean(0).cpu().numpy()
        resid_norm = ((resid_avg - resid_avg.min()) /
                      (resid_avg.max() - resid_avg.min() + 1e-8))
        axes[i,3].imshow(resid_norm, cmap="RdBu_r")
        c3 = ("#69f0ae"
              if CFG.CLASSES[p_freq.argmax()] == true_lbl
              else "#ff5252")
        axes[i,3].set_xlabel(
            f"Freq: {CFG.CLASSES[p_freq.argmax()]} "
            f"({p_freq.max()*100:.0f}%)",
            color=c3, fontsize=9)

        ax = axes[i, 4]
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.set_facecolor("#111111")
        bg = "#2d0a0a" if ens_pred == "fake" else "#0a2d0a"
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05,0.68), 0.90, 0.26,
            boxstyle="round,pad=0.02",
            facecolor=bg, edgecolor=vc, linewidth=2.5,
            transform=ax.transAxes))
        ax.text(0.5, 0.82,
                f"{'✓' if correct else '✗'} {ens_pred.upper()}",
                transform=ax.transAxes, ha="center",
                color=vc, fontsize=15, fontweight="bold")
        ax.text(0.5, 0.71, f"{ens_conf:.1f}% conf",
                transform=ax.transAxes, ha="center",
                color="white", fontsize=9)

        for j, (cls, prob, col) in enumerate(zip(
            ["Real","Fake"], p_ens, ["#00e5ff","#ff4444"])):
            y = 0.52 - j * 0.18
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.05, y), 0.90 * prob, 0.10,
                boxstyle="round,pad=0.01",
                facecolor=col, alpha=0.85,
                transform=ax.transAxes))
            ax.text(0.05, y + 0.115,
                    f"{cls}: {prob*100:.1f}%",
                    transform=ax.transAxes,
                    color="white", fontsize=8, va="bottom")

        for m_idx, (mn, mp) in enumerate(zip(
            ["Eff","Swin","Freq"],
            [p_eff.argmax() == p_ens.argmax(),
             p_swin.argmax() == p_ens.argmax(),
             p_freq.argmax() == p_ens.argmax()])):
            dc = "#69f0ae" if mp else "#ff5252"
            ax.text(0.15 + m_idx*0.30, 0.22,
                    "●" if mp else "○",
                    transform=ax.transAxes,
                    ha="center", color=dc, fontsize=14)
            ax.text(0.15 + m_idx*0.30, 0.12, mn,
                    transform=ax.transAxes,
                    ha="center", color="white", fontsize=8)

        for col in range(5):
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])
            axes[i, col].set_facecolor("#0d0d0d")
            for sp in axes[i, col].spines.values():
                sp.set_edgecolor("#333")

    fig.suptitle(
        "Ensemble Deepfake Detection — Combined Grad-CAM",
        color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    save_fig(fig, "ensemble_gradcam.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# CROSS-DATASET PLOTS
# ─────────────────────────────────────────────────────────────

def plot_cross_dataset_results(all_results):
    if not all_results:
        print("  No cross-dataset results to plot.")
        return

    model_names  = ["EfficientNet-B4","Swin-Tiny",
                    "FreqCNN","Ensemble"]
    model_colors = {"EfficientNet-B4":"#00e5ff",
                    "Swin-Tiny":"#e040fb",
                    "FreqCNN":"#69f0ae",
                    "Ensemble":"#ff6e40"}
    ds_names = list(all_results.keys())
    x        = np.arange(len(ds_names))
    width    = 0.18
    offsets  = [-1.5,-0.5,0.5,1.5]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, metric, title in zip(
        axes, ["acc","auc","f1"],
        ["Accuracy","AUC-ROC","F1-Score"]):
        for i, mname in enumerate(model_names):
            vals = [
                all_results[ds].get(mname, {}).get(metric, 0)
                for ds in ds_names]
            bars = ax.bar(x + offsets[i] * width, vals,
                          width=width, label=mname,
                          color=model_colors[mname], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        v + 0.008, f"{v:.2f}",
                        ha="center", va="bottom",
                        color="white", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names, rotation=25,
                           ha="right", color="white", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, color="white")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.legend(facecolor="#1a1a1a", labelcolor="white",
                  fontsize=8)

    fig.suptitle("Cross-Dataset Generalization",
                 color="white", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "cross_dataset_results.png")
    plt.show()


def plot_generalization_drop(in_dist_results, cross_results):
    if not cross_results or "Ensemble" not in in_dist_results:
        return

    model_names  = ["EfficientNet-B4","Swin-Tiny",
                    "FreqCNN","Ensemble"]
    model_colors = {"EfficientNet-B4":"#00e5ff",
                    "Swin-Tiny":"#e040fb",
                    "FreqCNN":"#69f0ae",
                    "Ensemble":"#ff6e40"}
    ds_names = list(cross_results.keys())

    fig, axes = plt.subplots(
        1, len(ds_names),
        figsize=(6 * len(ds_names), 6), sharey=True)
    if len(ds_names) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0d0d0d")

    for ax, ds_name in zip(axes, ds_names):
        ds_res = cross_results[ds_name]
        names, drops, colors = [], [], []
        for mname in model_names:
            if (mname not in in_dist_results or
                    mname not in ds_res):
                continue
            drop = (in_dist_results[mname]["acc"] -
                    ds_res[mname]["acc"])
            names.append(mname.replace("-","\n"))
            drops.append(drop)
            colors.append(model_colors[mname])

        bars = ax.bar(names, drops, color=colors, alpha=0.85)
        for bar, v in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + 0.005, f"-{v*100:.1f}%",
                    ha="center", va="bottom",
                    color="white", fontsize=10,
                    fontweight="bold")
        ax.set_title(f"Drop → {ds_name}", color="white")
        ax.set_ylabel("Accuracy Drop", color="white")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333")
        ax.axhline(0, color="#555", ls="--", lw=1)

    fig.suptitle(
        "Generalization Drop  (lower = more robust)",
        color="white", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "generalization_drop.png")
    plt.show()