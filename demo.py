"""
demo.py — Gradio web demo with live Grad-CAM visualization 
Run: python demo.py
"""

import os
os.environ.pop("SSL_CERT_FILE", None)   # remove broken cert path
import io
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from torchvision import transforms
from PIL import Image

from config import CFG
from models import load_all_models
from visualize import GradCAM, fig_to_pil


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────

sp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
fr_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

print("[Demo] Loading models from checkpoints...")
device = CFG.DEVICE
eff, swin, freq, ensemble = load_all_models(device)

# Attach Grad-CAM hook
cam_eff = GradCAM(eff, eff.backbone.blocks[-1][-1])

# Swin attention hook
swin_attn = [None]
def swin_hook(module, inp, out):
    swin_attn[0] = out.detach()
try:
    swin.backbone.layers[-1].blocks[-1].attn\
        .register_forward_hook(swin_hook)
    print("[Demo] Swin attention hook attached")
except Exception as e:
    print(f"[Demo] Swin hook: {e}")

print(f"[Demo] Ready on device: {device}")


# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────

def predict_with_gradcam(pil_img):
    if pil_img is None:
        return "Please upload an image.", None

    img_rgb = pil_img.convert("RGB")
    img_sp  = sp_transform(img_rgb).unsqueeze(0).to(device)
    img_fr  = fr_transform(img_rgb).unsqueeze(0).to(device)
    orig_np = np.array(img_rgb.resize((224, 224))) / 255.0

    # Grad-CAM (needs grad enabled)
    cam_map, _ = cam_eff.generate(img_sp)

    with torch.no_grad():
        p_eff  = F.softmax(eff(img_sp),  dim=1).cpu().numpy()[0]
        p_swin = F.softmax(swin(img_sp), dim=1).cpu().numpy()[0]
        p_freq = F.softmax(freq(img_fr), dim=1).cpu().numpy()[0]
        resids = freq.get_residuals(img_fr)
        X_ens  = torch.FloatTensor(
            np.concatenate([p_eff, p_swin, p_freq])
        ).unsqueeze(0).to(device)
        p_ens  = F.softmax(ensemble(X_ens), dim=1).cpu().numpy()[0]

    ens_pred = "FAKE" if p_ens[1] > 0.5 else "REAL"
    ens_conf = max(p_ens) * 100
    verdict  = "🔴  FAKE" if p_ens[1] > 0.5 else "🟢  REAL"
    v_color  = "#ff4444" if ens_pred == "FAKE" else "#44ff88"

    # ── Text result ──────────────────────────────────────────
    text_result = f"""
{'='*44}
  {verdict}
  Confidence : {ens_conf:.1f}%
{'='*44}

  Ensemble     →  Real: {p_ens[0]*100:.1f}%   Fake: {p_ens[1]*100:.1f}%
  ────────────────────────────────────────────
  EfficientNet →  Real: {p_eff[0]*100:.1f}%   Fake: {p_eff[1]*100:.1f}%
  Swin-Tiny    →  Real: {p_swin[0]*100:.1f}%   Fake: {p_swin[1]*100:.1f}%
  FreqCNN      →  Real: {p_freq[0]*100:.1f}%   Fake: {p_freq[1]*100:.1f}%

  Red/orange = suspicious regions
  Blue residual = normal  |  Red residual = artifact
"""

    # ── Visualization ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0d0d0d")

    # Panel 1: Original
    axes[0,0].imshow(orig_np)
    axes[0,0].set_title("Original Image",
                        color="white", fontsize=13)

    # Panel 2: Grad-CAM overlay
    cam_resized = np.array(
        Image.fromarray(
            (cam_map * 255).astype(np.uint8)
        ).resize((224, 224), Image.BILINEAR)
    ) / 255.0
    heatmap = plt.cm.jet(cam_resized)[..., :3]
    overlay = (orig_np * 0.45 + heatmap * 0.55).clip(0, 1)
    axes[0,1].imshow(overlay)
    axes[0,1].set_title(
        f"EfficientNet Grad-CAM\n"
        f"Pred: {'FAKE' if p_eff[1]>0.5 else 'REAL'} "
        f"({max(p_eff)*100:.1f}%)",
        color="#00e5ff", fontsize=11)
    sm = plt.cm.ScalarMappable(
        cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[0,1],
                        fraction=0.046, pad=0.04)
    cbar.set_label("Low → High Activation",
                   color="white", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(),
             color="white", fontsize=7)

    # Panel 3: Heatmap only
    axes[0,2].imshow(cam_resized, cmap="jet")
    axes[0,2].set_title(
        "Heatmap Only\n🔴 Red = Most suspicious",
        color="white", fontsize=11)
    py, px = np.unravel_index(
        np.argmax(cam_resized), cam_resized.shape)
    axes[0,2].annotate(
        "Peak activation",
        xy=(px, py), xytext=(px+30, py-30),
        color="white", fontsize=8,
        arrowprops=dict(
            arrowstyle="->", color="white", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#333", edgecolor="white",
                  alpha=0.8))

    # Panel 4: FreqCNN residual
    resid_avg  = resids[0, :3].mean(0).cpu().numpy()
    resid_norm = ((resid_avg - resid_avg.min()) /
                  (resid_avg.max() - resid_avg.min() + 1e-8))
    axes[1,0].imshow(resid_norm, cmap="RdBu_r", vmin=0, vmax=1)
    axes[1,0].set_title(
        f"FreqCNN Noise Residual\n"
        f"Pred: {'FAKE' if p_freq[1]>0.5 else 'REAL'} "
        f"({max(p_freq)*100:.1f}%)",
        color="#69f0ae", fontsize=11)

    # Panel 5: Swin attention
    axes[1,1].set_facecolor("#0d0d0d")
    if swin_attn[0] is not None:
        try:
            attn     = swin_attn[0]
            avg_attn = attn[0].mean(0).mean(0).cpu().numpy()
            side     = int(len(avg_attn) ** 0.5)
            attn_img = avg_attn[:side*side].reshape(side, side)
            attn_img = ((attn_img - attn_img.min()) /
                        (attn_img.max() - attn_img.min() + 1e-8))
            axes[1,1].imshow(attn_img, cmap="inferno")
            axes[1,1].set_title(
                f"Swin Attention Map\n"
                f"Pred: {'FAKE' if p_swin[1]>0.5 else 'REAL'} "
                f"({max(p_swin)*100:.1f}%)",
                color="#e040fb", fontsize=11)
        except Exception:
            axes[1,1].text(0.5, 0.5, "Attention N/A",
                          color="white", ha="center",
                          va="center",
                          transform=axes[1,1].transAxes)
            axes[1,1].set_title("Swin-Tiny", color="#e040fb")
    else:
        axes[1,1].text(0.5, 0.5, "Attention N/A",
                      color="white", ha="center", va="center",
                      transform=axes[1,1].transAxes)
        axes[1,1].set_title("Swin-Tiny", color="#e040fb")

    # Panel 6: Verdict card
    ax = axes[1, 2]
    ax.set_facecolor("#111111")
    ax.axis("off")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    badge_bg = "#2d0a0a" if ens_pred == "FAKE" else "#0a2d0a"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05,0.70), 0.90, 0.24,
        boxstyle="round,pad=0.02",
        facecolor=badge_bg, edgecolor=v_color,
        linewidth=3, transform=ax.transAxes))
    ax.text(0.5, 0.83, verdict,
            transform=ax.transAxes, ha="center",
            color=v_color, fontsize=18, fontweight="bold")
    ax.text(0.5, 0.73, f"Confidence: {ens_conf:.1f}%",
            transform=ax.transAxes, ha="center",
            color="white", fontsize=10)

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
                color="white", fontsize=9, va="bottom")

    for m_idx, (mn, mp) in enumerate(zip(
        ["Eff","Swin","Freq"],
        [p_eff.argmax() == p_ens.argmax(),
         p_swin.argmax() == p_ens.argmax(),
         p_freq.argmax() == p_ens.argmax()])):
        dc = "#69f0ae" if mp else "#ff5252"
        ax.text(0.15 + m_idx*0.30, 0.22,
                "●" if mp else "○",
                transform=ax.transAxes,
                ha="center", color=dc, fontsize=16)
        ax.text(0.15 + m_idx*0.30, 0.12, mn,
                transform=ax.transAxes,
                ha="center", color="white", fontsize=9)
    ax.text(0.5, 0.04, "● agrees  ○ disagrees",
            transform=ax.transAxes, ha="center",
            color="#888", fontsize=7)
    ax.set_title("Ensemble Verdict",
                color="white", fontsize=13)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#0d0d0d")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    fig.suptitle(
        f"Synthetic Image Detection  —  {verdict}  "
        f"({ens_conf:.1f}%)",
        color=v_color, fontsize=14,
        fontweight="bold", y=1.01)
    plt.tight_layout()
    vis_image = fig_to_pil(fig)
    return text_result, vis_image


# ─────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict_with_gradcam,
        inputs=gr.Image(
            type="pil",
            label="Upload Image (face or AI-generated)"
        ),
        outputs=[
            gr.Textbox(label="Detection Result", lines=13),
            gr.Image(
                label="Grad-CAM Visualization",
                type="pil")
        ],
        title="🔍 Synthetic / AI-Generated Image Detector with Grad-CAM",
        description=(
            "Upload any image to detect if it is real or AI-generated.\n"
            "• Red/orange regions = where the model found suspicious artifacts\n"
            "• Blue residual = normal noise  |  Red residual = GAN artifacts\n"
            "• Verdict card shows per-model agreement with the ensemble"
        ),
        theme=gr.themes.Soft(),
        flagging_mode="never"
    )

    print(f"\n[Demo] Launching on http://localhost:{CFG.GRADIO_PORT}")
    print("[Demo] Press Ctrl+C to stop\n")

    demo.launch(
        server_port=CFG.GRADIO_PORT,
        share=CFG.GRADIO_SHARE,
        inbrowser=True
    )