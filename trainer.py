"""
trainer.py — Training and evaluation engine for all models
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)

from config import CFG
from models import EnsembleMLP, save_model


# ─────────────────────────────────────────────────────────────
# WARMUP + CO SINE SCHEDULER (for Swin-Tiny)
# ─────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs,
                 total_epochs, base_lr, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.epoch         = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            progress = ((e - self.warmup_epochs) /
                        (self.total_epochs - self.warmup_epochs))
            lr = (self.min_lr +
                  0.5 * (self.base_lr - self.min_lr) *
                  (1 + np.cos(np.pi * progress)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────
# ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False,
                ncols=80, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = F.softmax(logits, dim=1)
        correct += (probs.argmax(1) == labels).sum().item()
        total   += imgs.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    f1  = f1_score(all_labels, all_probs.argmax(1), zero_division=0)
    return (total_loss / total, correct / total,
            auc, f1, all_probs, all_labels)


# ─────────────────────────────────────────────────────────────
# TRAIN BASE MODEL
# ─────────────────────────────────────────────────────────────

def train_base_model(model, name, train_loader, val_loader,
                     epochs, lr=CFG.LR, use_warmup=False):
    device    = CFG.DEVICE
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.10 if use_warmup else 0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=CFG.WEIGHT_DECAY)

    if use_warmup:
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=3,
            total_epochs=epochs, base_lr=lr,
            min_lr=lr * 0.01)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_auc, best_state = 0.0, None
    history = {k: [] for k in
               ["train_loss", "val_loss",
                "train_acc",  "val_acc",
                "val_auc",    "val_f1"]}

    print(f"\n{'='*60}")
    print(f"  Training: {name}  |  {epochs} epochs  "
          f"|  device={device}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc, vl_f1, _, _ = evaluate(
            model, val_loader, criterion, device)

        if use_warmup:
            scheduler.step()
        else:
            scheduler.step()

        for k, v in zip(history.keys(),
                        [tr_loss, vl_loss, tr_acc,
                         vl_acc, vl_auc, vl_f1]):
            history[k].append(v)

        tag = ""
        if vl_auc > best_auc:
            best_auc   = vl_auc
            best_state = copy.deepcopy(model.state_dict())
            save_model(model, name)
            tag = "  ← best"

        elapsed = time.time() - t0
        print(f"  Ep {epoch:02d}/{epochs} | "
              f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.4f} | "
              f"VlLoss={vl_loss:.4f} VlAcc={vl_acc:.4f} "
              f"AUC={vl_auc:.4f} F1={vl_f1:.4f} | "
              f"{elapsed:.0f}s{tag}")

    model.load_state_dict(best_state)
    print(f"\n  ✓ Best Val AUC: {best_auc:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────
# COLLECT BASE MODEL OUTPUTS (for ensemble)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(models_loaders, device):
    """
    models_loaders: list of (model, loader) tuples
    Returns X (N, num_models*2) and y (N,)
    """
    all_probs_list = [[] for _ in models_loaders]
    all_labels     = []
    loaders        = [ml[1] for ml in models_loaders]

    for batches in zip(*loaders):
        all_labels.append(batches[0][1].numpy())
        for i, (model, _) in enumerate(models_loaders):
            imgs = batches[i][0].to(device)
            model.eval()
            probs = F.softmax(model(imgs), dim=1).cpu().numpy()
            all_probs_list[i].append(probs)

    X = np.concatenate(
        [np.concatenate(p) for p in all_probs_list], axis=1)
    y = np.concatenate(all_labels)
    return X, y


# ─────────────────────────────────────────────────────────────
# TRAIN ENSEMBLE MLP
# ─────────────────────────────────────────────────────────────

def train_ensemble(eff, swin, freq,
                   train_sp, train_fr, val_sp, val_fr):
    device = CFG.DEVICE
    print("\n[Collecting base model outputs...]")

    X_train, y_train = collect_probs(
        [(eff, train_sp), (swin, train_sp), (freq, train_fr)],
        device)
    X_val, y_val = collect_probs(
        [(eff, val_sp), (swin, val_sp), (freq, val_fr)],
        device)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.LongTensor(y_val).to(device)

    ensemble  = EnsembleMLP(in_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        ensemble.parameters(), lr=CFG.LR_ENS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS_ENS,
        eta_min=CFG.LR_ENS * 0.01)

    best_auc, best_state = 0.0, None
    print(f"\n{'='*60}")
    print(f"  Ensemble MLP Training  |  {CFG.EPOCHS_ENS} epochs")
    print(f"{'='*60}")

    for epoch in range(1, CFG.EPOCHS_ENS + 1):
        ensemble.train()
        optimizer.zero_grad()
        loss = criterion(ensemble(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        ensemble.eval()
        with torch.no_grad():
            vp   = F.softmax(ensemble(X_vl), dim=1).cpu().numpy()
            vacc = accuracy_score(y_val, vp.argmax(1))
            vauc = roc_auc_score(y_val, vp[:, 1])
            vf1  = f1_score(y_val, vp.argmax(1), zero_division=0)

        tag = ""
        if vauc > best_auc:
            best_auc   = vauc
            best_state = copy.deepcopy(ensemble.state_dict())
            save_model(ensemble, "ensemble")
            tag = "  ← best"

        print(f"  Ep {epoch:02d}/{CFG.EPOCHS_ENS} | "
              f"Loss={loss.item():.4f} "
              f"VlAcc={vacc:.4f} AUC={vauc:.4f} "
              f"F1={vf1:.4f}{tag}")

    ensemble.load_state_dict(best_state)
    print(f"\n  ✓ Best Ensemble AUC: {best_auc:.4f}")
    return ensemble


# ─────────────────────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────────────────────

def full_evaluation(eff, swin, freq, ensemble,
                    test_sp, test_fr):
    device    = CFG.DEVICE
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*60}")
    print("  FULL TEST SET EVALUATION")
    print(f"{'='*60}")

    results = {}
    for name, model, loader in [
        ("EfficientNet-B4", eff,  test_sp),
        ("Swin-Tiny",       swin, test_sp),
        ("FreqCNN",         freq, test_fr),
    ]:
        _, acc, auc, f1, probs, labels = evaluate(
            model, loader, criterion, device)
        results[name] = {
            "acc": acc, "auc": auc, "f1": f1,
            "probs": probs, "labels": labels
        }
        print(f"  {name:20s} | "
              f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

    # Ensemble
    X_test, y_test = collect_probs(
        [(eff, test_sp), (swin, test_sp), (freq, test_fr)],
        device)
    X_t = torch.FloatTensor(X_test).to(device)
    ensemble.eval()
    with torch.no_grad():
        ep = F.softmax(ensemble(X_t), dim=1).cpu().numpy()
    ens_acc = accuracy_score(y_test, ep.argmax(1))
    ens_auc = roc_auc_score(y_test, ep[:, 1])
    ens_f1  = f1_score(y_test, ep.argmax(1), zero_division=0)
    results["Ensemble"] = {
        "acc": ens_acc, "auc": ens_auc, "f1": ens_f1,
        "probs": ep, "labels": y_test
    }
    print(f"\n  {'Ensemble':20s} | "
          f"Acc={ens_acc:.4f}  AUC={ens_auc:.4f}  "
          f"F1={ens_f1:.4f}  ← BEST")
    print(f"\n  Classification Report (Ensemble):\n")
    print(classification_report(
        y_test, ep.argmax(1), target_names=CFG.CLASSES))
    return results


# ─────────────────────────────────────────────────────────────
# CROSS-DATASET EVALUATION
# ─────────────────────────────────────────────────────────────

def cross_dataset_evaluation(eff, swin, freq, ensemble,
                              cross_datasets):
    from dataset import make_cross_loader
    device = CFG.DEVICE

    print(f"\n{'='*60}")
    print("  CROSS-DATASET GENERALIZATION EVALUATION")
    print(f"{'='*60}")

    all_results = {}

    for ds_name, ds_path in cross_datasets.items():
        if not os.path.isdir(ds_path):
            print(f"\n  [SKIP] {ds_name} — not found: {ds_path}")
            continue

        print(f"\n  Dataset : {ds_name}")
        try:
            sp_loader, fr_loader = make_cross_loader(ds_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        ds_res = {}
        for name, model, loader in [
            ("EfficientNet-B4", eff,  sp_loader),
            ("Swin-Tiny",       swin, sp_loader),
            ("FreqCNN",         freq, fr_loader),
        ]:
            criterion = nn.CrossEntropyLoss()
            _, acc, auc, f1, probs, labels = evaluate(
                model, loader, criterion, device)
            ds_res[name] = {
                "acc": acc, "auc": auc, "f1": f1,
                "probs": probs, "labels": labels
            }
            print(f"    {name:20s} | "
                  f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")

        # Ensemble
        p_eff  = ds_res["EfficientNet-B4"]["probs"]
        p_swin = ds_res["Swin-Tiny"]["probs"]
        p_freq = ds_res["FreqCNN"]["probs"]
        labels = ds_res["EfficientNet-B4"]["labels"]
        X_ens  = torch.FloatTensor(
            np.concatenate([p_eff, p_swin, p_freq], axis=1)
        ).to(device)
        ensemble.eval()
        with torch.no_grad():
            ep = F.softmax(ensemble(X_ens), dim=1).cpu().numpy()
        ens_acc = accuracy_score(labels, ep.argmax(1))
        ens_auc = roc_auc_score(labels, ep[:, 1])
        ens_f1  = f1_score(labels, ep.argmax(1), zero_division=0)
        ds_res["Ensemble"] = {
            "acc": ens_acc, "auc": ens_auc, "f1": ens_f1,
            "probs": ep, "labels": labels
        }
        print(f"    {'Ensemble':20s} | "
              f"Acc={ens_acc:.4f}  AUC={ens_auc:.4f}  "
              f"F1={ens_f1:.4f}  ← best")
        all_results[ds_name] = ds_res

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY — Ensemble Accuracy Across Datasets")
    print(f"{'='*60}")
    print(f"  {'Dataset':<28} {'Acc':>7} {'AUC':>7} {'F1':>7}")
    print(f"  {'-'*49}")
    for ds_name, ds_res in all_results.items():
        r = ds_res["Ensemble"]
        print(f"  {ds_name:<28} "
              f"{r['acc']:>7.4f} {r['auc']:>7.4f} {r['f1']:>7.4f}")

    return all_results


# ─────────────────────────────────────────────────────────────
# Fix missing import
# ─────────────────────────────────────────────────────────────
import os