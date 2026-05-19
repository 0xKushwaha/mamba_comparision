"""
cka_analysis.py — Centered Kernel Alignment (CKA) feature similarity analysis.

Measures how similar the feature representations are between:
  - Different model variants at the same layer
  - Same model before vs after the sequence processing block
  - CNN stem output vs final representation across models

CKA value ∈ [0, 1]:
  0 = completely different representations
  1 = identical representations (up to linear transformation)

Generates a heatmap showing which models learn similar features —
a key figure for the paper.

Usage:
    # Compare all trained models on a dataset
    python cka_analysis.py \
        --dataset rice \
        --data_path /path/to/Rice_Leaf_AUG \
        --ckpt_dir outputs/rice \
        --n_pool 3

    # Single model internal CKA (CNN stem vs sequence output)
    python cka_analysis.py \
        --dataset cifar100 --data_path ./data \
        --ckpt_dir outputs/cifar100 \
        --n_pool 2 --mode internal
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import build_model, DISPLAY_NAMES, MODEL_REGISTRY
from data import get_loaders, DATASET_INFO


# ============================================================
# CKA IMPLEMENTATION
# ============================================================

def center_gram(K: torch.Tensor) -> torch.Tensor:
    """Column/row-center a gram matrix."""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA between two feature matrices X [n, d1] and Y [n, d2].
    Uses the HSIC formulation with linear kernels.
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    dot_product = torch.norm(X.T @ Y, p="fro") ** 2
    norm_X      = torch.norm(X.T @ X, p="fro")
    norm_Y      = torch.norm(Y.T @ Y, p="fro")

    denom = norm_X * norm_Y
    if denom < 1e-10:
        return 0.0
    return (dot_product / denom).item()


# ============================================================
# FEATURE EXTRACTION HOOKS
# ============================================================

class FeatureExtractor:
    """
    Registers forward hooks to capture intermediate feature maps.
    Accumulates features across batches, then returns stacked tensor.
    """
    def __init__(self):
        self._features = []
        self._handle   = None

    def register(self, module: nn.Module):
        self._features = []
        self._handle   = module.register_forward_hook(self._hook)
        return self

    def _hook(self, module, input, output):
        # Flatten spatial dims and detach from computation graph
        if output.dim() == 4:                        # [B, C, H, W] — CNN feature map
            feat = output.flatten(1)                 # [B, C*H*W] — too large; use gap
            feat = output.mean(dim=[2, 3])           # [B, C]
        elif output.dim() == 3:                      # [B, L, d] — sequence output
            feat = output.mean(dim=1)                # [B, d]
        elif output.dim() == 2:                      # [B, d] — already pooled
            feat = output
        else:
            feat = output.flatten(1)
        self._features.append(feat.detach().cpu().float())

    def collect(self) -> torch.Tensor:
        return torch.cat(self._features, dim=0)      # [N, d]

    def remove(self):
        if self._handle:
            self._handle.remove()


def extract_features(model, hook_module, loader, device, max_batches=20) -> torch.Tensor:
    """Run inference and collect features from a hooked module."""
    extractor = FeatureExtractor()
    extractor.register(hook_module)

    model.eval()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= max_batches:
                break
            _ = model(imgs.to(device))

    extractor.remove()
    return extractor.collect()


# ============================================================
# CROSS-MODEL CKA MATRIX
# ============================================================

def compute_cross_model_cka(models_dict: dict, hook_getter,
                             loader, device, max_batches=20) -> np.ndarray:
    """
    Compute pairwise CKA between the representations of multiple models.

    Args:
        models_dict : {name: model}
        hook_getter : callable(model) → nn.Module to hook
        loader      : data loader

    Returns:
        n×n numpy array of CKA values
    """
    names    = list(models_dict.keys())
    features = {}

    for name, model in models_dict.items():
        hook_module = hook_getter(model)
        feat = extract_features(model, hook_module, loader, device, max_batches)
        features[name] = feat
        print(f"  Extracted features for {name}: {feat.shape}")

    n   = len(names)
    cka = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            val = linear_cka(features[ni], features[nj])
            cka[i, j] = val

    return cka, names


# ============================================================
# PLOT HELPERS
# ============================================================

def plot_cka_heatmap(cka_matrix: np.ndarray, labels: list,
                     title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    mask = np.zeros_like(cka_matrix, dtype=bool)
    # Show full matrix (not masked) — off-diagonal shows cross-model similarity

    sns.heatmap(cka_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 10})
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_internal_cka(stem_feats: torch.Tensor, seq_feats: torch.Tensor,
                      model_names: list, title: str, save_path: str):
    """
    Bar chart: for each model, how much do the CNN stem features change
    after passing through the sequence block?
    CKA(stem, output) ≈ 1 → sequence block barely changes representations
    CKA(stem, output) ≈ 0 → sequence block transforms representations significantly
    """
    vals = [linear_cka(stem_feats[n], seq_feats[n]) for n in model_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3" if v > 0.7 else "#4CAF50" if v > 0.4 else "#F44336" for v in vals]
    bars = ax.bar(model_names, vals, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=1, label="High similarity (≥0.9)")
    ax.set_ylabel("CKA(stem features, sequence output)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_trained_models(ckpt_dir: str, n_pool: int, d_model: int,
                        n_blocks: int, num_classes: int,
                        d_state: int = 16, n_heads: int = 4) -> dict:
    """
    Loads best checkpoint for each model variant from ckpt_dir.
    Expected path: ckpt_dir/<model>_np<n_pool>/best_seed*.pth
    """
    models = {}
    for mtype in MODEL_REGISTRY:
        run_dir = os.path.join(ckpt_dir, f"{mtype}_np{n_pool}")
        if not os.path.isdir(run_dir):
            print(f"  [Skip] No checkpoint dir for {mtype}")
            continue

        # Find best checkpoint (pick seed with highest acc from summary)
        summary_path = os.path.join(run_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            # Find seed with highest test_acc
            best = max(summary["per_seed"], key=lambda x: x["test_acc"])
            ckpt_path = best["ckpt"]
        else:
            # Fallback: grab any .pth file
            ptfiles = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
            if not ptfiles:
                print(f"  [Skip] No .pth found for {mtype}")
                continue
            ckpt_path = os.path.join(run_dir, ptfiles[0])

        if not os.path.exists(ckpt_path):
            print(f"  [Skip] Checkpoint not found: {ckpt_path}")
            continue

        model = build_model(mtype, num_classes, d_model=d_model, n_pool=n_pool,
                            n_blocks=n_blocks, d_state=d_state, n_heads=n_heads)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models[DISPLAY_NAMES[mtype]] = model
        print(f"  Loaded: {DISPLAY_NAMES[mtype]}  ← {ckpt_path}")

    return models


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="CKA feature similarity analysis")
    p.add_argument("--dataset",    required=True,
                   choices=["dtd","stl10","cifar100"])
    p.add_argument("--data_path",  required=True)
    p.add_argument("--ckpt_dir",   required=True,
                   help="e.g. outputs/rice")
    p.add_argument("--n_pool",     type=int, default=3)
    p.add_argument("--d_model",    type=int, default=64)
    p.add_argument("--n_blocks",   type=int, default=2)
    p.add_argument("--d_state",    type=int, default=16)
    p.add_argument("--n_heads",    type=int, default=4)
    p.add_argument("--mode",       choices=["cross","internal","both"], default="both")
    p.add_argument("--max_batches",type=int, default=20,
                   help="Batches to use for CKA estimation (more=accurate, slower)")
    p.add_argument("--output_dir", default=None,
                   help="Where to save plots (default: ckpt_dir/cka/)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.output_dir or os.path.join(args.ckpt_dir, "cka")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    _, _, test_loader, class_names = get_loaders(
        args.dataset, args.data_path,
        batch_size=64, num_workers=2)
    num_classes = len(class_names)

    # Load models
    print(f"\nLoading trained models from {args.ckpt_dir} (n_pool={args.n_pool})...")
    models = load_trained_models(
        args.ckpt_dir, args.n_pool, args.d_model, args.n_blocks,
        num_classes, args.d_state, args.n_heads)

    if not models:
        print("No models found. Run run_ablation.py first.")
        return

    for name, model in models.items():
        models[name] = model.to(device)

    # ---- Cross-model CKA (final representations) ----
    if args.mode in ("cross", "both"):
        print(f"\nComputing cross-model CKA (final representations)...")
        # Hook: output of model.norm (final sequence output before head)
        cka_mat, labels = compute_cross_model_cka(
            models,
            hook_getter=lambda m: m.norm,
            loader=test_loader,
            device=device,
            max_batches=args.max_batches,
        )
        title = (f"Cross-Model CKA — Final Representations\n"
                 f"{args.dataset.upper()}, n_pool={args.n_pool} (L={models[list(models.keys())[0]].stem})")
        plot_cka_heatmap(cka_mat, labels,
                         f"Cross-Model CKA (Final Representations)\n"
                         f"{args.dataset.upper()}, n_pool={args.n_pool}",
                         os.path.join(out_dir, f"cka_cross_np{args.n_pool}.png"))

        # Print matrix
        print(f"\n  CKA Matrix:")
        for i, li in enumerate(labels):
            row = "  ".join(f"{cka_mat[i,j]:.2f}" for j in range(len(labels)))
            print(f"    {li:25s}: {row}")

    # ---- Internal CKA (CNN stem vs sequence output per model) ----
    if args.mode in ("internal", "both"):
        print(f"\nComputing internal CKA (CNN stem → sequence output per model)...")

        stem_feats = {}
        seq_feats  = {}

        for name, model in models.items():
            # Stem output
            sf = extract_features(model, model.stem, test_loader, device, args.max_batches)
            stem_feats[name] = sf

            # Sequence output (after norm)
            sqf = extract_features(model, model.norm, test_loader, device, args.max_batches)
            seq_feats[name]  = sqf

        plot_internal_cka(
            stem_feats, seq_feats, list(models.keys()),
            f"Sequence Block Impact: CKA(stem, output)\n"
            f"{args.dataset.upper()}, n_pool={args.n_pool}\n"
            f"Low CKA = sequence block transformed representations significantly",
            os.path.join(out_dir, f"cka_internal_np{args.n_pool}.png"),
        )

        print(f"\n  Internal CKA (how much does the sequence block change features?):")
        for name in models:
            val = linear_cka(stem_feats[name], seq_feats[name])
            interp = "barely changed" if val > 0.8 else "moderately changed" if val > 0.5 else "significantly changed"
            print(f"    {name:25s}: {val:.3f}  ({interp})")

    print(f"\nDone. Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
