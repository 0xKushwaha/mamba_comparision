"""
train.py — Universal training script for SSM contribution study.

Trains a single model variant on a single dataset with one or more seeds.
Results are saved to JSON for aggregation by run_ablation.py.

Usage:
    # Single run
    python train.py --model cnn_mamba_bi --dataset rice \
                    --data_path /path/to/Rice_Leaf_AUG --n_pool 3

    # Multi-seed (for mean ± std reporting)
    python train.py --model cnn_mamba_bi --dataset cifar100 \
                    --data_path ./data --n_pool 2 --seeds 0 42 99

    # Full config
    python train.py --model cnn_attn --dataset tiny_imagenet \
                    --data_path /path/to/tiny-imagenet-200 \
                    --n_pool 3 --d_model 64 --n_blocks 2 \
                    --epochs 100 --lr 2e-3 --batch_size 128 \
                    --seeds 0 42 99 --output_dir outputs/
"""

import os
import json
import random
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import build_model, count_params, sequence_length, DISPLAY_NAMES
from data import get_loaders, DATASET_INFO


# ============================================================
# SEEDING
# ============================================================

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# TRAIN / EVAL EPOCH
# ============================================================

def run_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    model.train(train)
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ============================================================
# SINGLE-SEED TRAINING RUN
# ============================================================

def train_one_seed(args, seed: int, num_classes: int,
                   train_loader, val_loader, test_loader,
                   device, run_dir: str) -> dict:
    """
    Full training run for one seed. Returns result dict.
    """
    seed_everything(seed)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt = os.path.join(run_dir, f"best_seed{seed}.pth")

    # Build model
    model = build_model(
        args.model, num_classes,
        d_model=args.d_model, n_pool=args.n_pool,
        n_blocks=args.n_blocks, d_state=args.d_state,
        n_heads=args.n_heads,
    ).to(device)

    params     = count_params(model)
    seq_len    = sequence_length(args.img_size, args.n_pool)

    criterion  = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    pbar = tqdm(range(1, args.epochs + 1),
                desc=f"  [{DISPLAY_NAMES[args.model]}, seed={seed}]",
                dynamic_ncols=True)

    for epoch in pbar:
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        pbar.set_postfix(vl_acc=f"{vl_acc:.3f}", tr_acc=f"{tr_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save(best_weights, best_ckpt)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                pbar.write(f"    Early stop at epoch {epoch}")
                break

    # Final test evaluation with best weights
    model.load_state_dict(best_weights)
    te_loss, te_acc = run_epoch(model, test_loader, criterion, None, device, train=False)

    return {
        "seed":        seed,
        "test_acc":    round(te_acc * 100, 4),
        "test_loss":   round(te_loss, 6),
        "best_val_loss": round(best_val_loss, 6),
        "params_total": params["total"],
        "params_seq":   params["seq"],
        "size_kb":      round(params["size_kb"], 2),
        "seq_len":      seq_len,
        "ckpt":         best_ckpt,
        "history":      history,
    }


# ============================================================
# MULTI-SEED WRAPPER
# ============================================================

def train_all_seeds(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Model   : {DISPLAY_NAMES[args.model]}")
    print(f"  Dataset : {args.dataset}  ({DATASET_INFO[args.dataset]['description']})")
    print(f"  n_pool  : {args.n_pool}  →  L = {sequence_length(args.img_size, args.n_pool)} tokens")
    print(f"  d_model : {args.d_model},  n_blocks: {args.n_blocks},  epochs: {args.epochs}")
    print(f"  Seeds   : {args.seeds}")
    print(f"  Device  : {device}")
    print(f"{'='*65}")

    # Experiment output dir:  outputs/<dataset>/<model>_np<n_pool>/
    run_dir = os.path.join(args.output_dir, args.dataset,
                           f"{args.model}_np{args.n_pool}")
    os.makedirs(run_dir, exist_ok=True)

    # Load data once (fixed split_seed ensures same train/val/test across all seeds)
    train_loader, val_loader, test_loader, class_names = get_loaders(
        args.dataset, args.data_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        train_seed=args.seeds[0],   # initial; re-seeded per run inside loop
    )
    num_classes = len(class_names)
    print(f"  Classes : {num_classes}  |  "
          f"Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}\n")

    all_results = []
    for seed in args.seeds:
        # Reload loaders with per-seed shuffle seed for fair multi-seed comparison
        train_loader, val_loader, test_loader, _ = get_loaders(
            args.dataset, args.data_path,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_seed=args.split_seed,
            train_seed=seed,
        )
        result = train_one_seed(args, seed, num_classes,
                                train_loader, val_loader, test_loader,
                                device, run_dir)
        all_results.append(result)
        print(f"  Seed {seed:>4} →  Test Acc: {result['test_acc']:.2f}%  "
              f"| Params: {result['params_total']:,}  L={result['seq_len']}")

    # Summary statistics
    accs = [r["test_acc"] for r in all_results]
    summary = {
        "model":        args.model,
        "model_name":   DISPLAY_NAMES[args.model],
        "dataset":      args.dataset,
        "n_pool":       args.n_pool,
        "seq_len":      sequence_length(args.img_size, args.n_pool),
        "d_model":      args.d_model,
        "n_blocks":     args.n_blocks,
        "seeds":        args.seeds,
        "mean_acc":     round(float(np.mean(accs)), 4),
        "std_acc":      round(float(np.std(accs)),  4),
        "best_acc":     round(float(np.max(accs)),  4),
        "params_total": all_results[0]["params_total"],
        "params_seq":   all_results[0]["params_seq"],
        "size_kb":      all_results[0]["size_kb"],
        "timestamp":    datetime.datetime.now().isoformat(),
        "per_seed":     [{k: v for k, v in r.items() if k != "history"}
                         for r in all_results],
    }

    # Save results
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Result : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%  "
          f"(best={np.max(accs):.2f}%)")
    print(f"  Saved  : {summary_path}")

    return summary


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a single model variant")

    # Required
    p.add_argument("--model",      required=True,
                   choices=["pure_cnn","cnn_mlp","cnn_mamba_uni","cnn_mamba_bi","cnn_attn"],
                   help="Model variant")
    p.add_argument("--dataset",    required=True,
                   choices=["dtd","stl10","tiny_imagenet"],
                   help="Dataset to train on")
    p.add_argument("--data_path",  required=True,
                   help="Path to dataset root directory")

    # Architecture
    p.add_argument("--n_pool",   type=int, default=3,
                   help="CNN downsampling stages — controls sequence length L (default: 3)")
    p.add_argument("--d_model",  type=int, default=64,  help="Feature dimension (default: 64)")
    p.add_argument("--n_blocks", type=int, default=2,   help="Sequence processing blocks (default: 2)")
    p.add_argument("--d_state",  type=int, default=16,  help="Mamba SSM state dim (default: 16)")
    p.add_argument("--n_heads",  type=int, default=4,   help="Attention heads (default: 4)")
    p.add_argument("--img_size", type=int, default=None,
                   help="Image size override (default: dataset-specific)")

    # Training
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--lr",              type=float, default=2e-3)
    p.add_argument("--weight_decay",    type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience",        type=int,   default=25,
                   help="Early stopping patience (default: 25)")

    # Multi-seed
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 42, 99],
                   help="Seeds to train with (default: 0 42 99)")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Fixed seed for train/val/test split (default: 42)")

    # I/O
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir",  default="outputs",
                   help="Root output directory (default: outputs/)")

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    # Fill in default image size if not specified
    if args.img_size is None:
        from data import DATASET_INFO
        args.img_size = DATASET_INFO[args.dataset]["default_img_size"]

    train_all_seeds(args)
