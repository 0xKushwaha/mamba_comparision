"""
run_all.py — Runs the full experiment matrix for the SSM contribution study.

For a given dataset, trains all 5 model variants × all n_pool values × all seeds.
Aggregates results into a publication-ready table.

Datasets (ordered by task complexity):
  dtd      — Describable Textures (local texture, auto-download)
  stl10    — STL-10 (mixed object recognition, auto-download)
  cifar100 — CIFAR-100 (100-class, auto-download)

All three share image size 96×96 → identical sequence lengths:
  n_pool=4  →  L=36    (6×6  spatial tokens)
  n_pool=3  →  L=144   (12×12 spatial tokens)
  n_pool=2  →  L=576   (24×24 spatial tokens)

Usage:
    # DTD — texture (auto-downloads)
    python run_all.py --dataset dtd --data_path ./data

    # STL-10 — mixed (auto-downloads)
    python run_all.py --dataset stl10 --data_path ./data

    # CIFAR-100 — complex (auto-downloads)
    python run_all.py --dataset cifar100 --data_path ./data

    # Custom config
    python run_all.py --dataset stl10 --data_path ./data \
        --seeds 0 42 99 --epochs 150 --d_model 64 --n_blocks 2

    # Resume a crashed run (skips completed experiments)
    python run_all.py --dataset dtd --data_path ./data --skip_existing
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from models import DISPLAY_NAMES, sequence_length
from data import DATASET_INFO
from train import train_all_seeds, parse_args as _base_parse_args


# ============================================================
# EXPERIMENT MATRIX
# ============================================================

ALL_MODELS  = ["pure_cnn", "cnn_mlp", "cnn_mamba_uni", "cnn_mamba_bi", "cnn_attn"]

# n_pool values to sweep (controls sequence length L)
N_POOL_RANGE = [4, 3, 2]   # short → medium → long sequences


# ============================================================
# RESULT TABLE BUILDER
# ============================================================

def build_table(results: list, dataset: str) -> pd.DataFrame:
    """
    Converts list of summary dicts into a multi-index DataFrame:
        rows    = model variants
        columns = n_pool values (sequence lengths)
        cells   = "mean% ± std%"
    """
    rows = []
    for r in results:
        rows.append({
            "Model":   DISPLAY_NAMES[r["model"]],
            "n_pool":  r["n_pool"],
            "L":       r["seq_len"],
            "Acc":     f"{r['mean_acc']:.2f} ± {r['std_acc']:.2f}",
            "Mean":    r["mean_acc"],
            "Params":  r["params_total"],
            "SeqParams": r["params_seq"],
        })

    df = pd.DataFrame(rows)
    return df


def print_table(df: pd.DataFrame, dataset: str):
    """Prints a human-readable ablation table."""
    img_size = DATASET_INFO[dataset]["default_img_size"]

    print(f"\n{'='*75}")
    print(f"  ABLATION RESULTS  |  Dataset: {dataset.upper()}  "
          f"({DATASET_INFO[dataset]['description']})")
    print(f"{'='*75}")

    # Pivot: Model vs n_pool
    pivot = df.pivot_table(index="Model", columns="n_pool",
                           values="Acc", aggfunc="first")

    # Rename columns to show L
    col_map = {np_: f"np={np_} (L={sequence_length(img_size, np_)})"
               for np_ in pivot.columns}
    pivot.rename(columns=col_map, inplace=True)

    # Reorder rows to match ALL_MODELS order
    order = [DISPLAY_NAMES[m] for m in ALL_MODELS]
    pivot = pivot.reindex([o for o in order if o in pivot.index])

    print(pivot.to_string())
    print(f"{'='*75}")

    # Find best performing config overall
    best_row = df.loc[df["Mean"].idxmax()]
    print(f"\n  Best config: {best_row['Model']}  "
          f"n_pool={best_row['n_pool']} (L={best_row['L']})  "
          f"→  {best_row['Mean']:.2f}%")

    # Check key research questions
    print(f"\n  Key observations:")
    _print_observations(df, dataset, img_size)


def _print_observations(df, dataset, img_size):
    """Auto-generate key finding statements from results."""
    models_ordered = [DISPLAY_NAMES[m] for m in ALL_MODELS]

    for model_name in [DISPLAY_NAMES["cnn_mamba_bi"], DISPLAY_NAMES["cnn_mlp"]]:
        sub = df[df["Model"] == model_name].sort_values("n_pool", ascending=False)
        if len(sub) >= 2:
            delta = sub.iloc[0]["Mean"] - sub.iloc[-1]["Mean"]
            direction = "↑ rises" if delta > 0.5 else ("↓ falls" if delta < -0.5 else "→ flat")
            print(f"    [{model_name}] Accuracy as L grows: {direction}  "
                  f"(Δ={delta:+.2f}%)")

    # Mamba vs MLP comparison
    mamba_rows = df[df["Model"] == DISPLAY_NAMES["cnn_mamba_bi"]]
    mlp_rows   = df[df["Model"] == DISPLAY_NAMES["cnn_mlp"]]

    if len(mamba_rows) > 0 and len(mlp_rows) > 0:
        mamba_mean = mamba_rows["Mean"].mean()
        mlp_mean   = mlp_rows["Mean"].mean()
        delta      = mamba_mean - mlp_mean
        verdict    = "Mamba > MLP" if delta > 0.3 else ("Mamba < MLP" if delta < -0.3 else "Mamba ≈ MLP")
        print(f"    [Mamba vs MLP overall] {verdict}  (Δ={delta:+.2f}%)")

    # CNN alone vs best
    cnn_rows = df[df["Model"] == DISPLAY_NAMES["pure_cnn"]]
    if len(cnn_rows) > 0:
        cnn_mean  = cnn_rows["Mean"].mean()
        best_mean = df["Mean"].max()
        delta     = best_mean - cnn_mean
        print(f"    [Pure CNN vs best]  CNN baseline={cnn_mean:.2f}%,  "
              f"best={best_mean:.2f}%,  gap={delta:+.2f}%")


# ============================================================
# MAIN ABLATION RUNNER
# ============================================================

def run_ablation(args):
    dataset    = args.dataset
    output_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    img_size  = DATASET_INFO[dataset]["default_img_size"]
    all_results = []

    # Support running a single n_pool value (for splitting large jobs)
    n_pool_range = [args.n_pool_single] if args.n_pool_single else N_POOL_RANGE

    models_to_run = args.models

    total_runs  = len(models_to_run) * len(n_pool_range)
    run_counter = 0

    print(f"\n{'#'*65}")
    print(f"  ABLATION STUDY")
    print(f"  Dataset  : {dataset} ({DATASET_INFO[dataset]['description']})")
    print(f"  Models   : {len(models_to_run)} {models_to_run}")
    print(f"  n_pool   : {n_pool_range}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Total    : {total_runs} experiments × {len(args.seeds)} seeds each")
    print(f"  Output   : {output_dir}")
    print(f"{'#'*65}\n")

    for n_pool in n_pool_range:
        for model_type in models_to_run:
            run_counter += 1
            run_dir = os.path.join(output_dir, f"{model_type}_np{n_pool}")
            summary_path = os.path.join(run_dir, "summary.json")

            print(f"[{run_counter:>2}/{total_runs}] "
                  f"{DISPLAY_NAMES[model_type]:25s}  "
                  f"n_pool={n_pool}  L={sequence_length(img_size, n_pool)}")

            # Skip if already done (resumable)
            if args.skip_existing and os.path.exists(summary_path):
                print(f"       → Skipping (already exists)")
                with open(summary_path) as f:
                    all_results.append(json.load(f))
                continue

            # Build a namespace that matches what train.py expects
            run_args = argparse.Namespace(
                model          = model_type,
                dataset        = dataset,
                data_path      = args.data_path,
                n_pool         = n_pool,
                d_model        = args.d_model,
                n_blocks       = args.n_blocks,
                d_state        = args.d_state,
                n_heads        = args.n_heads,
                img_size       = img_size,
                epochs         = args.epochs,
                batch_size     = args.batch_size,
                lr             = args.lr,
                weight_decay   = args.weight_decay,
                label_smoothing= args.label_smoothing,
                patience       = args.patience,
                seeds          = args.seeds,
                split_seed     = args.split_seed,
                num_workers    = args.num_workers,
                output_dir     = args.output_dir,
            )

            summary = train_all_seeds(run_args)
            all_results.append(summary)

    # Merge with any existing results from previous runs
    # This prevents overwriting results from other models trained in separate jobs
    seen_keys = {(r["model"], r["n_pool"]) for r in all_results}
    for model_type in ALL_MODELS:
        for n_pool in N_POOL_RANGE:
            if (model_type, n_pool) in seen_keys:
                continue   # already in current run
            existing = os.path.join(output_dir, f"{model_type}_np{n_pool}", "summary.json")
            if os.path.exists(existing):
                with open(existing) as f:
                    all_results.append(json.load(f))

    # Build and print results table
    df = build_table(all_results, dataset)
    print_table(df, dataset)

    # Save aggregated results
    csv_path   = os.path.join(output_dir, "ablation_results.csv")
    table_path = os.path.join(output_dir, "ablation_table.txt")
    json_path  = os.path.join(output_dir, "ablation_all.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save human-readable table
    img_size_val = DATASET_INFO[dataset]["default_img_size"]
    pivot = df.pivot_table(index="Model", columns="n_pool", values="Acc", aggfunc="first")
    col_map = {np_: f"np={np_} (L={sequence_length(img_size_val, np_)})"
               for np_ in pivot.columns}
    pivot.rename(columns=col_map, inplace=True)
    order = [DISPLAY_NAMES[m] for m in ALL_MODELS]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    with open(table_path, "w") as f:
        f.write(f"Ablation: {dataset.upper()}\n")
        f.write("=" * 75 + "\n")
        f.write(pivot.to_string())
        f.write("\n")

    print(f"\n  Saved: {csv_path}")
    print(f"  Saved: {table_path}")
    print(f"  Saved: {json_path}")

    return all_results


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_ablation_args():
    p = argparse.ArgumentParser(description="Run full SSM contribution ablation study")

    # Required
    p.add_argument("--dataset",   required=True,
                   choices=["dtd","stl10","cifar100"])
    p.add_argument("--models",    nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS,
                   help="Model variants to train (default: all 5). "
                        "e.g. --models cnn_mamba_bi cnn_mlp")
    _default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    p.add_argument("--data_path", default=_default_data)

    # Architecture (shared across all runs)
    p.add_argument("--d_model",  type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=2)
    p.add_argument("--d_state",  type=int, default=16)
    p.add_argument("--n_heads",  type=int, default=4)

    # Training
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--lr",               type=float, default=2e-3)
    p.add_argument("--weight_decay",     type=float, default=0.05)
    p.add_argument("--label_smoothing",  type=float, default=0.05)
    p.add_argument("--patience",         type=int,   default=25)

    # Multi-seed
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 42, 99])
    p.add_argument("--split_seed", type=int, default=42)

    # I/O
    p.add_argument("--num_workers",   type=int, required=True,
                   help="Number of DataLoader worker processes (must specify: 0=no workers, 1+=parallel)")
    p.add_argument("--output_dir",    default="outputs")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip experiments whose summary.json already exists (resume)")
    p.add_argument("--n_pool_single", type=int, default=None,
                   choices=[2, 3, 4],
                   help="Run only one n_pool value (for splitting large jobs across PBS submissions)")

    return p.parse_args()


if __name__ == "__main__":
    args = run_ablation(parse_ablation_args())
