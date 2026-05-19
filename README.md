# Does Mamba Actually Help? A Controlled Ablation of SSM Contribution in Hybrid CNN Models for Visual Classification

**Ayush Kushwaha** — [arXiv preprint](https://github.com/0xKushwaha/mamba_cnn) · [Paper PDF](paper/)

---

## Overview

Hybrid CNN–Mamba architectures have grown quickly in the vision literature on the assumption that SSM blocks add representational power beyond what a convolutional stem already provides. This paper tests that assumption directly.

Five model variants share an **identical CNN stem and classification head**. Only the sequence processing block differs. Each variant is evaluated on three datasets spanning a task-complexity gradient, at three sequence lengths controlled by CNN pooling depth, with up to three random seeds.

The findings are clean:
- On **texture tasks (DTD)**, all five variants are statistically indistinguishable. The CNN stem already solves the task.
- On **object tasks (STL-10, CIFAR-100)**, Mamba outperforms the parameter-matched MLP by **2–13 pp**, ruling out a capacity explanation. The gap grows with sequence length.
- **Mamba vs. Attention**: Mamba wins at every sequence length on STL-10. On CIFAR-100, attention leads only at L=36; from L=144 onward Mamba takes over as attention degrades without positional encodings.
- **Bidirectionality adds nothing**: uni- and bidirectional Mamba are indistinguishable across the full grid.

---

## Model Variants

All five variants share the same CNN stem (`Conv2d–BN–ReLU–MaxPool2d` blocks) and a linear classification head. Only the inserted sequence block changes:

| Variant | Sequence block |
|---|---|
| `pure_cnn` | None — global average pool directly into head |
| `cnn_mlp` | 2-layer FFN with 4× hidden expansion (parameter-matched to Mamba) |
| `cnn_mamba_uni` | Unidirectional Mamba S6 |
| `cnn_mamba_bi` | Bidirectional Mamba (forward + backward scan) |
| `cnn_attn` | Multi-head self-attention + FFN, no positional encodings |

The MLP hidden dimension is tuned to match Mamba's parameter count at each sequence length, so accuracy differences are architectural rather than capacity effects.

---

## Sequence Length Control

All datasets are resized to 96×96. The number of pooling stages (`n_pool`) sets the token count seen by the sequence block:

| `n_pool` | Spatial grid | Sequence length L |
|---|---|---|
| 4 | 6×6 | 36 |
| 3 | 12×12 | 144 |
| 2 | 24×24 | 576 |

Model dimension is d=64 with 2 blocks throughout.

---

## Datasets

| Dataset | Classes | Train images | Role |
|---|---|---|---|
| DTD | 47 | 1,880 | Texture — low complexity anchor |
| STL-10 | 10 | ~4,500 | Mixed shape+texture — mid complexity |
| CIFAR-100 | 100 | 45,000 | Fine-grained — high complexity |

---

## Results

### DTD — Texture Classification

| Model | L=36 | L=144 | L=576 |
|---|---|---|---|
| Pure CNN | **26.44 ± 0.16** | 24.20 ± 0.50 | 22.52 ± 0.46 |
| CNN + MLP | 25.66 ± 0.81 | 25.39 ± 0.55 | **25.66 ± 0.13** |
| CNN + Mamba (Uni) | 23.81 ± 0.37 | 24.96 ± 1.23 | 24.95 ± 1.59 |
| CNN + Mamba (Bi) | 22.68 ± 0.71 | 24.73 ± 0.92 | 25.02 ± 1.45 |
| CNN + Attention | 25.48 ± 0.43 | **26.08 ± 0.16** | 25.05 ± 0.66 |

All variants within ~4% — sequence block type is irrelevant for texture.

### STL-10 — Mixed Complexity

| Model | L=36 | L=144 | L=576 |
|---|---|---|---|
| Pure CNN | 65.83 ± 0.78 | 58.50 ± 0.44 | 52.40 ± 0.80 |
| CNN + MLP | 69.34 ± 0.11 | 64.65 ± 0.39 | 63.40 ± 0.73 |
| CNN + Mamba (Uni) | **71.80 ± 0.08** | **71.97 ± 0.43** | **72.67 ± 1.14** |
| CNN + Mamba (Bi) | 71.02 ± 0.66 | 71.48 ± 0.34 | 72.14 ± 0.26 |
| CNN + Attention | 69.86 ± 0.73 | 66.57 ± 0.33 | 66.97 ± 0.58 |

Mamba leads MLP by 9.3 pp at L=576. Attention trails Mamba by 5.7 pp at L=576.

### CIFAR-100 — Fine-Grained Recognition

| Model | L=36 | L=144 | L=576 |
|---|---|---|---|
| Pure CNN | 38.95 ± 0.42 | 28.59 ± 0.25 | 26.67 ± 0.31 |
| CNN + MLP | 50.43 ± 0.15 | 44.35 ± 0.63 | 43.16 ± 0.20 |
| CNN + Mamba (Uni) | 54.05 ± 0.50 | **53.65 ± 0.36** | **55.79 ± 0.26** |
| CNN + Mamba (Bi) | 53.98 ± 0.83 | 53.47 ± 0.39 | 55.39 ± 0.31 |
| CNN + Attention | **58.49 ± 0.53** | 50.25 ± 0.26 | 52.51 ± 0.37 |

Attention wins only at L=36. Mamba leads by 12.6 pp over MLP at L=576.

---

## Project Structure

```
mamba_cnn/
├── models.py        — All 5 model variants
├── data.py          — Dataset loaders (DTD, STL-10, CIFAR-100)
├── train.py         — Training script (single model × dataset × seeds)
├── run_all.py       — Runs the full ablation matrix
├── paper/
│   ├── paper.tex
│   ├── paper.bib
│   ├── generate_figures.py
│   └── figures/
└── outputs/         — Per-run checkpoints and summary JSON files
```

---

## Setup

```bash
pip install torch torchvision mamba-ssm einops tqdm
```

Mamba's selective scan kernel requires a CUDA GPU. CPU-only inference is not supported for Mamba variants.

---

## Running Experiments

### Single run (one model, one dataset, multiple seeds)

```bash
python train.py \
  --model cnn_mamba_uni \
  --dataset stl10 \
  --data_path ./data \
  --n_pool 3 \
  --d_model 64 \
  --n_blocks 2 \
  --seeds 0 42 99 \
  --num_workers 4
```

### Full ablation (all 5 variants × 3 sequence lengths)

```bash
python run_all.py --dataset dtd     --data_path ./data --epochs 200 --num_workers 4
python run_all.py --dataset stl10   --data_path ./data --epochs 150 --num_workers 4
python run_all.py --dataset cifar100 --data_path ./data --epochs 100 --num_workers 4
```

Results are written to `outputs/<dataset>/` as per-seed JSON files and a `summary.json` for each configuration.

### Generate paper figures

```bash
python paper/generate_figures.py
# Saves to paper/figures/fig{1,2,3}_*.{pdf,png}
```

---

## Training Details

- **Optimiser:** AdamW, lr=2×10⁻³, weight decay=0.05
- **Schedule:** CosineAnnealingLR, η_min=10⁻⁶
- **Epochs:** 200 (DTD), 150 (STL-10), 100 (CIFAR-100)
- **Batch size:** 128
- **Augmentation:** random crop, horizontal flip, color jitter (train); center crop (test)
- **Checkpoint:** lowest validation loss epoch
- **Hardware:** NVIDIA A100 80 GB (multi-GPU via DataParallel)

---

## Citation

If you use this code or findings, please cite:

```bibtex
@article{kushwaha2025mamba,
  title   = {Does {Mamba} Actually Help? A Controlled Ablation of {SSM} Contribution
             in Hybrid {CNN} Models for Visual Classification},
  author  = {Kushwaha, Ayush},
  journal = {arXiv preprint},
  year    = {2025}
}
```
