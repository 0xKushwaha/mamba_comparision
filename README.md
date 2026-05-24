# Does Mamba Actually Help? A Controlled Ablation of SSM Contribution in Hybrid CNN Models

**Ayush Kushwaha**

---

## Overview

Hybrid CNN–Mamba architectures have grown quickly in the vision literature on the assumption that SSM blocks add representational power beyond what a convolutional stem already provides. This project tests that assumption directly through controlled ablations.

Five model variants share an **identical CNN stem and classification head**. Only the sequence processing block differs. Each variant is evaluated on three image datasets spanning a task-complexity gradient, at three sequence lengths controlled by CNN pooling depth, with multiple random seeds.

Beyond vision, the project also extends to **time-series** (UCI HAR) and **audio** (Google Speech Commands v2) domains, where the same sequence block variants are compared without any CNN stem.

---

## Model Variants

All five image variants share the same CNN stem (depthwise-separable `Conv2d–BN–GELU` blocks) and a linear classification head. Only the inserted sequence block changes:

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

All image datasets are resized to 96×96. The number of pooling stages (`n_pool`) sets the token count seen by the sequence block:

| `n_pool` | Spatial grid | Sequence length L |
|---|---|---|
| 4 | 6×6 | 36 |
| 3 | 12×12 | 144 |
| 2 | 24×24 | 576 |

Model dimension is d=64 with 2 blocks throughout.

---

## Datasets

### Image Classification

| Dataset | Classes | Train images | Role |
|---|---|---|---|
| DTD | 47 | 1,880 | Texture — low complexity anchor |
| STL-10 | 10 | ~4,500 | Mixed shape+texture — mid complexity |
| CIFAR-100 | 100 | 45,000 | Fine-grained — high complexity |

All three are auto-downloaded via torchvision.

### Time-Series

| Dataset | Classes | Sequence length | Features |
|---|---|---|---|
| UCI HAR | 6 | up to 128 timesteps | 9 inertial channels |

Auto-downloaded from UCI ML repository.

### Audio

| Dataset | Classes | Sequence length | Features |
|---|---|---|---|
| Google Speech Commands v2 | 35 | 101 frames | 120 (MFCC + delta + delta-delta) |

Auto-downloaded via torchaudio.

---

## Project Structure

```
mamba_cnn/
├── models.py            — All 5 image model variants
├── data.py              — Image dataset loaders (DTD, STL-10, CIFAR-100)
├── train.py             — Training script for image experiments (single model × dataset × seeds)
├── run_all.py           — Runs the full image ablation matrix
├── train_timeseries.py  — Time-series experiment (UCI HAR)
├── train_audio.py       — Audio experiment (Speech Commands v2)
├── download_data.py     — Utility for pre-downloading datasets
├── cka_analysis.py      — CKA similarity analysis between model variants
└── outputs/             — Per-run checkpoints and summary JSON files
```

---

## Setup

```bash
pip install torch torchvision torchaudio mamba-ssm einops tqdm
```

Mamba's selective scan kernel requires a CUDA GPU. CPU-only inference is not supported for Mamba variants.

---

## Running Experiments

### Image experiments — single run (one model, one dataset, multiple seeds)

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

### Image experiments — full ablation (all 5 variants × 3 sequence lengths)

```bash
python run_all.py --dataset dtd      --data_path ./data --num_workers 4
python run_all.py --dataset stl10    --data_path ./data --num_workers 4
python run_all.py --dataset cifar100 --data_path ./data --num_workers 4
```

Results are written to `outputs/<dataset>/` as per-seed JSON files and a `summary.json` for each configuration.

### Time-series experiment (UCI HAR)

```bash
python train_timeseries.py \
  --data_root ./data \
  --seq_lens 32 64 128 \
  --d_model 128 \
  --n_layers 4 \
  --epochs 100 \
  --n_seeds 3
```

Dataset auto-downloads on first run.

### Audio experiment (Speech Commands v2)

```bash
python train_audio.py \
  --data_root ./data \
  --seq_lens 32 64 101 \
  --d_model 256 \
  --n_layers 6 \
  --epochs 100 \
  --n_seeds 3
```

MFCC features are precomputed and cached on first run.

---

## Training Details

### Image experiments

- **Optimiser:** AdamW, lr=2×10⁻³, weight decay=0.05
- **Schedule:** CosineAnnealingLR, η_min=10⁻⁶
- **Epochs:** configurable (default 100)
- **Batch size:** 128
- **Augmentation:** random crop, horizontal flip, color jitter (train); center crop (test)
- **Regularisation:** label smoothing=0.05, early stopping (patience=25)
- **Checkpoint:** lowest validation loss epoch

### Time-series and audio experiments

- **Optimiser:** AdamW, lr=2×10⁻³, weight decay=0.05
- **Schedule:** CosineAnnealingLR, η_min=10⁻⁶
- **Epochs:** 100 (configurable)
- **Batch size:** 128 (time-series), 512 (audio)
- **Features:** raw inertial signals (time-series); MFCC + delta + delta-delta (audio)
