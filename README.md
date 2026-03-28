# When Does Mamba Actually Help? Characterizing SSM Contribution in Hybrid CNN Models for Visual Classification

## Abstract

State Space Models (SSMs), particularly Mamba, have emerged as efficient alternatives to attention mechanisms for sequence modeling. Their adoption in vision tasks has produced a wave of hybrid CNN-Mamba architectures — yet a fundamental question remains unanswered: **does the SSM component actually contribute meaningful representational power, or is the CNN stem doing all the work?**

This work presents the first controlled study of SSM contribution in tiny hybrid vision models. We systematically isolate the sequence modeling component and compare Mamba (unidirectional and bidirectional) against parameter-matched alternatives (MLP blocks, self-attention) across three datasets of increasing task complexity and three sequence length regimes. We identify the **task-complexity threshold** and **minimum sequence length** beyond which Mamba's selective state space mechanism provides measurable benefit.

---

## Research Question

> *In hybrid CNN-SSM tiny models, when does the SSM component contribute meaningfully versus acting as a residual MLP substitute? What is the minimum sequence length and task complexity where Mamba's selective gating provides measurable benefit over simpler alternatives?*

---

## Core Hypothesis

We hypothesize that for tasks dominated by **local texture** (e.g., plant/rice disease), CNN features saturate representational capacity before the sequence modeling stage — making the SSM equivalent to an MLP regardless of sequence length. SSM benefit should only emerge when:
1. The task requires **global spatial reasoning** beyond CNN receptive fields, AND
2. The token sequence is **long enough** (L ≥ threshold) to exploit the SSM's recurrent structure.

---

## Experimental Design

### Model Variants (5 total)

All variants share an **identical CNN stem** and **classification head**. Only the sequence processing block changes — enabling a clean ablation.

| Model | Sequence Block | Purpose |
|-------|---------------|---------|
| `pure_cnn` | None (GAP directly) | Lower bound — CNN alone |
| `cnn_mlp` | 2-layer FFN (4× expansion) | Parameter-matched baseline |
| `cnn_mamba_uni` | Unidirectional Mamba (S6) | SSM, causal scan only |
| `cnn_mamba_bi` | Bidirectional Mamba (S6) | SSM, spatial awareness |
| `cnn_attn` | Multi-head self-attention + FFN | Attention baseline |

### CNN Stem — Sequence Length Control

The stem is parameterized by `n_pool` (number of MaxPool2d downsampling stages), which directly controls sequence length L fed into the sequence block:

| n_pool | Rice (96×96) | CIFAR-100 (32×32) | Tiny ImageNet (64×64) |
|--------|-------------|-------------------|----------------------|
| 4 | L = 36 | L = 4 | L = 16 |
| 3 | L = 144 | L = 16 | L = 64 |
| 2 | L = 576 | L = 64 | L = 256 |

### Datasets (3 tasks, increasing complexity)

All three datasets are resized to a common **96×96** resolution. This means sequence lengths L ∈ {36, 144, 576} are **identical across datasets** — task complexity is the only variable changing between rows in the paper table.

| Dataset | Classes | Train Size | Task Type | Expected SSM Benefit |
|---------|---------|------------|-----------|---------------------|
| **DTD** (Describable Textures) | 47 | 1,880 | Pure local texture | Low — CNN receptive fields capture all necessary features |
| **STL-10** | 10 | ~4,500 | Shape + texture mix | Medium — transition zone |
| **Tiny ImageNet** | 200 | 100,000 | Diverse global structure | High — SSM should contribute at longer L |

**Why these three:**
- DTD is the canonical texture benchmark — reviewers immediately understand why CNN should dominate
- STL-10 is 96×96 natively (no upsampling distortion) and is widely used in efficient model papers
- Tiny ImageNet is the standard "harder than CIFAR" benchmark in efficient architecture work
- All three auto-download (DTD, STL-10 via torchvision) or have simple one-command downloads

### The Full Ablation Matrix

For each dataset × each n_pool × each model variant. Since all datasets use 96×96, L values are identical across rows:

```
                         n_pool=4     n_pool=3      n_pool=2
                         L=36         L=144         L=576
                         ──────────────────────────────────
pure_cnn                   acc          acc           acc
cnn_mlp                    acc          acc           acc
cnn_mamba_uni              acc          acc           acc
cnn_mamba_bi               acc          acc           acc
cnn_attn                   acc          acc           acc
```

Running this for all 3 datasets gives a 3 × 5 × 3 = 45-cell table.
The pattern across cells answers the research question.

---

## Key Questions the Table Answers

1. **Does L matter?** — Compare columns within a row. If accuracy rises with L for `cnn_mamba_*` but not `cnn_mlp`, the SSM recurrence is being exploited.

2. **Does the SSM beat MLP?** — Compare `cnn_mamba_bi` vs `cnn_mlp` at same n_pool. If they match → selective gating provides no benefit.

3. **Is it task-dependent?** — Compare the same cell across datasets. The cross-over point (where `cnn_mamba_bi` starts beating `cnn_mlp`) defines the task-complexity threshold.

4. **Does bidirectionality matter?** — Compare `cnn_mamba_uni` vs `cnn_mamba_bi`. For spatial (non-sequential) data, bidirectional should always win.

5. **Does the CNN alone suffice?** — Compare `pure_cnn` vs all others. If `pure_cnn` matches everything on Rice Disease, the sequence block is decorative for texture tasks.

---

## Project Structure

```
mamba_cnn/
├── README.md           ← This file
├── models.py           ← All 5 model variants
├── data.py             ← Dataset loaders (Rice, CIFAR-100, Tiny ImageNet)
├── train.py            ← Training script (single experiment)
├── run_ablation.py     ← Runs full experiment matrix
└── cka_analysis.py     ← CKA feature similarity visualization
```

---

## Setup

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn tqdm
```

---

## Running Experiments

### Single Experiment

```bash
python train.py \
  --model cnn_mamba_bi \
  --dataset stl10 \
  --data_path ./data \
  --n_pool 3 \
  --d_model 64 \
  --n_blocks 2 \
  --seeds 0 42 99
```

### Full Ablation (all variants × all n_pool values)

```bash
# DTD — textures (auto-downloads, fastest — only 1880 train images)
python run_ablation.py --dataset dtd --data_path ./data --epochs 200

# STL-10 — mixed (auto-downloads, good starting point)
python run_ablation.py --dataset stl10 --data_path ./data --epochs 150

# Tiny ImageNet — complex (manual download required)
python run_ablation.py --dataset tiny_imagenet --data_path /path/to/tiny-imagenet-200 --epochs 100
```

Results are saved to `outputs/<dataset>/ablation_results.csv` and `ablation_table.txt`.

### CKA Analysis (after training)

```bash
python cka_analysis.py \
  --dataset rice \
  --data_path /path/to/Rice_Leaf_AUG \
  --ckpt outputs/rice/cnn_mamba_bi_np3/best.pth
```

---

## Expected Results Template

*(Fill in after experiments)*

### DTD — Does L matter for texture tasks?

| Model | L=36 (np=4) | L=144 (np=3) | L=576 (np=2) | Δ (short→long) |
|-------|------------|--------------|--------------|----------------|
| pure_cnn | — | — | — | — |
| cnn_mlp | — | — | — | — |
| cnn_mamba_uni | — | — | — | — |
| cnn_mamba_bi | — | — | — | — |
| cnn_attn | — | — | — | — |

**Hypothesis**: Δ ≈ 0 for all models on DTD. CNN dominates, L is irrelevant for texture.

### STL-10 — Transition Point

*(same table structure)*

**Hypothesis**: `cnn_mamba_bi` starts outperforming `cnn_mlp` at longer L.

### Tiny ImageNet — Where SSM Contributes

*(same table structure)*

**Hypothesis**: Clear `cnn_mamba_bi` > `cnn_mlp` gap at L ≥ 144.

---

## Paper Contribution Summary

1. **First controlled ablation** isolating SSM vs MLP in tiny hybrid models
2. **Task-complexity threshold identification**: empirically defined cross-over point
3. **Sequence length analysis**: minimum L for SSM benefit
4. **Design guideline**: practitioners should not use SSMs for texture/disease classification at sub-1MB scale
5. **Sub-1MB deployment**: ONNX export for all variants

---

## Interpretation Guide

```
If (cnn_mamba_bi ≈ cnn_mlp) across all L and datasets:
    → SSM selective gating adds nothing; use MLP or pure CNN

If (cnn_mamba_bi > cnn_mlp) only at large L:
    → SSM needs long sequences; use more tokens (fewer downsamples)

If (cnn_mamba_bi > cnn_mlp) only on complex datasets:
    → SSM only helps for global-reasoning tasks; don't use for texture

If (cnn_mamba_bi > cnn_mlp) consistently:
    → SSM is genuinely better; architecture is justified
```

---

## Target Venues

| Venue | Type | Deadline | Notes |
|-------|------|----------|-------|
| WACV 2026 | Conference | Aug 2025 | Good fit (vision + efficiency) |
| BMVC 2025 | Conference | May 2025 | Accessible, peer-reviewed |
| Neural Networks | Journal | Rolling | Strong fit for systematic study |
| arXiv | Preprint | Anytime | Do first to establish priority |
