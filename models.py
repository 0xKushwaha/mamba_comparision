"""
models.py — All model variants for SSM contribution study.

All 5 variants share an identical CNN stem and classification head.
Only the sequence processing block changes, enabling a clean ablation.

Variants:
  pure_cnn       — CNN stem → GAP → Head  (no sequence modeling)
  cnn_mlp        — CNN stem → FFN blocks → Head  (parameter-matched baseline)
  cnn_mamba_uni  — CNN stem → Unidirectional Mamba → Head
  cnn_mamba_bi   — CNN stem → Bidirectional Mamba → Head
  cnn_attn       — CNN stem → Self-Attention blocks → Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SHARED BUILDING BLOCKS
# ============================================================

class DepthwiseSeparable(nn.Module):
    """Depthwise-separable convolution: DW(3×3) → BN → GELU → PW(1×1) → BN → GELU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.pw(self.dw(x))


def build_cnn_stem(d_model: int, n_pool: int) -> nn.Sequential:
    """
    Build CNN stem with n_pool downsampling stages.

    Spatial resolution after stem:
        img_size / 2^n_pool  →  sequence length = (img_size / 2^n_pool)^2

    Channel progression:
        n_pool=1 : Conv(3→d_model) + Pool
        n_pool=2 : Conv(3→32)      + Pool  → DWS(32→d_model)     + Pool
        n_pool=3 : Conv(3→16)      + Pool  → DWS(16→32) + Pool   → DWS(32→d_model) + Pool
        n_pool=4 : Conv(3→16)      + Pool  → DWS(16→32) + Pool   → DWS(32→d_model) + Pool → DWS(d_model→d_model) + Pool
    """
    assert 1 <= n_pool <= 4, "n_pool must be in [1, 2, 3, 4]"

    def entry(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(), nn.MaxPool2d(2))

    stages = []
    if n_pool == 1:
        stages += [entry(3, d_model)]
    elif n_pool == 2:
        stages += [entry(3, 32), DepthwiseSeparable(32, d_model), nn.MaxPool2d(2)]
    elif n_pool == 3:
        stages += [entry(3, 16), DepthwiseSeparable(16, 32), nn.MaxPool2d(2),
                   DepthwiseSeparable(32, d_model), nn.MaxPool2d(2)]
    elif n_pool == 4:
        stages += [entry(3, 16), DepthwiseSeparable(16, 32), nn.MaxPool2d(2),
                   DepthwiseSeparable(32, d_model), nn.MaxPool2d(2),
                   DepthwiseSeparable(d_model, d_model), nn.MaxPool2d(2)]

    return nn.Sequential(*stages)


# ============================================================
# SEQUENCE PROCESSING BLOCKS
# ============================================================

# ---- 1. MLP Block (FFN, parameter-matched to Mamba) ----

class MLPBlock(nn.Module):
    """
    Standard 2-layer FFN with residual connection and pre-norm.
    Expansion factor=4 matches Mamba's parameter count at same d_model.
    """
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model))

    def forward(self, x):
        return x + self.ffn(self.norm(x))


# ---- 2. Mamba Block (unidirectional S6 selective scan) ----

class MambaBlock(nn.Module):
    """
    Mamba S6 block with selective state space scan.
    Set bidirectional=True for bidirectional (spatial-aware) variant.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.d_inner  = d_model * expand
        self.dt_rank  = max(1, self.d_inner // 16)
        self.d_state  = d_state

        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                  padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A_init = torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32)
                  .unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_init)
        self.D     = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state space scan (causal, left-to-right)."""
        B, L, _ = x.shape
        A   = -torch.exp(self.A_log.float())
        D   = self.D.float()
        xBC = self.x_proj(x)
        dt_raw, B_p, C = xBC.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt  = F.softplus(self.dt_proj(dt_raw))
        dA  = torch.exp(torch.einsum("bld,ds->blds", dt, A))
        dB  = torch.einsum("bld,bls->blds", dt, B_p)
        h   = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys  = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x[:, i, :, None]
            ys.append(torch.einsum("bds,bs->bd", h, C[:, i]))
        return torch.stack(ys, dim=1) + x * D.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x   = self.norm(x)
        xz  = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)

        # Causal convolution (trim to original length)
        x_h = self.conv1d(x_h.transpose(1, 2))[:, :, :x_h.shape[1]].transpose(1, 2)
        x_h = F.silu(x_h)

        if self.bidirectional:
            # Average forward and backward scans — spatial awareness for image tokens
            y = (self._scan(x_h) + self._scan(x_h.flip(1)).flip(1)) * 0.5
        else:
            y = self._scan(x_h)

        return self.out_proj(y * F.silu(z)) + residual


# ---- 3. Attention Block (MHSA + FFN) ----

class AttentionBlock(nn.Module):
    """
    Standard Transformer encoder block: pre-norm MHSA + pre-norm FFN.
    n_heads chosen to divide d_model evenly; head_dim = d_model // n_heads.
    """
    def __init__(self, d_model: int, n_heads: int = 4, expansion: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = d_model * expansion
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# MODEL CLASSES
# ============================================================

class _BaseHybrid(nn.Module):
    """
    Shared base for all variants.
    Subclasses implement _make_seq_blocks() and forward_seq().
    """
    def __init__(self, num_classes: int, d_model: int, n_pool: int, n_blocks: int):
        super().__init__()
        self.stem   = build_cnn_stem(d_model, n_pool)
        self.norm   = nn.LayerNorm(d_model)
        self.head   = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_model, num_classes))
        self.d_model  = d_model
        self.n_blocks = n_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                             # [B, d_model, H, W]
        x = x.flatten(2).transpose(1, 2)            # [B, L, d_model]
        x = self.forward_seq(x)                      # [B, L, d_model]
        x = self.norm(x)
        return self.head(x.mean(1))                  # [B, num_classes]

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---- Model 1: PureCNN ----

class PureCNN(_BaseHybrid):
    """
    CNN stem → global average pool → head.
    No sequence modeling whatsoever. Accuracy lower-bound.
    """
    def __init__(self, num_classes: int, d_model: int = 64, n_pool: int = 3, n_blocks: int = 0):
        super().__init__(num_classes, d_model, n_pool, n_blocks)

    def forward_seq(self, x):
        return x   # pass-through; norm + mean-pool in base class does the job


# ---- Model 2: CNN_MLP ----

class CNN_MLP(_BaseHybrid):
    """
    CNN stem → n_blocks FFN residual blocks → head.
    Parameter count closely matches CNN_Mamba at same d_model and n_blocks.
    """
    def __init__(self, num_classes: int, d_model: int = 64, n_pool: int = 3,
                 n_blocks: int = 2, expansion: int = 4):
        super().__init__(num_classes, d_model, n_pool, n_blocks)
        self.blocks = nn.Sequential(*[MLPBlock(d_model, expansion) for _ in range(n_blocks)])

    def forward_seq(self, x):
        return self.blocks(x)


# ---- Model 3: CNN_Mamba (uni or bi) ----

class CNN_Mamba(_BaseHybrid):
    """
    CNN stem → n_blocks Mamba S6 blocks → head.
    bidirectional=False: causal scan only (unidirectional).
    bidirectional=True : forward + backward averaged (spatial awareness).
    """
    def __init__(self, num_classes: int, d_model: int = 64, n_pool: int = 3,
                 n_blocks: int = 2, d_state: int = 16, bidirectional: bool = True):
        super().__init__(num_classes, d_model, n_pool, n_blocks)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, bidirectional=bidirectional)
            for _ in range(n_blocks)
        ])

    def forward_seq(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---- Model 4: CNN_Attn ----

class CNN_Attn(_BaseHybrid):
    """
    CNN stem → n_blocks Transformer encoder blocks → head.
    """
    def __init__(self, num_classes: int, d_model: int = 64, n_pool: int = 3,
                 n_blocks: int = 2, n_heads: int = 4, expansion: int = 4):
        super().__init__(num_classes, d_model, n_pool, n_blocks)
        self.blocks = nn.Sequential(*[
            AttentionBlock(d_model, n_heads, expansion) for _ in range(n_blocks)
        ])

    def forward_seq(self, x):
        return self.blocks(x)


# ============================================================
# FACTORY FUNCTION
# ============================================================

MODEL_REGISTRY = {
    "pure_cnn":       PureCNN,
    "cnn_mlp":        CNN_MLP,
    "cnn_mamba_uni":  lambda **kw: CNN_Mamba(bidirectional=False, **kw),
    "cnn_mamba_bi":   lambda **kw: CNN_Mamba(bidirectional=True,  **kw),
    "cnn_attn":       CNN_Attn,
}

DISPLAY_NAMES = {
    "pure_cnn":      "Pure CNN",
    "cnn_mlp":       "CNN + MLP",
    "cnn_mamba_uni": "CNN + Mamba (Uni)",
    "cnn_mamba_bi":  "CNN + Mamba (Bi)",
    "cnn_attn":      "CNN + Attention",
}


def build_model(model_type: str, num_classes: int, d_model: int = 64,
                n_pool: int = 3, n_blocks: int = 2, d_state: int = 16,
                n_heads: int = 4) -> nn.Module:
    """
    Factory function. Returns an untrained model of the requested type.

    Args:
        model_type : one of MODEL_REGISTRY keys
        num_classes: number of output classes
        d_model    : embedding / channel dimension
        n_pool     : CNN downsampling stages (controls sequence length L)
        n_blocks   : number of sequence processing blocks (0 for pure_cnn)
        d_state    : Mamba SSM state dimension (Mamba variants only)
        n_heads    : attention heads (cnn_attn only)

    Returns:
        nn.Module ready for training
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_type}'. Choose from: {list(MODEL_REGISTRY)}")

    cls = MODEL_REGISTRY[model_type]

    # pure_cnn ignores n_blocks, d_state, n_heads
    if model_type == "pure_cnn":
        return cls(num_classes=num_classes, d_model=d_model, n_pool=n_pool)

    # Mamba variants accept d_state
    if model_type in ("cnn_mamba_uni", "cnn_mamba_bi"):
        return cls(num_classes=num_classes, d_model=d_model, n_pool=n_pool,
                   n_blocks=n_blocks, d_state=d_state)

    # Attention accepts n_heads
    if model_type == "cnn_attn":
        return cls(num_classes=num_classes, d_model=d_model, n_pool=n_pool,
                   n_blocks=n_blocks, n_heads=n_heads)

    # MLP / others
    return cls(num_classes=num_classes, d_model=d_model, n_pool=n_pool, n_blocks=n_blocks)


def count_params(model: nn.Module) -> dict:
    """Returns total, stem, sequence-block, and head parameter counts."""
    def _n(module): return sum(p.numel() for p in module.parameters())

    total  = _n(model)
    stem   = _n(model.stem)
    head   = _n(model.head) + _n(model.norm)
    seq    = total - stem - head

    return {
        "total":  total,
        "stem":   stem,
        "seq":    seq,
        "head":   head,
        "size_kb": total * 4 / 1024,
    }


def sequence_length(img_size: int, n_pool: int) -> int:
    """Compute the token sequence length after n_pool downsampling stages."""
    spatial = img_size // (2 ** n_pool)
    return spatial * spatial


# ============================================================
# QUICK SANITY CHECK
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL PARAMETER COUNTS (d_model=64, n_blocks=2)")
    print("=" * 60)
    for mtype in MODEL_REGISTRY:
        m = build_model(mtype, num_classes=10, d_model=64, n_pool=3, n_blocks=2)
        p = count_params(m)
        print(f"  {DISPLAY_NAMES[mtype]:25s}  "
              f"total={p['total']:>8,}  seq={p['seq']:>7,}  ({p['size_kb']:.1f} KB)")

    print()
    print("  Sequence lengths by dataset × n_pool:")
    for dataset, img_size in [("Rice (96)", 96), ("CIFAR-100 (32)", 32), ("TinyImageNet (64)", 64)]:
        lengths = [f"np={np}→L={sequence_length(img_size, np)}" for np in [2, 3, 4]]
        print(f"    {dataset:22s}: {', '.join(lengths)}")
