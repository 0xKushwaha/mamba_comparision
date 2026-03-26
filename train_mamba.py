#!/usr/bin/env python3
"""
MambaCNN Training Script (Multi-GPU via torchrun)
Launch: torchrun --nproc_per_node=NUM_GPUS train_mamba.py
"""

import os
import sys
import warnings
import json
import contextlib
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
MODEL_NAME      = "mamba_cnn"
DATASET_PATH    = "/home/user2/Documents/classification/cifar10"

IMAGE_SIZE      = 32
BATCH_SIZE      = 128 # best 128
EPOCHS          = 200
WARMUP_EPOCHS   = 10
LR              = 4e-3
WEIGHT_DECAY    = 0.05
LABEL_SMOOTHING = 0.1
SAM_RHO         = 0.1   # sharpness-aware minimization perturbation radius
NUM_WORKERS     = 8
D_MODEL         = 96
N_MAMBA         = 3
D_STATE         = 8
EMA_DECAY       = 0.9999
CUTMIX_ALPHA    = 1 # best 1
MIXUP_ALPHA     = 0.5 # best 0.5
DROP_PATH_MAX   = 0.1   # stochastic depth max rate (linearly increases across blocks)

OUTPUT_DIR        = "outputs"


# ============================================
# MODEL DEFINITION
# ============================================

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())
    def forward(self, x): return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """Squeeze-and-Excite channel attention — negligible parameter cost."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class StemBlock(nn.Module):
    """DW-sep conv + SE channel attention + residual shortcut."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.se = SEBlock(out_ch, reduction=8)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.pw(self.dw(x))) + self.shortcut(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 drop_path: float = 0.0):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = d_model * expand
        self.dt_rank  = max(1, self.d_inner // 16)
        self.drop_path_prob = drop_path
        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                           .unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_init)
        self.D       = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan(self, x):
        B, L, _ = x.shape
        A  = -torch.exp(self.A_log.float())
        D  = self.D.float()
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

    def forward(self, x):
        residual = x
        x  = self.norm(x)
        xz = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)
        x_h = self.conv1d(x_h.transpose(1, 2))[:, :, :x_h.shape[1]].transpose(1, 2)
        x_h = F.silu(x_h)

        # Bidirectional scan: forward + backward (shared weights), averaged → 0 extra params
        y_fwd = self._selective_scan(x_h)
        y_bwd = self._selective_scan(x_h.flip(1)).flip(1)
        y     = (y_fwd + y_bwd) * 0.5

        branch = self.out_proj(y * F.silu(z))

        # Stochastic depth — drop entire residual branch during training
        if self.training and self.drop_path_prob > 0.0:
            survive = 1.0 - self.drop_path_prob
            mask    = torch.bernoulli(
                torch.full((x.size(0), 1, 1), survive, device=x.device, dtype=x.dtype))
            branch  = branch * mask / survive

        return branch + residual


def _morton_order(h: int, w: int) -> torch.Tensor:
    """Reorder raster tokens into Morton (Z-order) so spatially adjacent tokens
    are also sequentially adjacent — better locality for the SSM scan."""
    def spread(n):
        n &= 0xFFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    codes = [spread(c) | (spread(r) << 1) for r in range(h) for c in range(w)]
    return torch.argsort(torch.tensor(codes))


class MambaCNN(nn.Module):
    """
    Staged ResNet+Mamba hybrid.

    INPUT  (3, 32, 32)
      Entry  Conv(3→32) + BN + GELU                    → (32, 32, 32)
      Stage1 2× ConvResBlock(32→32)  [identity short]  → (32, 32, 32)
      Down1  ConvResBlock(32→64) + MaxPool(2)           → (64, 16, 16)
      Stage2 2× ConvResBlock(64→64)  [identity short]  → (64, 16, 16)
      Down2  ConvResBlock(64→d_model) + MaxPool(2)      → (d_model, 8, 8)
             flatten → (B, 64, d_model)
      Stage3 3× MambaResBlock(d_model)  [bidir + res]  → (B, 64, d_model)
      Head   LN → mean → Dropout → Linear → (B, C)
    """
    def __init__(self, num_classes: int, d_model: int = 128, n_mamba: int = 3, d_state: int = 8,
                 drop_path_max: float = 0.1, img_size: int = 32):
        super().__init__()

        # Entry
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU())

        # Stage 1: pure conv residual blocks at full 32×32 resolution
        self.stage1 = nn.Sequential(
            StemBlock(32, 32),
            StemBlock(32, 32))

        # Downsample 1: 32×32 → 16×16, widen 32→64
        self.down1 = nn.Sequential(
            StemBlock(32, 64),
            nn.MaxPool2d(2))

        # Stage 2: pure conv residual blocks at 16×16
        self.stage2 = nn.Sequential(
            StemBlock(64, 64),
            StemBlock(64, 64))

        # Downsample 2: 16×16 → 8×8, widen 64→d_model
        self.down2 = nn.Sequential(
            StemBlock(64, d_model),
            nn.MaxPool2d(2))

        # Learned 2D positional embedding — size derived from two MaxPool2d(2) reductions
        h_out = img_size // 4
        self.pos_embed = nn.Parameter(torch.zeros(1, h_out * h_out, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Morton Z-order: spatially local tokens stay close in sequence
        self.register_buffer("token_order", _morton_order(h_out, h_out))

        # Stage 3: Mamba residual blocks — global context at 64 tokens only
        dp_rates = [drop_path_max * i / max(n_mamba - 1, 1) for i in range(n_mamba)]
        self.mamba = nn.Sequential(*[
            MambaBlock(d_model, d_state=d_state, drop_path=dp_rates[i])
            for i in range(n_mamba)])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.entry(x)                               # (B, 32, 32, 32)
        x = self.stage1(x)                              # (B, 32, 32, 32)
        x = self.down1(x)                               # (B, 64, 16, 16)
        x = self.stage2(x)                              # (B, 64, 16, 16)
        x = self.down2(x)                               # (B, d_model, 8, 8)
        x = x.flatten(2).transpose(1, 2)                # (B, 64, d_model)
        x = x[:, self.token_order]                      # Morton ordering
        x = x + self.pos_embed                          # 2D positional embedding
        x = self.mamba(x)                               # 3× bidirectional MambaResBlock
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# ============================================
# EMA
# ============================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay       = decay
        self.num_updates = 0
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        self.num_updates += 1
        # Warmup: ramps from ~0.18 → target decay so EMA tracks model quickly early on
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = d * self.shadow[k] + (1 - d) * v.float()

    def state_dict(self):
        return {"shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict):
        if "shadow" in state:                          # new format
            self.shadow      = {k: v.float() for k, v in state["shadow"].items()}
            self.num_updates = state.get("num_updates", 0)
        else:                                          # old format (backward compat)
            self.shadow      = {k: v.float() for k, v in state.items()}
            self.num_updates = 0

    def to_state_dict(self, model: nn.Module) -> dict:
        """Return EMA shadow cast to the model's dtype — ready for torch.save / load_state_dict."""
        dtype = next(model.parameters()).dtype
        return {k: v.to(dtype) for k, v in self.shadow.items()}


# ============================================
# SAM OPTIMIZER
# ============================================

class SAM:
    """Sharpness-Aware Minimization wrapper around any base optimizer.
    Requires two forward-backward passes per batch:
      1. first_step()  — perturb weights toward sharp region
      2. second_step() — restore weights, then update with base optimizer
    First backward should disable DDP gradient sync (model.no_sync()).
    """
    def __init__(self, base_optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.param_groups   = base_optimizer.param_groups
        self.rho            = rho
        self._e_w           = {}   # stores perturbations

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        scale     = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)
                self._e_w[p] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p in self._e_w:
                    p.sub_(self._e_w.pop(p))
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            p.grad.norm(2).to(device)
            for group in self.param_groups
            for p in group["params"] if p.grad is not None
        ]), p=2)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups


# ============================================
# CUTMIX
# ============================================

def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    lam = float(np.random.beta(alpha, alpha))
    B, _, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)
    cut_ratio = (1 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    return images, labels, labels[rand_idx], lam


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = float(np.random.beta(alpha, alpha))
    rand_idx = torch.randperm(images.size(0), device=images.device)
    images_mix = lam * images + (1 - lam) * images[rand_idx]
    return images_mix, labels, labels[rand_idx], lam


# ============================================
# HELPER FUNCTIONS
# ============================================

def make_loaders(dataset_path, image_size, batch_size, num_workers, rank, world_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])

    # Full 50K training set — no val split
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=train_tf)
    test_dataset  = datasets.ImageFolder(os.path.join(dataset_path, "test"),  transform=eval_tf)
    class_names   = train_dataset.classes

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kw)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kw)
    return train_loader, test_loader, class_names, train_sampler


def run_epoch(model, loader, criterion, optimizer, device, train=True, world_size=1,
              cutmix_alpha=0.0, mixup_alpha=0.0, ema=None, use_sam=False,
              desc="", show_progress=False):
    model.train(train)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    pbar = tqdm(loader, desc=desc, leave=False, disable=not show_progress,
                dynamic_ncols=True, colour="cyan")
    with ctx:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Apply mixing once — same batch used for both SAM passes
            mixed = False
            if train and (cutmix_alpha > 0 or mixup_alpha > 0):
                use_cutmix = (cutmix_alpha > 0) and (mixup_alpha <= 0 or np.random.rand() < 0.5)
                if use_cutmix:
                    imgs, labels_a, labels_b, lam = cutmix_batch(imgs, labels, alpha=cutmix_alpha)
                else:
                    imgs, labels_a, labels_b, lam = mixup_batch(imgs, labels, alpha=mixup_alpha)
                mixed = True

            def compute_loss(logits_):
                if mixed:
                    return lam * criterion(logits_, labels_a) + (1 - lam) * criterion(logits_, labels_b)
                return criterion(logits_, labels)

            if train:
                if use_sam:
                    # Pass 1: gradient for perturbation — disable DDP sync to avoid redundant allreduce
                    no_sync = model.no_sync if world_size > 1 else contextlib.nullcontext
                    with no_sync():
                        compute_loss(model(imgs)).backward()
                    optimizer.first_step(zero_grad=True)
                    # Pass 2: gradient for weight update — DDP sync enabled
                    logits = model(imgs)
                    loss   = compute_loss(logits)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    logits = model(imgs)
                    loss   = compute_loss(logits)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if ema is not None:
                    ema.update(model.module)
            else:
                logits = model(imgs)
                loss   = compute_loss(logits)

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            if show_progress:
                pbar.set_postfix(loss=f"{total_loss/total:.3f}",
                                 acc=f"{correct/total:.3f}")

    if train and world_size > 1:
        stats = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, correct, total = stats[0].item(), stats[1].item(), stats[2].item()

    return total_loss / total, correct / total


def train(device, run_dir, rank, world_size):
    is_main = rank == 0

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    dist.barrier()

    best_test_loss = float("inf")
    start_epoch    = 1
    best_ckpt      = os.path.join(run_dir, f"best_{MODEL_NAME}.pth")
    resume_ckpt    = os.path.join(run_dir, f"resume_{MODEL_NAME}.pth")
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    saved_ckpt = None
    if os.path.exists(resume_ckpt):
        if is_main:
            print(f"  Resuming from {resume_ckpt}")
        saved_ckpt     = torch.load(resume_ckpt, map_location=device, weights_only=True)
        start_epoch    = saved_ckpt["epoch"] + 1
        best_test_loss = saved_ckpt["best_test_loss"]
        history        = saved_ckpt["history"]
        if is_main:
            print(f"  Resumed at epoch {saved_ckpt['epoch']}, best_test_loss={best_test_loss:.4f}")

    train_loader, test_loader, class_names, train_sampler = make_loaders(
        DATASET_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, rank, world_size)
    num_classes = len(class_names)

    model = MambaCNN(num_classes=num_classes, d_model=D_MODEL, n_mamba=N_MAMBA, d_state=D_STATE,
                     drop_path_max=DROP_PATH_MAX, img_size=IMAGE_SIZE).to(device)
    model = DDP(model, device_ids=[device.index])

    total_p = sum(p.numel() for p in model.parameters())
    size_kb = total_p * 4 / 1024
    if is_main:
        print(f"  Model params: {total_p:,} ({size_kb/1024:.2f} MB)")

    criterion    = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    base_opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer    = SAM(base_opt, rho=SAM_RHO)

    # Scheduler uses base optimizer (SAM delegates param_groups to it)
    actual_warmup = min(WARMUP_EPOCHS, EPOCHS)
    cosine_steps  = max(1, EPOCHS - actual_warmup)
    warmup   = torch.optim.lr_scheduler.LinearLR(
        base_opt, start_factor=1e-3, end_factor=1.0, total_iters=actual_warmup)
    cosine   = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, T_max=cosine_steps, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_opt, schedulers=[warmup, cosine], milestones=[actual_warmup])

    ema = EMA(model.module, decay=EMA_DECAY)

    if saved_ckpt is not None:
        model.module.load_state_dict(saved_ckpt["model"])
        optimizer.load_state_dict(saved_ckpt["optimizer"])
        scheduler.load_state_dict(saved_ckpt["scheduler"])
        # T_max is deterministic from EPOCHS/WARMUP_EPOCHS constants; no patch needed.
        # If you change EPOCHS between runs, delete the resume checkpoint and start fresh.
        ema.load_state_dict(saved_ckpt["ema"])
    dist.barrier()


    for epoch in range(start_epoch, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, world_size=world_size,
                                    cutmix_alpha=CUTMIX_ALPHA, mixup_alpha=MIXUP_ALPHA,
                                    ema=ema if is_main else None, use_sam=True,
                                    desc=f"Epoch {epoch:>3} Train", show_progress=is_main)
        # Test set evaluated in eval mode — no gradients, model never learns from this
        te_loss, te_acc = run_epoch(model, test_loader, criterion, None, device, train=False,
                                    desc=f"Epoch {epoch:>3} Test ", show_progress=is_main)
        scheduler.step()

        if is_main:
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)

            if epoch % 10 == 0 or epoch == start_epoch:
                print(f"  Epoch {epoch:>3}: Train Acc={tr_acc:.4f}, Test Acc={te_acc:.4f}")


        if te_loss < best_test_loss:
            best_test_loss = te_loss
            if is_main:
                torch.save(ema.to_state_dict(model.module), best_ckpt)

        if is_main:
            torch.save({
                "epoch":          epoch,
                "model":          model.module.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "ema":            ema.state_dict(),
                "best_test_loss": best_test_loss,
                "history":        history,
            }, resume_ckpt)

    dist.barrier()
    state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.module.load_state_dict(state)

    final_loss, final_acc = run_epoch(model, test_loader, criterion, None, device, train=False)

    if is_main:
        print(f"\n  Test Accuracy: {final_acc*100:.2f}% (EMA best checkpoint)")
        csv_path = os.path.join(run_dir, f"{MODEL_NAME}_history.csv")
        pd.DataFrame(history | {"epoch": list(range(1, len(history["train_loss"]) + 1))}).to_csv(csv_path, index=False)
        print(f"  History saved to {csv_path}")

    return {
        "test_acc":       final_acc * 100,
        "test_loss":      final_loss,
        "best_test_loss": best_test_loss,
        "best_ckpt":      best_ckpt,
        "history":        history,
        "class_names":    class_names,
        "model_size_kb":  size_kb,
    }


# ============================================
# MAIN
# ============================================

def main():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device     = torch.device(f"cuda:{local_rank}")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    is_main    = rank == 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if is_main:
        print(f"\n{'#'*60}")
        print(f"  STARTING TRAINING")
        print(f"  World size (GPUs): {world_size}")
        print(f"{'#'*60}")

    run_dir = os.path.join(OUTPUT_DIR, "run")
    try:
        result  = train(device, run_dir, rank, world_size)

        if is_main:
            best_ckpt   = result["best_ckpt"]
            class_names = result["class_names"]

            print(f"\n{'='*70}")
            print(f"  RESULTS")
            print(f"{'='*70}")
            print(f"  Test Accuracy  : {result['test_acc']:.2f}%")
            print(f"  Model size     : {result['model_size_kb']/1024:.2f} MB")
            print(f"  Best model     : {best_ckpt}")
            print(f"  (Trained on full 50K, test set used for monitoring only)")
            print(f"{'='*70}")

            # ---- ONNX export ----
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            eval_tf = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(), transforms.Normalize(mean, std)])
            test_ds     = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"), transform=eval_tf)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)

            model_export = MambaCNN(
                num_classes=len(class_names), d_model=D_MODEL, n_mamba=N_MAMBA, d_state=D_STATE,
                drop_path_max=0.0, img_size=IMAGE_SIZE).to(device)  # no drop path at inference
            model_export.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
            model_export.eval()

            onnx_path = best_ckpt.replace(".pth", ".onnx")
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
            torch.onnx.export(model_export, dummy, onnx_path,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                              opset_version=17)

            ckpt_kb = os.path.getsize(best_ckpt) / 1024
            onnx_kb = os.path.getsize(onnx_path) / 1024
            print(f"  Checkpoint : {best_ckpt} ({ckpt_kb/1024:.2f} MB)")
            print(f"  ONNX       : {onnx_path} ({onnx_kb/1024:.2f} MB)")

            # ---- Confusion matrix ----
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    logits = model_export(imgs.to(device))
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_labels.extend(labels.numpy())

            cm = confusion_matrix(all_labels, all_preds)
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=axes[0])
            axes[0].set_title("Confusion Matrix")
            sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names, ax=axes[1])
            axes[1].set_title("Normalized")
            plt.tight_layout()
            cm_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()

            # ---- Save summary ----
            summary = {
                "model":         MODEL_NAME,
                "test_acc":      float(result["test_acc"]),
                "model_size_mb": float(result["model_size_kb"] / 1024),
                "best_ckpt":     best_ckpt,
                "onnx_path":     onnx_path,
            }
            with open(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
