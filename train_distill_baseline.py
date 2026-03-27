#!/usr/bin/env python3
"""
Knowledge Distillation: ViT-B/16 Teacher → MambaCNN Student
Launch: torchrun --nproc_per_node=NUM_GPUS train_distill.py

Phase 1 (auto, rank-0 only): Fine-tune ViT-B/16 on CIFAR-10 as teacher (~99% acc)
Phase 2: Train MambaCNN student with soft labels from frozen teacher
"""

import os
import sys
import warnings
import json
import contextlib
import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import timm

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================

# ---- Teacher ----
TEACHER_MODEL   = "vit_base_patch16_224"   # timm model; pretrained=True → ImageNet weights
TEACHER_SIZE    = 224                       # ViT requires 224×224
TEACHER_CKPT    = "outputs_distill/teacher_vit.pth"
TEACHER_EPOCHS  = 50
TEACHER_LR      = 5e-5
TEACHER_BATCH   = 128

# ---- Distillation ----
DISTILL_TEMP    = 4.0    # temperature — softens teacher distribution
DISTILL_ALPHA   = 0.1   # weight on hard CE loss  (1-alpha on soft KL loss)

# ---- Student (mirrors baseline) ----
MODEL_NAME        = "mamba_cnn_distill"
DATASET_PATH      = "/home/user2/Documents/classification/cifar10"
IMAGE_SIZE        = 32
BATCH_SIZE        = 256
EPOCHS            = 200
WARMUP_EPOCHS     = 10
LR                = 4e-3
WEIGHT_DECAY      = 0.05
LABEL_SMOOTHING   = 0.1
SAM_RHO           = 0.05
NUM_WORKERS       = 4
D_MODEL           = 96
N_MAMBA           = 3
D_STATE           = 8
EMA_DECAY         = 0.9999
DROP_PATH_MAX     = 0.1

OUTPUT_DIR        = "outputs_distill"


# ============================================
# STUDENT MODEL  (identical to train_mamba.py)
# ============================================

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class StemBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.se = SEBlock(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.se(self.pw(self.dw(x))) + self.shortcut(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.d_inner        = d_model * expand
        self.dt_rank        = max(1, self.d_inner // 16)
        self.d_state        = d_state
        self.drop_path_prob = drop_path
        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d  = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                 padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                           .unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log    = nn.Parameter(A_init)
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, x):
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

    def forward(self, x):
        residual = x
        x   = self.norm(x)
        xz  = self.in_proj(x)
        x_h, z = xz.chunk(2, dim=-1)
        x_h = self.conv1d(x_h.transpose(1, 2))[:, :, :x_h.shape[1]].transpose(1, 2)
        x_h = F.silu(x_h)
        y   = (self._scan(x_h) + self._scan(x_h.flip(1)).flip(1)) * 0.5  # bidirectional
        branch = self.out_proj(y * F.silu(z))
        if self.training and self.drop_path_prob > 0:
            s      = 1.0 - self.drop_path_prob
            mask   = torch.bernoulli(torch.full((x.size(0),1,1), s, device=x.device, dtype=x.dtype))
            branch = branch * mask / s
        return branch + residual


def _morton_order(h, w):
    def spread(n):
        n &= 0xFFFF
        n = (n|(n<<8))&0x00FF00FF; n = (n|(n<<4))&0x0F0F0F0F
        n = (n|(n<<2))&0x33333333; n = (n|(n<<1))&0x55555555
        return n
    codes = [spread(c)|(spread(r)<<1) for r in range(h) for c in range(w)]
    return torch.argsort(torch.tensor(codes))


class MambaCNN(nn.Module):
    def __init__(self, num_classes, d_model=96, n_mamba=3, d_state=8, drop_path_max=0.1, img_size=32):
        super().__init__()
        self.entry  = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.stage1 = nn.Sequential(StemBlock(32,32), StemBlock(32,32))
        self.down1  = nn.Sequential(StemBlock(32,64),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(StemBlock(64,64), StemBlock(64,64))
        self.down2  = nn.Sequential(StemBlock(64,d_model), nn.MaxPool2d(2))
        h_out = img_size // 4  # two MaxPool2d(2) reductions
        self.pos_embed = nn.Parameter(torch.zeros(1, h_out * h_out, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.register_buffer("token_order", _morton_order(h_out, h_out))
        dp = [drop_path_max * i / max(n_mamba-1, 1) for i in range(n_mamba)]
        self.mamba = nn.Sequential(*[MambaBlock(d_model, d_state, drop_path=dp[i]) for i in range(n_mamba)])
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.entry(x); x = self.stage1(x)
        x = self.down1(x); x = self.stage2(x); x = self.down2(x)
        x = x.flatten(2).transpose(1,2)
        x = x[:, self.token_order] + self.pos_embed
        x = self.mamba(x); x = self.norm(x)
        return self.head(x.mean(1))


# ============================================
# EMA
# ============================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay; self.num_updates = 0
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}
    def update(self, model):
        self.num_updates += 1
        d = min(self.decay, (1+self.num_updates)/(10+self.num_updates))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = d*self.shadow[k] + (1-d)*v.float()
    def state_dict(self):
        return {"shadow": self.shadow, "num_updates": self.num_updates}
    def load_state_dict(self, state):
        if "shadow" in state:
            self.shadow = {k: v.float() for k,v in state["shadow"].items()}
            self.num_updates = state.get("num_updates", 0)
        else:
            self.shadow = {k: v.float() for k,v in state.items()}
    def to_state_dict(self, model):
        dtype = next(model.parameters()).dtype
        return {k: v.to(dtype) for k,v in self.shadow.items()}


# ============================================
# SAM OPTIMIZER
# ============================================

class SAM:
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.param_groups   = base_optimizer.param_groups
        self.rho = rho; self._e_w = {}
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self.rho / (self._grad_norm() + 1e-12)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                e = p.grad * scale.to(p.device); p.add_(e); self._e_w[p] = e
        if zero_grad: self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for g in self.param_groups:
            for p in g["params"]:
                if p in self._e_w: p.sub_(self._e_w.pop(p))
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    def zero_grad(self): self.base_optimizer.zero_grad()
    def _grad_norm(self):
        dev = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([p.grad.norm(2).to(dev)
            for g in self.param_groups for p in g["params"] if p.grad is not None]), 2)
    def state_dict(self): return self.base_optimizer.state_dict()
    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)
        self.param_groups = self.base_optimizer.param_groups


# ============================================
# DISTILLATION LOSS
# ============================================

def distill_loss(s_logits, t_logits, labels, temp=4.0, alpha=0.1):
    """
    alpha  * CE(student, hard_labels)
    (1-alpha) * T² * KL(student_soft || teacher_soft)
    """
    hard = F.cross_entropy(s_logits, labels, label_smoothing=LABEL_SMOOTHING)
    s_soft = F.log_softmax(s_logits / temp, dim=-1)
    t_soft = F.softmax(t_logits  / temp, dim=-1)
    soft = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temp ** 2)
    return alpha * hard + (1 - alpha) * soft


# ============================================
# DATA LOADERS
# ============================================

def make_student_loaders(rank, world_size):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"),  transform=eval_tf)
    sampler  = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, **kw),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,   **kw),
            train_ds.classes, sampler)


def make_teacher_loaders():
    """224×224 loaders for teacher fine-tuning (single-GPU, rank 0 only)."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((TEACHER_SIZE, TEACHER_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([
        transforms.Resize((TEACHER_SIZE, TEACHER_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"),  transform=eval_tf)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True, batch_size=TEACHER_BATCH)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(test_ds, shuffle=False, **kw)


# ============================================
# PHASE 1 — TEACHER FINE-TUNING (rank 0 only)
# ============================================

TEACHER_RESUME_CKPT = "outputs_distill/teacher_resume.pth"

def finetune_teacher(device):
    """Fine-tune ViT-B/16 on CIFAR-10. Runs on rank 0, single GPU. Fully resumable."""
    os.makedirs(os.path.dirname(TEACHER_CKPT), exist_ok=True)

    teacher   = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=TEACHER_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc    = 0.0
    start_epoch = 1

    # Resume teacher if checkpoint exists
    if os.path.exists(TEACHER_RESUME_CKPT):
        ckpt        = torch.load(TEACHER_RESUME_CKPT, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc    = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"\n  [Teacher] Resuming from epoch {ckpt['epoch']}, best_acc={best_acc:.4f}")
    else:
        total_p = sum(p.numel() for p in teacher.parameters())
        print(f"\n  [Teacher] Fine-tuning {TEACHER_MODEL} for {TEACHER_EPOCHS} epochs "
              f"({total_p/1e6:.1f}M params)...")

    train_loader, test_loader = make_teacher_loaders()

    for epoch in range(start_epoch, TEACHER_EPOCHS + 1):
        # Train
        teacher.train()
        for imgs, labels in tqdm(train_loader, desc=f"[Teacher] Epoch {epoch:>3} Train",
                                 leave=False, dynamic_ncols=True, colour="green"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(teacher(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval
        teacher.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"[Teacher] Epoch {epoch:>3} Test ",
                                     leave=False, dynamic_ncols=True):
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (teacher(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total

        if epoch % 10 == 0 or epoch == start_epoch:
            print(f"  [Teacher] Epoch {epoch:>3}: Test Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(teacher.state_dict(), TEACHER_CKPT)   # best weights only

        # Resume checkpoint — saves every epoch
        torch.save({
            "epoch":     epoch,
            "model":     teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc":  best_acc,
        }, TEACHER_RESUME_CKPT)

    print(f"  [Teacher] Best Test Acc: {best_acc*100:.2f}% — saved to {TEACHER_CKPT}")
    return best_acc


# ============================================
# PHASE 2 — STUDENT DISTILLATION TRAINING
# ============================================

def train_student(device, run_dir, rank, world_size):
    is_main = rank == 0

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
    dist.barrier()

    # ---- Load frozen teacher on every rank ----
    teacher = timm.create_model(TEACHER_MODEL, pretrained=False, num_classes=10).to(device)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device, weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if is_main:
        print(f"  [Teacher] Loaded from {TEACHER_CKPT} (frozen)")

    # ---- Student ----
    train_loader, test_loader, class_names, train_sampler = make_student_loaders(rank, world_size)
    num_classes = len(class_names)

    student = MambaCNN(num_classes, D_MODEL, N_MAMBA, D_STATE, DROP_PATH_MAX, IMAGE_SIZE).to(device)
    student = DDP(student, device_ids=[device.index])

    total_p = sum(p.numel() for p in student.parameters())
    if is_main:
        print(f"  [Student] Params: {total_p:,} ({total_p*4/1024/1024:.2f} MB)")

    base_opt  = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = SAM(base_opt, rho=SAM_RHO)

    actual_warmup = min(WARMUP_EPOCHS, EPOCHS)
    cosine_steps  = max(1, EPOCHS - actual_warmup)
    warmup   = torch.optim.lr_scheduler.LinearLR(base_opt, 1e-3, 1.0, total_iters=actual_warmup)
    cosine   = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=cosine_steps, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(base_opt, [warmup, cosine], milestones=[actual_warmup])

    ema = EMA(student.module, decay=EMA_DECAY)

    best_test_loss = float("inf")
    start_epoch    = 1
    best_ckpt      = os.path.join(run_dir, f"best_{MODEL_NAME}.pth")
    resume_ckpt    = os.path.join(run_dir, f"resume_{MODEL_NAME}.pth")
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    if os.path.exists(resume_ckpt):
        if is_main: print(f"  Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=True)
        student.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        # T_max is deterministic from EPOCHS/WARMUP_EPOCHS constants; no patch needed.
        # If you change EPOCHS between runs, delete the resume checkpoint and start fresh.
        ema.load_state_dict(ckpt["ema"])
        start_epoch    = ckpt["epoch"] + 1
        best_test_loss = ckpt["best_test_loss"]
        history        = ckpt["history"]
        if is_main:
            print(f"  Resumed at epoch {ckpt['epoch']}, best_test_loss={best_test_loss:.4f}")
    dist.barrier()


    for epoch in range(start_epoch, EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        student.train()
        tr_loss = tr_correct = tr_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>3} Train", leave=False,
                    disable=not is_main, dynamic_ncols=True, colour="cyan")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Teacher sees 224×224 (bilinear upsample from student's 32×32 input)
            with torch.no_grad():
                imgs_224     = F.interpolate(imgs, (TEACHER_SIZE, TEACHER_SIZE),
                                             mode="bilinear", align_corners=False)
                t_logits     = teacher(imgs_224)   # soft targets — no grad

            def step(sync=True):
                s_logits = student(imgs)
                loss     = distill_loss(s_logits, t_logits, labels, DISTILL_TEMP, DISTILL_ALPHA)
                loss.backward()
                return s_logits, loss

            # SAM first pass (no DDP sync)
            no_sync = student.no_sync if world_size > 1 else contextlib.nullcontext
            with no_sync():
                s_logits, _ = step()
            optimizer.first_step(zero_grad=True)

            # SAM second pass (with DDP sync)
            s_logits, loss = step()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)

            if is_main:
                ema.update(student.module)

            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (s_logits.argmax(1) == labels).sum().item()
            tr_total   += imgs.size(0)
            if is_main:
                pbar.set_postfix(loss=f"{tr_loss/tr_total:.3f}",
                                 acc=f"{tr_correct/tr_total:.3f}")

        if world_size > 1:
            stats = torch.tensor([tr_loss, tr_correct, tr_total], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            tr_loss, tr_correct, tr_total = stats[0].item(), stats[1].item(), stats[2].item()

        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # ---- Test (eval mode, no grad, model never learns from this) ----
        student.eval()
        te_loss = te_correct = te_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch:>3} Test ",
                                     leave=False, disable=not is_main, dynamic_ncols=True):
                imgs, labels = imgs.to(device), labels.to(device)
                logits   = student(imgs)
                te_loss    += F.cross_entropy(logits, labels).item() * imgs.size(0)
                te_correct += (logits.argmax(1) == labels).sum().item()
                te_total   += imgs.size(0)
        te_loss /= te_total
        te_acc   = te_correct / te_total

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
                torch.save(ema.to_state_dict(student.module), best_ckpt)

            torch.save({
                "epoch": epoch, "model": student.module.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(), "best_test_loss": best_test_loss, "history": history,
            }, resume_ckpt)

    dist.barrier()

    # ---- Final evaluation with best EMA checkpoint ----
    student.module.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    student.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (student(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    final_acc = correct / total

    if is_main:
        print(f"\n  Test Accuracy (EMA best): {final_acc*100:.2f}%")
        csv_path = os.path.join(run_dir, f"{MODEL_NAME}_history.csv")
        epochs = list(range(1, len(history["train_loss"]) + 1))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss", "train_acc", "test_acc"])
            writer.writeheader()
            for i, ep in enumerate(epochs):
                writer.writerow({"epoch": ep, "train_loss": history["train_loss"][i],
                                 "test_loss": history["test_loss"][i], "train_acc": history["train_acc"][i],
                                 "test_acc": history["test_acc"][i]})
        print(f"  History saved to {csv_path}")

    return final_acc, class_names, total_p


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
        print(f"  KNOWLEDGE DISTILLATION TRAINING")
        print(f"  World size (GPUs): {world_size}")
        print(f"  Teacher : {TEACHER_MODEL} (fine-tuned on CIFAR-10)")
        print(f"  Student : MambaCNN  D={D_MODEL}, N={N_MAMBA}, Ds={D_STATE}")
        print(f"  Temp={DISTILL_TEMP}, Alpha={DISTILL_ALPHA}")
        print(f"{'#'*60}")

    # ---- Phase 1: Teacher fine-tuning (rank 0 only, skipped if ckpt exists) ----
    if is_main and not os.path.exists(TEACHER_CKPT):
        finetune_teacher(device)
    dist.barrier()   # all ranks wait until teacher is ready

    # ---- Phase 2: Student distillation ----
    run_dir = os.path.join(OUTPUT_DIR, "run")
    try:
        final_acc, class_names, total_p = train_student(device, run_dir, rank, world_size)
        if is_main:
            print(f"\n{'='*60}")
            print(f"  DISTILLATION RESULTS")
            print(f"{'='*60}")
            print(f"  Test Accuracy : {final_acc*100:.2f}%")
            print(f"  Student params: {total_p:,}")
            print(f"  Teacher       : {TEACHER_MODEL}")
            print(f"{'='*60}")
            summary = {"teacher": TEACHER_MODEL, "test_acc": float(final_acc*100),
                       "distill_temp": DISTILL_TEMP, "distill_alpha": DISTILL_ALPHA,
                       "student_params": total_p}
            with open(os.path.join(OUTPUT_DIR, "distill_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        import subprocess, sys
        nproc = max(1, torch.cuda.device_count())
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--master_port=29501",
            *sys.argv,
        ]
        subprocess.run(cmd, check=True)
    else:
        main()
