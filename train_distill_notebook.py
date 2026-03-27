"""
Knowledge Distillation: ViT-B/16 Teacher → MambaCNN Student
Single-GPU / CPU version — paste into a Jupyter cell and run.
"""

import os, warnings, json, contextlib, csv
import numpy as np
from tqdm.notebook import tqdm          # notebook-friendly progress bars

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import timm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION  — edit these as needed
# ============================================================

# ---- Teacher ----
TEACHER_MODEL  = "vit_base_patch16_224"
TEACHER_SIZE   = 224
TEACHER_CKPT   = (("/kaggle/working/outputs_distill/teacher_vit.pth")
                  if os.path.isdir("/kaggle/working") else "outputs_distill/teacher_vit.pth")
TEACHER_EPOCHS = 10          # reduce for quick tests; 50 for full training
TEACHER_LR     = 5e-5
TEACHER_BATCH  = 64          # lower than original to fit most GPUs

# ---- Distillation curriculum ----
DISTILL_TEMP_START  = 8.0
DISTILL_TEMP_END    = 2.0
DISTILL_ALPHA_START = 0.5
DISTILL_ALPHA_END   = 0.05

# ---- Student ----
MODEL_NAME      = "mamba_cnn_distill"
DATA_DIR        = ("/kaggle/working/cifar10_data"
                   if os.path.isdir("/kaggle/working") else "./cifar10_data")
IMAGE_SIZE      = 32
BATCH_SIZE      = 128
EPOCHS          = 50                 # set to 200 for full training
WARMUP_EPOCHS   = 5
LR              = 4e-3
WEIGHT_DECAY    = 0.05
SAM_RHO         = 0.05
NUM_WORKERS     = 2
D_MODEL         = 96
N_MAMBA         = 3
D_STATE         = 8
EMA_DECAY       = 0.9999
DROP_PATH_MAX   = 0.1

OUTPUT_DIR = ("/kaggle/working/outputs_distill"
              if os.path.isdir("/kaggle/working") else "outputs_distill")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# STUDENT MODEL
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class StemBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.GELU())
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.se = SEBlock(out_ch)
        self.shortcut = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                        nn.BatchNorm2d(out_ch))
                         if in_ch != out_ch else nn.Identity())
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.se(self.pw(self.dw(x))) + self.shortcut(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, drop_path=0.0):
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
        y   = (self._scan(x_h) + self._scan(x_h.flip(1)).flip(1)) * 0.5
        branch = self.out_proj(y * F.silu(z))
        if self.training and self.drop_path_prob > 0:
            s    = 1.0 - self.drop_path_prob
            mask = torch.bernoulli(torch.full((x.size(0),1,1), s, device=x.device, dtype=x.dtype))
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
    def __init__(self, num_classes, d_model=96, n_mamba=3, d_state=8,
                 drop_path_max=0.1, img_size=32):
        super().__init__()
        self.entry  = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False),
                                    nn.BatchNorm2d(32), nn.GELU())
        self.stage1 = nn.Sequential(StemBlock(32,32), StemBlock(32,32))
        self.down1  = nn.Sequential(StemBlock(32,64),  nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(StemBlock(64,64), StemBlock(64,64))
        self.down2  = nn.Sequential(StemBlock(64,d_model), nn.MaxPool2d(2))
        h_out = img_size // 4
        self.pos_embed = nn.Parameter(torch.zeros(1, h_out * h_out, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.register_buffer("token_order", _morton_order(h_out, h_out))
        dp = [drop_path_max * i / max(n_mamba-1, 1) for i in range(n_mamba)]
        self.mamba = nn.Sequential(*[MambaBlock(d_model, d_state, drop_path=dp[i])
                                     for i in range(n_mamba)])
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.entry(x); x = self.stage1(x)
        x = self.down1(x); x = self.stage2(x); x = self.down2(x)
        x = x.flatten(2).transpose(1,2)
        x = x[:, self.token_order] + self.pos_embed
        x = self.mamba(x); x = self.norm(x)
        return self.head(x.mean(1))


# ============================================================
# EMA
# ============================================================

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


# ============================================================
# SAM OPTIMIZER
# ============================================================

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


# ============================================================
# DISTILLATION LOSS + SCHEDULE
# ============================================================

def get_distill_schedule(epoch, total_epochs):
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    temp  = DISTILL_TEMP_START  - (DISTILL_TEMP_START  - DISTILL_TEMP_END)  * progress
    alpha = DISTILL_ALPHA_START - (DISTILL_ALPHA_START - DISTILL_ALPHA_END) * progress
    return temp, alpha


def distill_loss(s_logits, t_logits, labels, temp=4.0, alpha=0.1):
    hard  = F.cross_entropy(s_logits, labels)
    s_soft = F.log_softmax(s_logits / temp, dim=-1)
    t_soft = F.softmax(t_logits / temp, dim=-1)
    soft  = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temp ** 2)
    return alpha * hard + (1 - alpha) * soft


# ============================================================
# DATA LOADERS  (auto-download CIFAR-10 via torchvision)
# ============================================================

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

student_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(0.2, 0.2, 0.1),
    transforms.ToTensor(), transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
])
student_eval_tf = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean, std)])

teacher_train_tf = transforms.Compose([
    transforms.Resize((TEACHER_SIZE, TEACHER_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(), transforms.Normalize(mean, std)])
teacher_eval_tf = transforms.Compose([
    transforms.Resize((TEACHER_SIZE, TEACHER_SIZE)),
    transforms.ToTensor(), transforms.Normalize(mean, std)])

def _setup_cifar_root():
    """
    Find a complete CIFAR-10 extraction or prepare a clean directory for download.
    A complete extraction has exactly 5 data_batch_* files inside cifar-10-batches-py.
    Incomplete dirs (from failed prior runs) are deleted automatically.
    """
    import shutil

    search_bases = []
    if os.path.isdir("/kaggle"):
        search_bases += ["/kaggle/working", "/kaggle/input"]
        # also scan one level deeper (e.g. /kaggle/input/some-dataset/)
        for base in list(search_bases):
            try:
                for name in os.listdir(base):
                    search_bases.append(os.path.join(base, name))
            except OSError:
                pass

    for root in search_bases:
        batches_dir = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(batches_dir):
            continue
        try:
            n_batches = sum(1 for f in os.listdir(batches_dir)
                            if f.startswith("data_batch_"))
        except OSError:
            continue
        if n_batches == 5:
            print(f"[Data] Complete CIFAR-10 found at {root}")
            return root, False          # (data_root, need_download)
        else:
            # Partial extraction — remove it so torchvision can start fresh
            try:
                shutil.rmtree(batches_dir)
                print(f"[Data] Removed incomplete extraction at {batches_dir}")
            except OSError:
                pass                    # read-only; skip, keep searching

    # Nothing usable found — download to writable working dir
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[Data] Downloading CIFAR-10 to {DATA_DIR}")
    return DATA_DIR, True

_data_root, _download = _setup_cifar_root()

_cifar_train_s = datasets.CIFAR10(_data_root, train=True,  download=_download, transform=student_train_tf)
_cifar_test_s  = datasets.CIFAR10(_data_root, train=False, download=False,     transform=student_eval_tf)
_cifar_train_t = datasets.CIFAR10(_data_root, train=True,  download=False,     transform=teacher_train_tf)
_cifar_test_t  = datasets.CIFAR10(_data_root, train=False, download=False,     transform=teacher_eval_tf)

kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
student_train_loader = DataLoader(_cifar_train_s, batch_size=BATCH_SIZE, shuffle=True,  **kw)
student_test_loader  = DataLoader(_cifar_test_s,  batch_size=BATCH_SIZE, shuffle=False, **kw)
teacher_train_loader = DataLoader(_cifar_train_t, batch_size=TEACHER_BATCH, shuffle=True,  **kw)
teacher_test_loader  = DataLoader(_cifar_test_t,  batch_size=TEACHER_BATCH, shuffle=False, **kw)

CLASS_NAMES = _cifar_train_s.classes
print(f"Classes: {CLASS_NAMES}")


# ============================================================
# PHASE 1 — TEACHER FINE-TUNING
# ============================================================

TEACHER_RESUME_CKPT = os.path.join(OUTPUT_DIR, "teacher_resume.pth")

def finetune_teacher():
    if os.path.exists(TEACHER_CKPT):
        print(f"[Teacher] Checkpoint found at {TEACHER_CKPT}, skipping fine-tuning.")
        return

    teacher   = timm.create_model(TEACHER_MODEL, pretrained=True, num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=TEACHER_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc  = 0.0
    start_epoch = 1

    if os.path.exists(TEACHER_RESUME_CKPT):
        ckpt = torch.load(TEACHER_RESUME_CKPT, map_location=DEVICE, weights_only=True)
        teacher.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc    = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"[Teacher] Resuming from epoch {ckpt['epoch']}, best_acc={best_acc:.4f}")
    else:
        total_p = sum(p.numel() for p in teacher.parameters())
        print(f"[Teacher] Fine-tuning {TEACHER_MODEL} — {total_p/1e6:.1f}M params")

    for epoch in range(start_epoch, TEACHER_EPOCHS + 1):
        teacher.train()
        for imgs, labels in tqdm(teacher_train_loader,
                                 desc=f"[Teacher] Epoch {epoch:>3} Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(teacher(imgs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        teacher.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(teacher_test_loader,
                                     desc=f"[Teacher] Epoch {epoch:>3} Test", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (teacher(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        print(f"[Teacher] Epoch {epoch:>3}: Test Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(teacher.state_dict(), TEACHER_CKPT)

        torch.save({"epoch": epoch, "model": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc}, TEACHER_RESUME_CKPT)

    print(f"[Teacher] Best Test Acc: {best_acc*100:.2f}%  →  saved to {TEACHER_CKPT}")


finetune_teacher()


# ============================================================
# PHASE 2 — STUDENT DISTILLATION TRAINING
# ============================================================

run_dir = os.path.join(OUTPUT_DIR, "run")
os.makedirs(run_dir, exist_ok=True)

# Load frozen teacher
teacher = timm.create_model(TEACHER_MODEL, pretrained=False, num_classes=10).to(DEVICE)
teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE, weights_only=True))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"[Teacher] Loaded from {TEACHER_CKPT} (frozen)")

# Build student
student = MambaCNN(len(CLASS_NAMES), D_MODEL, N_MAMBA, D_STATE, DROP_PATH_MAX, IMAGE_SIZE).to(DEVICE)
total_p = sum(p.numel() for p in student.parameters())
print(f"[Student] Params: {total_p:,}  ({total_p*4/1024/1024:.2f} MB)")

base_opt  = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
optimizer = SAM(base_opt, rho=SAM_RHO)

actual_warmup = min(WARMUP_EPOCHS, EPOCHS)
cosine_steps  = max(1, EPOCHS - actual_warmup)
warmup    = torch.optim.lr_scheduler.LinearLR(base_opt, 1e-3, 1.0, total_iters=actual_warmup)
cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=cosine_steps, eta_min=1e-6)
scheduler = torch.optim.lr_scheduler.SequentialLR(base_opt, [warmup, cosine], milestones=[actual_warmup])

ema = EMA(student, decay=EMA_DECAY)

best_test_loss = float("inf")
start_epoch    = 1
best_ckpt      = os.path.join(run_dir, f"best_{MODEL_NAME}.pth")
resume_ckpt    = os.path.join(run_dir, f"resume_{MODEL_NAME}.pth")
history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

if os.path.exists(resume_ckpt):
    print(f"Resuming from {resume_ckpt}")
    ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=True)
    student.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    ema.load_state_dict(ckpt["ema"])
    start_epoch    = ckpt["epoch"] + 1
    best_test_loss = ckpt["best_test_loss"]
    history        = ckpt["history"]
    print(f"Resumed at epoch {ckpt['epoch']}, best_test_loss={best_test_loss:.4f}")

for epoch in range(start_epoch, EPOCHS + 1):
    student.train()
    tr_loss = tr_correct = tr_total = 0
    distill_temp, distill_alpha = get_distill_schedule(epoch, EPOCHS)

    pbar = tqdm(student_train_loader, desc=f"Epoch {epoch:>3} Train", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            imgs_224 = F.interpolate(imgs, (TEACHER_SIZE, TEACHER_SIZE),
                                     mode="bilinear", align_corners=False)
            t_logits = teacher(imgs_224)

        # SAM first pass
        s_logits = student(imgs)
        loss = distill_loss(s_logits, t_logits, labels, distill_temp, distill_alpha)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # SAM second pass
        s_logits = student(imgs)
        loss = distill_loss(s_logits, t_logits, labels, distill_temp, distill_alpha)
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.second_step(zero_grad=True)

        ema.update(student)

        tr_loss    += loss.item() * imgs.size(0)
        tr_correct += (s_logits.argmax(1) == labels).sum().item()
        tr_total   += imgs.size(0)
        pbar.set_postfix(loss=f"{tr_loss/tr_total:.3f}", acc=f"{tr_correct/tr_total:.3f}")

    tr_loss /= tr_total
    tr_acc   = tr_correct / tr_total

    student.eval()
    te_loss = te_correct = te_total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(student_test_loader,
                                 desc=f"Epoch {epoch:>3} Test ", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = student(imgs)
            te_loss    += F.cross_entropy(logits, labels).item() * imgs.size(0)
            te_correct += (logits.argmax(1) == labels).sum().item()
            te_total   += imgs.size(0)
    te_loss /= te_total
    te_acc   = te_correct / te_total

    scheduler.step()

    history["train_loss"].append(tr_loss)
    history["test_loss"].append(te_loss)
    history["train_acc"].append(tr_acc)
    history["test_acc"].append(te_acc)

    print(f"Epoch {epoch:>3}: Train={tr_acc:.4f}  Test={te_acc:.4f} | "
          f"Temp={distill_temp:.2f}  Alpha={distill_alpha:.3f}")

    if te_loss < best_test_loss:
        best_test_loss = te_loss
        torch.save(ema.to_state_dict(student), best_ckpt)

    torch.save({"epoch": epoch, "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "best_test_loss": best_test_loss,
                "history": history}, resume_ckpt)


# ============================================================
# FINAL EVALUATION
# ============================================================

student.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
student.eval()
correct = total = 0
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in student_test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = student(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_acc = correct / total
print(f"\nFinal Test Accuracy (EMA best): {final_acc*100:.2f}%")

# Save CSV history
csv_path = os.path.join(run_dir, f"{MODEL_NAME}_history.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","test_loss","train_acc","test_acc"])
    writer.writeheader()
    for i in range(len(history["train_loss"])):
        writer.writerow({"epoch": i+1, "train_loss": history["train_loss"][i],
                         "test_loss": history["test_loss"][i],
                         "train_acc": history["train_acc"][i],
                         "test_acc":  history["test_acc"][i]})
print(f"History saved → {csv_path}")

# ---- Plots ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["test_loss"],  label="Test Loss")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history["train_acc"], label="Train Acc")
axes[1].plot(history["test_acc"],  label="Test Acc")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix — {final_acc*100:.2f}% Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=150)
plt.show()

# Save summary JSON
with open(os.path.join(OUTPUT_DIR, "distill_summary.json"), "w") as f:
    json.dump({"teacher": TEACHER_MODEL, "test_acc": round(final_acc*100, 4),
               "student_params": total_p}, f, indent=2)
print("Done.")
