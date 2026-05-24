#!/usr/bin/env python3
"""
Time-series experiment: UCI HAR (Human Activity Recognition) dataset
1D sequences → embedding stem → sequence block (no CNN).
Tests SSM contributions in pure temporal domain.
"""

import argparse, random, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from urllib.request import urlretrieve
import zipfile

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MLPBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        mamba_params = 4 * d_model * d_model
        hidden_dim = max(d_model, mamba_params // (2 * d_model))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.net(self.norm(x))


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        xn = self.norm1(x)
        x = x + self.attn(xn, xn, xn, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, bidirectional: bool = False):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise RuntimeError("mamba_ssm required for Mamba variants")
        self.bidirectional = bidirectional
        self.fwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        if bidirectional:
            self.bwd = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        xn = self.norm(x)
        out = self.fwd(xn)
        if self.bidirectional:
            out_b = self.bwd(xn.flip(1)).flip(1)
            out = self.proj(torch.cat([out, out_b], dim=-1))
        return x + out


VARIANTS = {
    "linear_pool": lambda _: None,
    "mlp": lambda d: MLPBlock(d),
    "mamba_uni": lambda d: MambaBlock(d, False),
    "mamba_bi": lambda d: MambaBlock(d, True),
    "attn": lambda d: AttentionBlock(d),
}


class TimeSeriesClassifier(nn.Module):
    def __init__(self, variant: str, d_model: int, n_features: int, n_classes: int, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        block_fn = VARIANTS[variant]
        if block_fn(d_model) is None:
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([VARIANTS[variant](d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embed(x)
        if self.blocks is None:
            x = x.mean(1)
        else:
            for block in self.blocks:
                x = block(x)
            x = self.norm(x).mean(1)
        return self.head(x)


INERTIAL_SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


class HARDataset(Dataset):
    """UCI HAR dataset using raw inertial signals: [128 timesteps x 9 channels]."""
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self._download_if_needed()
        self._load_data(train)

    def _download_if_needed(self):
        data_dir = os.path.join(self.data_root, "UCI HAR Dataset")
        if os.path.exists(data_dir):
            return
        os.makedirs(self.data_root, exist_ok=True)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = os.path.join(self.data_root, "har.zip")
        print(f"Downloading UCI HAR Dataset to {zip_path}...")
        urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.data_root)
        os.remove(zip_path)

    def _load_data(self, train):
        subset = "train" if train else "test"
        data_dir = os.path.join(self.data_root, "UCI HAR Dataset", subset, "Inertial Signals")
        channels = []
        for sig in INERTIAL_SIGNALS:
            path = os.path.join(data_dir, f"{sig}_{subset}.txt")
            channels.append(np.loadtxt(path))  # [N, 128]
        # Stack to [N, 128, 9] then convert
        X = np.stack(channels, axis=-1).astype(np.float32)
        # Normalize each channel independently
        X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-6)
        y_path = os.path.join(self.data_root, "UCI HAR Dataset", subset, f"y_{subset}.txt")
        self.X = torch.from_numpy(X)          # [N, 128, 9]
        self.y = torch.from_numpy(np.loadtxt(y_path)).long() - 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optim, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def run(args, variant, seed, train_loader, val_loader, test_loader, n_features, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    model = TimeSeriesClassifier(variant, args.d_model, n_features, n_classes, args.n_layers).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    for _ in range(args.epochs):
        train_epoch(model, train_loader, optim, device)
        val_loss, _ = eval_model(model, val_loader, device)
        sched.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, test_acc = eval_model(model, test_loader, device)
    return test_acc


_DEFAULT_VARIANTS = ["mlp", "mamba_uni", "mamba_bi", "attn"]
_FULL_SEQ_LEN = 128   # UCI HAR native window length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    default="./data")
    parser.add_argument("--d_model",      type=int,   default=128)
    parser.add_argument("--n_layers",     type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--lr",           type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--n_seeds",      type=int,   default=3)
    parser.add_argument("--variants",  nargs="+", default=_DEFAULT_VARIANTS)
    parser.add_argument("--seq_lens",  nargs="+", type=int, default=[32, 64],
                        help="Sequence lengths to evaluate (each must be <= 128). "
                             "Example: --seq_lens 32 64 128")
    args = parser.parse_args()

    for sl in args.seq_lens:
        if sl > _FULL_SEQ_LEN:
            raise ValueError(f"--seq_lens {sl} exceeds UCI HAR window length {_FULL_SEQ_LEN}.")

    print("Loading UCI HAR dataset (one time)...")
    train_ds = HARDataset(args.data_root, train=True)
    test_ds  = HARDataset(args.data_root, train=False)
    n_features = train_ds.X.size(2)   # 9 channels
    n_classes  = 6

    # Fixed train/val split — extract raw tensors so we can slice per seq_len
    n_total = len(train_ds)
    n_val   = int(0.2 * n_total)
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(n_total, generator=g)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_X = train_ds.X[train_idx]   # [N_train, 128, 9]
    train_y = train_ds.y[train_idx]
    val_X   = train_ds.X[val_idx]     # [N_val,   128, 9]
    val_y   = train_ds.y[val_idx]
    test_X  = test_ds.X               # [N_test,  128, 9]
    test_y  = test_ds.y

    kw = {"batch_size": args.batch_size, "num_workers": 4,
          "pin_memory": True, "persistent_workers": True}

    all_results = {}   # (seq_len, variant) → (mean, std)

    for seq_len in args.seq_lens:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}")
        print(f"{'='*60}")

        train_loader = DataLoader(TensorDataset(train_X[:, :seq_len, :], train_y),
                                  shuffle=True,  **kw)
        val_loader   = DataLoader(TensorDataset(val_X[:,   :seq_len, :], val_y),
                                  shuffle=False, **kw)
        test_loader  = DataLoader(TensorDataset(test_X[:,  :seq_len, :], test_y),
                                  shuffle=False, **kw)

        for variant in args.variants:
            if "mamba" in variant and not MAMBA_AVAILABLE:
                print(f"[SKIP] {variant}: mamba_ssm not installed")
                continue
            accs = []
            for seed in range(args.n_seeds):
                acc = run(args, variant, seed,
                          train_loader, val_loader, test_loader, n_features, n_classes)
                accs.append(acc * 100)
                print(f"{variant:20s}  seed={seed}  acc={acc*100:.2f}%")
            m, s = np.mean(accs), np.std(accs)
            all_results[(seq_len, variant)] = (m, s)
            print(f"  → {m:.2f} ± {s:.2f}\n")

    print("\n=== UCI HAR Results ===")
    for seq_len in args.seq_lens:
        print(f"\nseq_len={seq_len}")
        for variant in args.variants:
            key = (seq_len, variant)
            if key in all_results:
                m, s = all_results[key]
                print(f"  {variant:20s}  {m:.2f} ± {s:.2f}")


if __name__ == "__main__":
    main()
