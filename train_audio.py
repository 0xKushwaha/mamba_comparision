#!/usr/bin/env python3
"""
Audio experiment: Google Speech Commands v2
MFCC + delta + delta-delta features (1D) → embedding stem → sequence block (no CNN).
Tests SSM contributions in pure audio domain.
"""

import argparse, random, os, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

SAMPLE_RATE   = 16000
N_MFCC        = 40
N_FEATURES    = N_MFCC * 3   # MFCC + delta + delta-delta
TARGET_SEQ_LEN = 101


class MLPBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Compute actual Mamba S6 param count (d_state=16, d_conv=4, expand=2)
        # so hidden_dim makes MLP exactly parameter-matched to one MambaBlock
        d_inner  = 2 * d_model
        dt_rank  = max(1, d_model // 16)
        mamba_blk = (
            d_model * d_inner * 2           # in_proj  (no bias)
          + d_inner * 4 + d_inner           # conv1d   weight + bias
          + d_inner * (dt_rank + 32)        # x_proj   (d_state=16 → +32)
          + dt_rank * d_inner + d_inner     # dt_proj  weight + bias
          + d_inner * d_model               # out_proj (no bias)
          + d_inner * 16 + d_inner          # A_log + D
          + 2 * d_model                     # LayerNorm
        )
        # MLP block params = (2*d_model+1)*h + 3*d_model  →  solve for h
        hidden_dim = max(d_model, -(-( mamba_blk - 3 * d_model) // (2 * d_model + 1)))
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
    "mlp":         lambda d: MLPBlock(d),
    "mamba_uni":   lambda d: MambaBlock(d, False),
    "mamba_bi":    lambda d: MambaBlock(d, True),
    "attn":        lambda d: AttentionBlock(d),
}


class AudioClassifier(nn.Module):
    def __init__(self, variant: str, d_model: int, n_classes: int, n_layers: int = 6):
        super().__init__()
        self.embed = nn.Linear(N_FEATURES, d_model)
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


# ── Precompute helpers (thread-local so each thread owns its MFCC transform) ──

_tl = threading.local()

def _compute_one(idx, path, mm):
    if not hasattr(_tl, "mfcc_fn"):
        import warnings
        warnings.filterwarnings("ignore")
        _tl.mfcc_fn = T.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC)
    wav, sr = torchaudio.load(path)
    label = os.path.basename(os.path.dirname(path))
    if sr != SAMPLE_RATE:
        wav = T.Resample(sr, SAMPLE_RATE)(wav)
    wav = wav[:, :SAMPLE_RATE]
    if wav.size(1) < SAMPLE_RATE:
        wav = nn.functional.pad(wav, (0, SAMPLE_RATE - wav.size(1)))
    mfcc   = _tl.mfcc_fn(wav)
    delta  = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    feat = torch.cat([mfcc, delta, delta2], dim=1)   # (1, N_FEATURES, T)
    feat = (feat - feat.mean()) / (feat.std() + 1e-6)
    if feat.size(2) < TARGET_SEQ_LEN:
        feat = nn.functional.pad(feat, (0, TARGET_SEQ_LEN - feat.size(2)))
    else:
        feat = feat[:, :, :TARGET_SEQ_LEN]
    mm[idx] = feat.squeeze(0).T.numpy()              # (T, N_FEATURES)
    return idx, label


# ── Dataset ──────────────────────────────────────────────────────────────────

# All 35 Speech Commands v2 classes — hardcoded to avoid scanning ~105k files at init
_SPEECH_COMMANDS_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]
_LABEL_MAP = {l: i for i, l in enumerate(_SPEECH_COMMANDS_LABELS)}


class SpeechCommandsDataset(Dataset):
    def __init__(self, root, subset):
        self.root   = root
        self.subset = subset
        self.ds     = torchaudio.datasets.SPEECHCOMMANDS(root, subset=subset, download=True)
        self.label_map = _LABEL_MAP
        self.cache_dir = os.path.join(root, f"mfcc_cache_{subset}")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._precompute()

    def _precompute(self, num_workers=16):
        n           = len(self.ds)
        mm_path     = os.path.join(self.cache_dir, "all_mfcc.npy")
        labels_path = os.path.join(self.cache_dir, "labels.pt")

        if not os.path.exists(mm_path):
            print(f"Precomputing {n} MFCC features ({num_workers} threads) ...", flush=True)
            mm     = np.memmap(mm_path, dtype="float32", mode="w+",
                               shape=(n, TARGET_SEQ_LEN, N_FEATURES))
            walker    = list(self.ds._walker)
            label_buf = [None] * n
            counter   = [0]
            lock      = threading.Lock()

            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                futures = {ex.submit(_compute_one, i, walker[i], mm): i for i in range(n)}
                for f in as_completed(futures):
                    idx, label = f.result()
                    label_buf[idx] = label
                    with lock:
                        counter[0] += 1
                        if counter[0] % 5000 == 0:
                            print(f"  {counter[0]}/{n}", flush=True)

            mm.flush()
            label_ids = [self.label_map[l] for l in label_buf]
            torch.save(torch.tensor(label_ids, dtype=torch.long), labels_path)

        print(f"Loading {self.subset} into RAM ...", flush=True)
        mm = np.memmap(mm_path, dtype="float32", mode="r",
                       shape=(n, TARGET_SEQ_LEN, N_FEATURES))
        self.mfccs  = torch.from_numpy(np.array(mm))
        self.labels = torch.load(labels_path, weights_only=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mfccs[idx], self.labels[idx]


# ── GPU-native loader (no DataLoader overhead, shuffle on GPU) ────────────────

class GPULoader:
    """Replaces DataLoader for GPU-resident tensors. All ops stay on GPU."""
    def __init__(self, X, y, batch_size, shuffle=False):
        self.X, self.y   = X, y
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.n           = len(y)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.X.device) if self.shuffle \
              else torch.arange(self.n, device=self.X.device)
        for start in range(0, self.n, self.batch_size):
            b = idx[start : start + self.batch_size]
            yield self.X[b], self.y[b]


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optim, scaler):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = n = 0
    for x, y in loader:
        optim.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss   = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = n = 0
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss   = loss_fn(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


def run(args, variant, seed, train_loader, val_loader, test_loader, n_classes, device):
    set_seed(seed)

    model  = AudioClassifier(variant, args.d_model, n_classes, args.n_layers).to(device)
    scaler = torch.cuda.amp.GradScaler()
    optim  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched  = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state    = None
    patience      = args.patience
    no_improve    = 0
    for _ in range(args.epochs):
        train_epoch(model, train_loader, optim, scaler)
        val_loss, _ = eval_model(model, val_loader)
        sched.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    _, test_acc = eval_model(model, test_loader)
    return test_acc


_DEFAULT_VARIANTS = ["mlp", "mamba_uni", "mamba_bi", "attn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",     default="./data")
    parser.add_argument("--d_model",       type=int,   default=256)
    parser.add_argument("--n_layers",      type=int,   default=6)
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--lr",            type=float, default=2e-3)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--batch_size",    type=int,   default=512)
    parser.add_argument("--n_seeds",       type=int,   default=3)
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--variants",  nargs="+", default=_DEFAULT_VARIANTS)
    parser.add_argument("--seq_lens",  nargs="+", type=int, default=[32, 64],
                        help="Sequence lengths to evaluate (each must be <= %(default)s). "
                             "Example: --seq_lens 32 64 101")
    args = parser.parse_args()

    for sl in args.seq_lens:
        if sl > TARGET_SEQ_LEN:
            raise ValueError(f"--seq_lens {sl} exceeds cache length {TARGET_SEQ_LEN}. "
                             f"Change TARGET_SEQ_LEN and delete the cache to rebuild.")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading datasets (one time)...")
    train_ds  = SpeechCommandsDataset(args.data_root, "training")
    val_ds    = SpeechCommandsDataset(args.data_root, "validation")
    test_ds   = SpeechCommandsDataset(args.data_root, "testing")
    n_classes = len(train_ds.label_map)

    print(f"Moving datasets to {device} ...")
    for ds in (train_ds, val_ds, test_ds):
        ds.mfccs  = ds.mfccs.to(device)
        ds.labels = ds.labels.to(device)

    all_results = {}   # (seq_len, variant) → (mean, std)

    for seq_len in args.seq_lens:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}")
        print(f"{'='*60}")

        # Slice to desired length — no copy, just a view
        train_loader = GPULoader(train_ds.mfccs[:, :seq_len, :], train_ds.labels,
                                 args.batch_size, shuffle=True)
        val_loader   = GPULoader(val_ds.mfccs[:,   :seq_len, :], val_ds.labels,
                                 args.batch_size, shuffle=False)
        test_loader  = GPULoader(test_ds.mfccs[:,  :seq_len, :], test_ds.labels,
                                 args.batch_size, shuffle=False)

        for variant in args.variants:
            if "mamba" in variant and not MAMBA_AVAILABLE:
                print(f"[SKIP] {variant}: mamba_ssm not installed")
                continue
            accs = []
            for seed in range(args.n_seeds):
                acc = run(args, variant, seed,
                          train_loader, val_loader, test_loader, n_classes, device)
                accs.append(acc * 100)
                print(f"{variant:20s}  seed={seed}  acc={acc*100:.2f}%")
            m, s = np.mean(accs), np.std(accs)
            all_results[(seq_len, variant)] = (m, s)
            print(f"  → {m:.2f} ± {s:.2f}\n")

    print("\n=== Speech Commands Results ===")
    for seq_len in args.seq_lens:
        print(f"\nseq_len={seq_len}")
        for variant in args.variants:
            key = (seq_len, variant)
            if key in all_results:
                m, s = all_results[key]
                print(f"  {variant:20s}  {m:.2f} ± {s:.2f}")


if __name__ == "__main__":
    main()
