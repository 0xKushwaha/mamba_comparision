"""
data.py — Dataset loaders for all three experimental datasets.

Supported datasets:
  dtd           — Describable Textures Dataset (torchvision, auto-download)
                  47 texture classes, 5640 images, official train/val/test splits
                  → Local texture task: CNN should dominate regardless of L

  stl10         — STL-10 (torchvision, auto-download)
                  10 object classes, 5000 train / 8000 test, native 96×96
                  → Mixed complexity: transition zone between texture and global

  tiny_imagenet — Tiny ImageNet (ImageFolder, manual download required)
                  200 classes, 100K images, resized to 96×96
                  → High complexity: global spatial reasoning, SSM should help

All three datasets are resized to a common 96×96 resolution so that
sequence lengths L ∈ {36, 144, 576} are identical across datasets.
This makes cross-dataset comparison clean and direct.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet normalization (standard for all three datasets)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ============================================================
# AUGMENTATION CONFIGS  (per-dataset policies)
# ============================================================

def _train_tf(img_size: int, dataset: str) -> transforms.Compose:
    """
    Per-dataset augmentation policy.

    DTD (texture):
      - Random H/V flips: textures have no canonical orientation
      - Gentle rotation: rotated textures are still the same texture
      - Moderate color jitter: lighting variations in texture images
      - NO RandAugment: can distort texture patterns (posterize, solarize, etc.)
      - NO RandomErasing: erasing removes the very thing we're classifying

    STL-10 / Tiny ImageNet (objects):
      - Random crop + horizontal flip: standard object augmentation
      - RandAugment: improves generalization for diverse object classes
      - Random erasing: occlusion robustness
    """
    base = [transforms.Resize((img_size, img_size))]

    if dataset == "dtd":
        base += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(img_size, padding=img_size // 10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ]
        base += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]

    else:  # stl10, cifar100, tiny_imagenet
        base += [
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(0.2, 0.2, 0.1),
        ]
        base += [transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)]
        base += [transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))]

    return transforms.Compose(base)


def _eval_tf(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


# ============================================================
# SEEDED DATA LOADER HELPERS
# ============================================================

def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(dataset, batch_size: int, shuffle: bool,
                 num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=g if shuffle else None,
        worker_init_fn=_seed_worker,
    )


# ============================================================
# DTD — DESCRIBABLE TEXTURES DATASET
# ============================================================

def load_dtd(data_path: str, img_size: int, batch_size: int,
             num_workers: int, split_seed: int = 42, train_seed: int = 0):
    """
    DTD has official train / val / test splits (partition=1 by default).
    47 classes × 120 images/class = 5640 total.
    Each split: ~1880 images.

    Returns: train_loader, val_loader, test_loader, class_names
    """
    train_ds = datasets.DTD(data_path, split="train", partition=1,
                            transform=_train_tf(img_size, "dtd"), download=True)
    val_ds   = datasets.DTD(data_path, split="val",   partition=1,
                            transform=_eval_tf(img_size), download=True)
    test_ds  = datasets.DTD(data_path, split="test",  partition=1,
                            transform=_eval_tf(img_size), download=True)

    class_names = train_ds.classes

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=train_seed)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=0)
    test_loader  = _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, seed=0)

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# STL-10
# ============================================================

def load_stl10(data_path: str, img_size: int, batch_size: int,
               num_workers: int, val_fraction: float = 0.1,
               split_seed: int = 42, train_seed: int = 0):
    """
    STL-10: 10 classes, 96×96 images.
    Official train (5000) and test (8000) splits.
    We carve 10% from train as validation.

    Returns: train_loader, val_loader, test_loader, class_names
    """
    # torchvision expects root = parent of stl10_binary/
    # on disk: data/stl10_binary/  → pass data/ as root
    stl_root = data_path if os.path.isdir(os.path.join(data_path, "stl10_binary")) \
               else os.path.dirname(data_path)

    train_ds_full = datasets.STL10(stl_root, split="train", download=False,
                                   transform=_train_tf(img_size, "stl10"))
    eval_ds_full  = datasets.STL10(stl_root, split="train", download=False,
                                   transform=_eval_tf(img_size))
    test_ds       = datasets.STL10(stl_root, split="test",  download=False,
                                   transform=_eval_tf(img_size))

    class_names = train_ds_full.classes
    n     = len(train_ds_full)
    n_val = int(val_fraction * n)

    idx      = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed)).tolist()
    train_ds = Subset(train_ds_full, idx[n_val:])
    val_ds   = Subset(eval_ds_full,  idx[:n_val])

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=train_seed)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=0)
    test_loader  = _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, seed=0)

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# CIFAR-100
# ============================================================

def load_cifar100(data_path: str, img_size: int, batch_size: int,
                  num_workers: int, val_fraction: float = 0.1,
                  split_seed: int = 42, train_seed: int = 0):
    """
    CIFAR-100: 100 classes, 32×32 native (resized to img_size).
    Official train (50K) and test (10K) splits.
    We carve val_fraction from train as validation.

    Complexity sits between STL-10 (10 classes) and Tiny ImageNet (200 classes),
    making it a useful intermediate point for the task-complexity axis.

    Returns: train_loader, val_loader, test_loader, class_names
    """
    train_ds_full = datasets.CIFAR100(data_path, train=True,  download=False,
                                      transform=_train_tf(img_size, "cifar100"))
    eval_ds_full  = datasets.CIFAR100(data_path, train=True,  download=False,
                                      transform=_eval_tf(img_size))
    test_ds       = datasets.CIFAR100(data_path, train=False, download=False,
                                      transform=_eval_tf(img_size))

    class_names = train_ds_full.classes
    n     = len(train_ds_full)          # 50,000
    n_val = int(val_fraction * n)       # 5,000

    idx      = torch.randperm(n, generator=torch.Generator().manual_seed(split_seed)).tolist()
    train_ds = Subset(train_ds_full, idx[n_val:])   # 45,000
    val_ds   = Subset(eval_ds_full,  idx[:n_val])   # 5,000

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=train_seed)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=0)
    test_loader  = _make_loader(test_ds,  batch_size, shuffle=False, num_workers=num_workers, seed=0)

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# TINY IMAGENET
# ============================================================

def load_tiny_imagenet(data_path: str, img_size: int, batch_size: int,
                       num_workers: int, split_seed: int = 42, train_seed: int = 0):
    """
    Tiny ImageNet — 200 classes, 100K images (64×64 native, resized to img_size).

    Expects this directory structure:
        data_path/
            train/   (200 class subdirectories, 500 images each)
            val/     (200 class subdirectories, 50 images each)

    NOTE: The official download has val/ in a flat layout (all images in one folder
    with a val_annotations.txt file). You must rearrange it first:

        python -c "
        import os, shutil
        val_dir = '/path/to/tiny-imagenet-200/val'
        with open(os.path.join(val_dir, 'val_annotations.txt')) as f:
            for line in f:
                fname, cls = line.split()[:2]
                os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
                shutil.move(os.path.join(val_dir, 'images', fname),
                            os.path.join(val_dir, cls, fname))
        "

    Returns: train_loader, val_loader, test_loader, class_names
    (test_loader = val_loader — Tiny ImageNet has no public test labels)
    """
    # Accept either:
    #   data_path = .../data                  → appends tiny-imagenet-200/
    #   data_path = .../data/tiny-imagenet-200 → uses directly
    if os.path.basename(data_path) == "tiny-imagenet-200":
        tiny_root = data_path
    else:
        tiny_root = os.path.join(data_path, "tiny-imagenet-200")

    train_path = os.path.join(tiny_root, "train")
    val_path   = os.path.join(tiny_root, "val")

    if not os.path.isdir(train_path):
        raise FileNotFoundError(
            f"Tiny ImageNet train/ not found at: {train_path}\n"
            "Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
            "Then rearrange val/ (see docstring above).")

    train_ds    = datasets.ImageFolder(train_path, transform=_train_tf(img_size, "tiny_imagenet"))
    val_ds      = datasets.ImageFolder(val_path,   transform=_eval_tf(img_size))
    class_names = train_ds.classes

    train_loader = _make_loader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, seed=train_seed)
    val_loader   = _make_loader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, seed=0)

    return train_loader, val_loader, val_loader, class_names   # test = val


# ============================================================
# UNIFIED LOADER FACTORY
# ============================================================

DATASET_INFO = {
    "dtd": {
        "default_img_size": 96,
        "description":      "DTD Textures — local texture, CNN dominates",
        "classes":          47,
        "train_size":       "~1880",
        "complexity":       "LOW  (texture / local features)",
    },
    "stl10": {
        "default_img_size": 96,
        "description":      "STL-10 — mixed object recognition",
        "classes":          10,
        "train_size":       "~4500 (after val split)",
        "complexity":       "MEDIUM (shape + texture mix)",
    },
    "cifar100": {
        "default_img_size": 96,
        "description":      "CIFAR-100 — 100-class object recognition",
        "classes":          100,
        "train_size":       "~45000 (after val split)",
        "complexity":       "MEDIUM-HIGH (100 fine-grained classes)",
    },
    "tiny_imagenet": {
        "default_img_size": 96,
        "description":      "Tiny ImageNet — high complexity, global reasoning",
        "classes":          200,
        "train_size":       "100K",
        "complexity":       "HIGH (diverse objects, global structure)",
    },
}


def get_loaders(dataset: str, data_path: str, img_size: int = None,
                batch_size: int = 128, num_workers: int = 4,
                split_seed: int = 42, train_seed: int = 0):
    """
    Unified loader factory.
    Returns: (train_loader, val_loader, test_loader, class_names)

    All datasets default to 96×96, giving identical sequence lengths:
        n_pool=4 → L=36   (6×6  tokens)
        n_pool=3 → L=144  (12×12 tokens)
        n_pool=2 → L=576  (24×24 tokens)
    """
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_INFO)}")

    if img_size is None:
        img_size = DATASET_INFO[dataset]["default_img_size"]

    loaders = {
        "dtd":           load_dtd,
        "stl10":         load_stl10,
        "cifar100":      load_cifar100,
        "tiny_imagenet": load_tiny_imagenet,
    }

    return loaders[dataset](data_path, img_size, batch_size, num_workers,
                            split_seed=split_seed, train_seed=train_seed)
