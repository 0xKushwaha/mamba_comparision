"""
download_data.py — Download datasets for the SSM contribution study.

Supports all auto-downloadable datasets. Tiny ImageNet instructions are
printed but must be downloaded manually (cluster has no internet access).

Usage:
    # Download everything (recommended)
    python download_data.py --data_path ./data

    # Download specific datasets only
    python download_data.py --data_path ./data --datasets cifar100 stl10 dtd

    # Verify existing downloads without re-downloading
    python download_data.py --data_path ./data --verify_only

After downloading, run the ablation:
    python run_ablation.py --dataset cifar100 --data_path ./data --epochs 100
"""

import os
import sys
import shutil
import zipfile
import argparse
import urllib.request
from torchvision import datasets

# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================

def _progress_bar(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {pct:.1f}%  ({downloaded/1e6:.0f}/{total_size/1e6:.0f} MB)",
              end="", flush=True)


def download_cifar100(data_path: str, verify_only: bool = False) -> bool:
    """
    CIFAR-100 — 100 classes, 50K train + 10K test, 32×32 images (~170 MB).
    Complexity: MEDIUM-HIGH. Acts as bridge between STL-10 and Tiny ImageNet.
    """
    print("\n─── CIFAR-100  (~170 MB) ───────────────────────────────")
    marker = os.path.join(data_path, "cifar-100-python", "train")

    if verify_only:
        ok = os.path.exists(marker)
        print(f"  Status : {'✓ found' if ok else '✗ NOT found — run without --verify_only'}")
        return ok

    if os.path.exists(marker):
        print("  ✓ Already downloaded — skipping")
        return True

    print(f"  Destination : {data_path}")
    datasets.CIFAR100(data_path, train=True,  download=True)
    datasets.CIFAR100(data_path, train=False, download=True)

    train_ds = datasets.CIFAR100(data_path, train=True,  download=False)
    test_ds  = datasets.CIFAR100(data_path, train=False, download=False)
    print(f"  Train : {len(train_ds):,} images  ✓")
    print(f"  Test  : {len(test_ds):,} images   ✓")
    print(f"  Classes : 100")
    return True


def download_dtd(data_path: str, verify_only: bool = False) -> bool:
    """
    DTD — Describable Textures, 47 classes, 5640 images (~600 MB).
    Complexity: LOW (pure texture, CNN dominates).
    """
    print("\n─── DTD  (~600 MB) ─────────────────────────────────────")
    marker = os.path.join(data_path, "dtd", "images")

    if verify_only:
        ok = os.path.isdir(marker)
        print(f"  Status : {'✓ found' if ok else '✗ NOT found — run without --verify_only'}")
        return ok

    if os.path.isdir(marker):
        print("  ✓ Already downloaded — skipping")
        return True

    print(f"  Destination : {data_path}")
    for split in ("train", "val", "test"):
        ds = datasets.DTD(data_path, split=split, partition=1, download=True)
        print(f"  {split.capitalize():5s} : {len(ds):,} images  ✓")
    print(f"  Classes : 47")
    return True


def download_stl10(data_path: str, verify_only: bool = False) -> bool:
    """
    STL-10 — 10 classes, 5K train + 8K test, 96×96 native images (~2.6 GB).
    Complexity: MEDIUM (shape + texture mix).
    """
    print("\n─── STL-10  (~2.6 GB) ──────────────────────────────────")
    marker = os.path.join(data_path, "stl10_binary")

    if verify_only:
        ok = os.path.isdir(marker)
        print(f"  Status : {'✓ found' if ok else '✗ NOT found — run without --verify_only'}")
        return ok

    if os.path.isdir(marker):
        print("  ✓ Already downloaded — skipping")
        return True

    print(f"  Destination : {data_path}")
    train_ds = datasets.STL10(data_path, split="train", download=True)
    test_ds  = datasets.STL10(data_path, split="test",  download=True)
    print(f"  Train : {len(train_ds):,} images  ✓")
    print(f"  Test  : {len(test_ds):,} images   ✓")
    print(f"  Classes : 10")
    return True


def download_tiny_imagenet(data_path: str, verify_only: bool = False) -> bool:
    """
    Tiny ImageNet — 200 classes, 100K images, 64×64 native (~236 MB).
    Complexity: HIGH (diverse objects, global spatial reasoning).
    Downloads automatically if internet is available; also handles val/ rearrangement.
    """
    print("\n─── Tiny ImageNet  (~236 MB) ────────────────────────────")
    tiny_root = os.path.join(data_path, "tiny-imagenet-200")
    marker    = os.path.join(tiny_root, "train")

    if verify_only:
        ok = os.path.isdir(marker)
        print(f"  Status : {'✓ found' if ok else '✗ NOT found — run without --verify_only'}")
        return ok

    if os.path.isdir(marker):
        print("  ✓ Already downloaded — skipping")
        return True

    URL      = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_path, "tiny-imagenet-200.zip")

    # Download
    if not os.path.exists(zip_path):
        print(f"  Downloading from {URL}")
        try:
            urllib.request.urlretrieve(URL, zip_path, reporthook=_progress_bar)
            print()  # newline after progress bar
        except Exception as e:
            print(f"\n  ✗ Download failed: {e}")
            print("  → Download manually and place tiny-imagenet-200/ in your data_path")
            return False
    else:
        print("  ✓ Zip already present — skipping download")

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)
    os.remove(zip_path)
    print("  ✓ Extracted")

    # Rearrange val/ from flat layout to ImageFolder layout
    # Before: val/images/val_XXXXX.JPEG  +  val/val_annotations.txt
    # After : val/<class_id>/val_XXXXX.JPEG
    val_dir  = os.path.join(tiny_root, "val")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(ann_file):
        print("  Rearranging val/ into class subdirectories...")
        moved = 0
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split()
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(val_dir, "images", fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.exists(src):
                    shutil.move(src, dst)
                    moved += 1
        images_dir = os.path.join(val_dir, "images")
        if os.path.isdir(images_dir) and not os.listdir(images_dir):
            os.rmdir(images_dir)
        print(f"  ✓ Moved {moved:,} val images into class folders")
    else:
        print("  ⚠  val_annotations.txt not found — val/ may already be arranged")

    print(f"  Classes : 200")
    return True


# ============================================================
# DISPATCH TABLE
# ============================================================

DOWNLOAD_FNS = {
    "cifar100":      download_cifar100,
    # "dtd":           download_dtd,
    # "stl10":         download_stl10,
    # "tiny_imagenet": download_tiny_imagenet,
}

ALL_DATASETS = list(DOWNLOAD_FNS.keys())


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Download datasets for the SSM contribution ablation study",
    )
    _default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    p.add_argument("--data_path",   default=_default_data,
                   help=f"Root directory to store datasets (default: ./data/)")
    p.add_argument("--datasets",    nargs="+", default=ALL_DATASETS,
                   choices=ALL_DATASETS,
                   help=f"Datasets to download (default: all — {ALL_DATASETS})")
    p.add_argument("--verify_only", action="store_true",
                   help="Only verify existing downloads; do not download anything")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.data_path, exist_ok=True)

    print("=" * 60)
    print("  SSM Contribution Study — Dataset Downloader")
    print(f"  Root    : {os.path.abspath(args.data_path)}")
    print(f"  Mode    : {'verify only' if args.verify_only else 'download'}")
    print(f"  Targets : {args.datasets}")
    print("=" * 60)

    failed = []
    for ds in args.datasets:
        try:
            ok = DOWNLOAD_FNS[ds](args.data_path, verify_only=args.verify_only)
            if not ok:
                failed.append(ds)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(ds)

    print("\n" + "=" * 60)
    if failed:
        print(f"  FAILED  : {failed}")
        sys.exit(1)
    else:
        print(f"  {'Verified' if args.verify_only else 'Downloaded'} : {args.datasets}  ✓")
        print(f"\n  Ready to run ablation:")
        for ds in args.datasets:
            print(f"    python run_ablation.py --dataset {ds} "
                  f"--data_path {os.path.abspath(args.data_path)} --epochs 100")
    print("=" * 60)


if __name__ == "__main__":
    main()
