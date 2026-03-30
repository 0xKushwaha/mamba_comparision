"""
download_data.py — Download all datasets on your LOCAL machine, then transfer to cluster.

RUN THIS ON YOUR LOCAL MACHINE (not the cluster — cluster has no internet):
    python download_data.py

Downloads:
  DTD           ~600 MB   (torchvision)
  STL-10        ~2.6 GB   (torchvision)
  Tiny ImageNet ~236 MB   (urllib + auto val-rearrange)

After finishing, prints the exact rsync command to transfer to your cluster.
"""

import os
import shutil
import zipfile
import urllib.request
from torchvision import datasets

# Always save next to this script  →  mamba_cnn/data/
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print(f"\n  ⚠  Run this on your LOCAL machine, not the cluster.")
print(f"     Saving to: {DATA_DIR}\n")


# # ── DTD ──────────────────────────────────────────────────────
# print("─── DTD (Describable Textures Dataset) ─── ~600 MB")
# for split in ["train", "val", "test"]:
#     datasets.DTD(root=DATA_DIR, split=split, partition=1, download=True)
#     print(f"  ✓ {split}")
# print()


# # ── STL-10 ───────────────────────────────────────────────────
# print("─── STL-10 ─── ~2.6 GB")
# for split in ["train", "test"]:
#     datasets.STL10(root=DATA_DIR, split=split, download=True)
#     print(f"  ✓ {split}")
# print()


# ── Tiny ImageNet ─────────────────────────────────────────────
TINY_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
print("─── Tiny ImageNet ─── ~236 MB")

if os.path.isdir(os.path.join(TINY_DIR, "train")):
    print("  ✓ Already downloaded")

else:
    ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
    URL      = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Download with progress
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {pct:.1f}%  ({downloaded/1e6:.0f}/{total_size/1e6:.0f} MB)",
                  end="", flush=True)

    if not os.path.exists(ZIP_PATH):
        print(f"  Downloading from {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=_progress)
        print()  # newline after progress bar
    else:
        print("  ✓ Zip already present, skipping download")

    # Extract
    print("  Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("  ✓ Extracted")

    # Rearrange val/ from flat layout into class subdirectories
    # Original: val/images/val_XXXXX.JPEG  +  val/val_annotations.txt
    # Required: val/<class_id>/val_XXXXX.JPEG
    val_dir = os.path.join(TINY_DIR, "val")
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
        # Clean up empty images/ folder
        images_dir = os.path.join(val_dir, "images")
        if os.path.isdir(images_dir) and not os.listdir(images_dir):
            os.rmdir(images_dir)
        print(f"  ✓ Moved {moved} val images into class folders")
    else:
        print("  ⚠  val_annotations.txt not found — val/ may already be arranged")

print()


# ── Transfer to cluster ───────────────────────────────────────
CLUSTER_USER = "ravipr49_soe"
CLUSTER_HOST = "paramrudra.iuac.res.in"
CLUSTER_PORT = "4422"
CLUSTER_DIR  = f"/scratch/{CLUSTER_USER}/classification/data"

print("=" * 60)
print("  All datasets ready. Transfer to cluster with:\n")
print(f"  scp -r -P {CLUSTER_PORT} {DATA_DIR} \\")
print(f"    {CLUSTER_USER}@{CLUSTER_HOST}:{CLUSTER_DIR}")
print()
print("  Or rsync (resumes on disconnect — better for large files):")
print(f"  rsync -avz --progress -e 'ssh -p {CLUSTER_PORT}' \\")
print(f"    {DATA_DIR}/ \\")
print(f"    {CLUSTER_USER}@{CLUSTER_HOST}:{CLUSTER_DIR}/")
print("=" * 60)
