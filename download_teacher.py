#!/usr/bin/env python3
"""
Pre-download ViT-B/16 ImageNet weights into the timm/torch hub cache.
Run once before distillation training so Phase 1 doesn't re-download mid-run.

Usage: python download_teacher.py
"""

import timm

TEACHER_MODEL = "vit_base_patch16_224"

print(f"Downloading {TEACHER_MODEL} into cache...")
model = timm.create_model(TEACHER_MODEL, pretrained=True)
total_p = sum(p.numel() for p in model.parameters())
print(f"Done. {TEACHER_MODEL} cached ({total_p/1e6:.1f}M params).")
print("The distillation script will load it from cache automatically.")
