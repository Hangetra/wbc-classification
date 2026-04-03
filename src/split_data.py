"""
split_data.py
-------------
Stratified 60/20/20 split of raw Bodzas WBC images into
data/processed/{train, val, test}/<ClassName>/ subfolders.

Usage:
    python src/split_data.py
    python src/split_data.py --raw_dir data/raw/bodzas_wbc --out_dir data/processed --seed 42
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES = [
    "Basophile",
    "Eosinophile",
    "Lymphoblast",
    "Lymphocyte",
    "Monocyte",
    "Myeloblast",
    "Neutrophile_Band",
    "Neutrophile_Segment",
    "Normoblast",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
# TEST_RATIO  = 0.20  (remainder)


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified WBC dataset splitter")
    parser.add_argument("--raw_dir",  default="../data/raw/bodzas_wbc",       help="Root dir with one subfolder per class")
    parser.add_argument("--out_dir",  default="../data/processed", help="Output root dir")
    parser.add_argument("--seed",     default=42, type=int,     help="Random seed for reproducibility")
    parser.add_argument("--overwrite", action="store_true",     help="Delete existing processed dir and redo split")
    return parser.parse_args()


def collect_images(class_dir: Path) -> list[Path]:
    """Return sorted list of image paths in a directory."""
    return sorted([
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def split_indices(n: int, train_r: float, val_r: float, seed: int):
    """Return (train_idx, val_idx, test_idx) for n samples."""
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def copy_files(files: list[Path], indices: list[int], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        shutil.copy2(files[i], dest_dir / files[i].name)


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Safety checks
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    if out_dir.exists():
        if args.overwrite:
            print(f"[INFO] Removing existing processed dir: {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                "Use --overwrite to redo the split."
            )

    random.seed(args.seed)

    split_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    missing_classes = []

    print(f"\n{'='*55}")
    print(f"  WBC Stratified Split  (seed={args.seed})")
    print(f"  Ratios: train={TRAIN_RATIO}  val={VAL_RATIO}  test={1-TRAIN_RATIO-VAL_RATIO:.2f}")
    print(f"{'='*55}")

    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            missing_classes.append(cls)
            print(f"  [WARN] Class folder not found: {cls_dir}")
            continue

        images = collect_images(cls_dir)
        n = len(images)
        if n == 0:
            print(f"  [WARN] No images found in: {cls_dir}")
            continue

        train_idx, val_idx, test_idx = split_indices(n, TRAIN_RATIO, VAL_RATIO, args.seed)

        for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            dest = out_dir / split / cls
            copy_files(images, idx, dest)
            split_counts[split][cls] = len(idx)

        print(f"  {cls:<25}  total={n:>4}  "
              f"train={len(train_idx):>3}  val={len(val_idx):>3}  test={len(test_idx):>3}")

    # Summary table
    print(f"\n{'='*55}")
    print("  Split totals:")
    for split in ("train", "val", "test"):
        total = sum(split_counts[split].values())
        print(f"    {split:<6}: {total} images")
    print(f"{'='*55}\n")

    if missing_classes:
        print(f"[WARN] Missing class folders: {missing_classes}")
        print("       Add the raw images and re-run with --overwrite\n")

    print("[DONE] Split complete.")
    print(f"       Processed data written to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
