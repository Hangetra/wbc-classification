"""
preprocess.py
-------------
Preprocessing pipeline for MLL23 external validation images.

Steps per image:
  1. Load TIF as RGB
  2. Macenko stain normalisation  — mitigates Pappenheim vs Giemsa colour shift
  3. Resize 288×288 → 300×300     — INTER_AREA (slight downsample, minimal loss)
  4. Save as PNG into preprocessed/<BodzasClassName>/

Output folder structure mirrors the Bodzas layout (Bodzas class names used)
so that evaluate.py can use the same image-loading logic regardless of dataset.

Usage:
    python validation/src/preprocess.py
    python validation/src/preprocess.py --skip_normalisation
"""

from __future__ import annotations

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add core src to path — reuse Macenko implementation, do not duplicate it
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
from preprocessing import macenko_normalise   # noqa: E402  (AML extension module)

# Local config
sys.path.append(str(Path(__file__).resolve().parent))
from config import (                           # noqa: E402
    RAW_DIR, PREP_DIR, TARGET_SIZE,
    MLL23_TO_BODZAS, MLL23_EXPECTED_COUNTS, IMG_EXTENSIONS,
)

# ---------------------------------------------------------------------------
# NOTE: macenko_normalise is imported from extension/src/preprocessing.py
# which already lives in the project. This avoids duplicating ~80 lines of
# OD-space stain normalisation code. If the AML extension is not present,
# copy the function here and remove the import above.
# ---------------------------------------------------------------------------


def preprocess_image(path: Path, skip_normalisation: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for a single MLL23 image.

    Args:
        path:               Path to raw TIF file
        skip_normalisation: Skip Macenko step (for ablation comparison)

    Returns:
        Preprocessed (TARGET_SIZE × TARGET_SIZE × 3) uint8 RGB array
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f'Cannot read image: {path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if not skip_normalisation:
        try:
            img_rgb = macenko_normalise(img_rgb)
        except Exception as exc:
            # Fall back to original rather than crashing the whole pipeline
            print(f'    [WARN] Stain normalisation failed ({path.name}): {exc}')

    # 288 → 300: slight downsample — INTER_AREA is best for this direction
    img_out = cv2.resize(
        img_rgb,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return img_out


def verify_downloads() -> bool:
    """
    Check that all 9 class folders are present and report image counts
    vs expected counts from the published paper.
    Returns True if all classes are present, False otherwise.
    """
    print('Verifying MLL23 downloads:')
    print(f'  {"MLL23 folder":<25} {"Found":>7} {"Expected":>9} {"Status"}')
    print('  ' + '-' * 55)

    all_present = True
    for mll23_cls, bodzas_cls in MLL23_TO_BODZAS.items():
        cls_dir  = RAW_DIR / mll23_cls
        found    = sum(
            1 for ext in IMG_EXTENSIONS
            for _ in cls_dir.glob(f'*{ext}')
        ) if cls_dir.exists() else 0
        expected = MLL23_EXPECTED_COUNTS.get(mll23_cls, '?')

        if not cls_dir.exists():
            status = '✗ MISSING'
            all_present = False
        elif found == 0:
            status = '✗ EMPTY'
            all_present = False
        else:
            status = '✓'

        print(f'  {mll23_cls:<25} {found:>7} {str(expected):>9}   {status}')

    if not all_present:
        print()
        print('  Download missing ZIPs from: https://zenodo.org/uploads/14277609')
        print('  Unzip into: validation/data/raw/<class_name>/')
    return all_present


def run_preprocessing(skip_normalisation: bool = False):
    """
    Preprocess all images from all 9 matched MLL23 classes.
    Output is saved into PREP_DIR/<BodzasClassName>/ preserving
    the same folder structure as the Bodzas processed dataset.
    Already-processed images are skipped — safe to re-run.
    """
    print('=' * 60)
    print('  MLL23 preprocessing')
    print('=' * 60)

    verify_downloads()
    print()

    total_processed = 0
    total_skipped   = 0
    total_failed    = 0

    for mll23_cls, bodzas_cls in MLL23_TO_BODZAS.items():
        raw_cls_dir  = RAW_DIR  / mll23_cls
        prep_cls_dir = PREP_DIR / bodzas_cls
        prep_cls_dir.mkdir(parents=True, exist_ok=True)

        if not raw_cls_dir.exists():
            print(f'  Skipping {mll23_cls} — folder not found')
            continue

        images = [
            p for ext in IMG_EXTENSIONS
            for p in raw_cls_dir.glob(f'*{ext}')
        ]

        if not images:
            print(f'  Skipping {mll23_cls} — no images found')
            continue

        n_proc = n_skip = n_fail = 0

        for img_path in tqdm(images, desc=f'  {bodzas_cls:<25}', leave=True):
            out_path = prep_cls_dir / (img_path.stem + '.png')

            if out_path.exists():
                n_skip += 1
                continue

            try:
                img = preprocess_image(img_path, skip_normalisation)
                # Save as PNG (lossless), converting back to BGR for OpenCV
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                n_proc += 1
            except Exception as exc:
                print(f'\n    [ERROR] {img_path.name}: {exc}')
                n_fail += 1

        total_processed += n_proc
        total_skipped   += n_skip
        total_failed    += n_fail

    print()
    print('=' * 60)
    print(f'  Preprocessing complete')
    print(f'  Processed : {total_processed}')
    print(f'  Skipped   : {total_skipped} (already existed)')
    print(f'  Failed    : {total_failed}')
    print(f'  Output    : {PREP_DIR}')
    print(f'  Stain norm: {"DISABLED (ablation)" if skip_normalisation else "ENABLED (Macenko)"}')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess MLL23 images for external validation'
    )
    parser.add_argument(
        '--skip_normalisation',
        action='store_true',
        help='Skip Macenko stain normalisation (useful for ablation study)',
    )
    args = parser.parse_args()
    run_preprocessing(skip_normalisation=args.skip_normalisation)
