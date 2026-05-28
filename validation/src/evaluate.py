"""
evaluate.py
-----------
Run inference on preprocessed MLL23 images and compute per-class metrics.
Ground truth comes directly from folder names (MLL23 is fully annotated).

Since preprocessed folders use Bodzas class naming (set by preprocess.py),
the evaluation logic is dataset-agnostic — it works identically to the
core 04_evaluate.ipynb pipeline.

Outputs (all in results/):
    classifications.csv         one row per image: true class, predicted class,
                                confidence, per-class probabilities
    metrics.csv                 per-class precision / recall / F1 / support
    bodzas_vs_mll23.csv         side-by-side F1 comparison if Bodzas report found

Usage:
    python validation/src/evaluate.py
    python validation/src/evaluate.py --batch_size 64
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
from dataset import CLASSES, NUM_CLASSES          # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from config import (                               # noqa: E402
    PREP_DIR, RESULTS_DIR, MODEL_PATH,
    BODZAS_REPORT_PATH, IMG_EXTENSIONS, TARGET_SIZE,
)


# ── Image loader ───────────────────────────────────────────────────────────────
def load_image(path: Path) -> np.ndarray:
    """Load a preprocessed PNG as float32 (H, W, 3) ready for the model."""
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [TARGET_SIZE, TARGET_SIZE])
    return img.numpy()


# ── Data collection ────────────────────────────────────────────────────────────
def collect_images() -> list[tuple[str, Path]]:
    """
    Collect all preprocessed images. Folder names are Bodzas class names
    (set by preprocess.py), so true label = folder name directly.

    Returns:
        List of (bodzas_class_name, image_path) sorted by class then filename.
    """
    items = []
    missing = []

    for cls in CLASSES:
        cls_dir = PREP_DIR / cls
        if not cls_dir.exists():
            missing.append(cls)
            continue
        class_images = sorted([
            p for ext in IMG_EXTENSIONS for p in cls_dir.glob(f'*{ext}')
        ])
        items.extend((cls, p) for p in class_images)

    if missing:
        print(f'[WARN] Classes not found in preprocessed data: {missing}')
        print(f'       Run preprocess.py first, or check that ZIPs were downloaded.')

    return items


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(
    model: tf.keras.Model,
    images: list[tuple[str, Path]],
    batch_size: int,
) -> pd.DataFrame:
    """
    Run batched inference and return a DataFrame with one row per image.
    """
    rows = []

    for start in tqdm(range(0, len(images), batch_size), desc='Inferring'):
        batch    = images[start : start + batch_size]
        imgs_arr = np.stack([load_image(p) for _, p in batch])   # (B, 300, 300, 3)
        probs    = model.predict(imgs_arr, verbose=0)             # (B, 9)
        preds    = np.argmax(probs, axis=1)
        confs    = np.max(probs, axis=1)

        for (true_cls, img_path), pred_idx, conf, prob_row in zip(
            batch, preds, confs, probs
        ):
            row = {
                'image_filename' : img_path.name,
                'true_class'     : true_cls,
                'true_idx'       : CLASSES.index(true_cls),
                'predicted_class': CLASSES[pred_idx],
                'predicted_idx'  : int(pred_idx),
                'confidence'     : float(conf),
                'correct'        : true_cls == CLASSES[pred_idx],
            }
            for cls_name, p in zip(CLASSES, prob_row):
                row[f'prob_{cls_name}'] = float(p)
            rows.append(row)

    return pd.DataFrame(rows)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class and aggregate metrics from classification results."""
    y_true = df['true_idx'].values
    y_pred = df['predicted_idx'].values

    report = classification_report(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        digits=4,
        output_dict=True,
    )
    return pd.DataFrame(report).T


def compare_with_bodzas(df_mll23: pd.DataFrame) -> pd.DataFrame | None:
    """
    Load Bodzas test set report and produce a side-by-side F1 comparison.
    Returns None if the Bodzas report is not available.
    """
    if not BODZAS_REPORT_PATH.exists():
        return None

    df_bodzas = pd.read_csv(BODZAS_REPORT_PATH, index_col=0)

    cols_mll23   = ['precision', 'recall', 'f1-score', 'support']
    cols_bodzas  = ['precision', 'recall', 'f1-score', 'support']

    df_m = df_mll23.loc[
        [c for c in CLASSES if c in df_mll23.index], cols_mll23
    ].copy()
    df_m.columns = ['mll23_precision', 'mll23_recall', 'mll23_f1', 'mll23_support']

    df_b = df_bodzas.loc[
        [c for c in CLASSES if c in df_bodzas.index], cols_bodzas
    ].copy()
    df_b.columns = ['bodzas_precision', 'bodzas_recall', 'bodzas_f1', 'bodzas_support']

    df_cmp = df_b.join(df_m, how='outer')
    df_cmp['f1_delta'] = df_cmp['mll23_f1'] - df_cmp['bodzas_f1']
    return df_cmp


# ── Main ───────────────────────────────────────────────────────────────────────
def run_evaluation(batch_size: int = 32):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not PREP_DIR.exists() or not any(PREP_DIR.iterdir()):
        raise FileNotFoundError(
            f'Preprocessed data not found: {PREP_DIR}\n'
            f'Run first: python extension/external_validation/src/preprocess.py'
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Trained model not found: {MODEL_PATH}\n'
            f'Complete the core WBC training pipeline first.'
        )

    # Load model
    print(f'Loading model: {MODEL_PATH}')
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f'Input shape: {model.input_shape}')

    # Collect images
    images = collect_images()
    if not images:
        raise ValueError('No preprocessed images found. Run preprocess.py first.')

    counts = {cls: sum(1 for c, _ in images if c == cls) for cls in CLASSES}
    print(f'\nImages per class (MLL23):')
    for cls, n in counts.items():
        marker = ' ← low sample count' if 0 < n < 100 else ''
        print(f'  {cls:<25}: {n:>6}{marker}')
    print(f'  {"TOTAL":<25}: {len(images):>6}')

    # Inference
    print(f'\nRunning inference (batch_size={batch_size}):')
    df_clf = run_inference(model, images, batch_size)

    # Save classifications
    clf_path = RESULTS_DIR / 'classifications.csv'
    df_clf.to_csv(clf_path, index=False)

    # Metrics
    df_metrics = compute_metrics(df_clf)
    metrics_path = RESULTS_DIR / 'metrics.csv'
    df_metrics.to_csv(metrics_path)

    # Print summary
    overall_acc = df_clf['correct'].mean()
    print(f'\nMLL23 external validation results:')
    print(f'  Overall accuracy: {overall_acc:.4f}')
    print()
    print(classification_report(
        df_clf['true_idx'].values,
        df_clf['predicted_idx'].values,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASSES,
        digits=4,
    ))

    # Side-by-side comparison with Bodzas
    df_cmp = compare_with_bodzas(df_metrics)
    cmp_path = RESULTS_DIR / 'bodzas_vs_mll23.csv'

    if df_cmp is not None:
        df_cmp.to_csv(cmp_path)
        print('Bodzas vs MLL23 F1 comparison:')
        print(f'  {"Class":<25} {"Bodzas F1":>10} {"MLL23 F1":>10} {"Δ F1":>8}')
        print('  ' + '-' * 56)
        for cls in CLASSES:
            if cls in df_cmp.index:
                r      = df_cmp.loc[cls]
                b_f1   = r.get('bodzas_f1', float('nan'))
                m_f1   = r.get('mll23_f1',  float('nan'))
                delta  = r.get('f1_delta',  float('nan'))
                trend  = ' ▼' if delta < -0.05 else (' ▲' if delta > 0.05 else '  ')
                print(f'  {cls:<25} {b_f1:>10.4f} {m_f1:>10.4f} {delta:>+8.4f}{trend}')
        print(f'\n  Saved → {cmp_path}')
    else:
        df_metrics.to_csv(cmp_path)
        print(f'[INFO] Bodzas report not found at {BODZAS_REPORT_PATH}')
        print(f'       Run 04_evaluate.ipynb first to enable side-by-side comparison.')

    print(f'\nAll results saved to {RESULTS_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate WBC model on MLL23 external validation set'
    )
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Inference batch size (default: 32)')
    args = parser.parse_args()
    run_evaluation(batch_size=args.batch_size)
