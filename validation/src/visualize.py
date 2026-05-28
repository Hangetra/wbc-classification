"""
visualize.py
------------
Visualisations for MLL23 external validation results.

Figures produced:
  1. confusion_matrix.png           normalised confusion matrix on MLL23
  2. per_class_f1.png               per-class F1 bar chart (MLL23 only)
  3. bodzas_vs_mll23_f1.png         grouped bar chart: Bodzas vs MLL23 F1
  4. f1_delta.png                   signed delta (MLL23 − Bodzas) per class
  5. confidence_by_class.png        model confidence distribution per class
  6. confidence_correct_vs_wrong.png confidence for correct vs incorrect predictions

Usage:
    python validation/src/visualize.py
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'src'))
from dataset import CLASSES                        # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIG_DIR            # noqa: E402

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams['figure.dpi'] = 130
SHORT_NAMES = {                                    # compact x-axis labels
    'Basophile'            : 'Baso',
    'Eosinophile'          : 'Eosi',
    'Lymphoblast'          : 'LyBl',
    'Lymphocyte'           : 'Lyco',
    'Monocyte'             : 'Mono',
    'Myeloblast'           : 'MyBl',
    'Neutrophile_Band'     : 'Band',
    'Neutrophile_Segment'  : 'Segm',
    'Normoblast'           : 'Norm',
}


# ── Loaders ────────────────────────────────────────────────────────────────────
def _load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f'Required file not found: {path}\n'
            f'Run evaluate.py first.'
        )
    return pd.read_csv(path, index_col=0 if path.name != 'classifications.csv' else None)


def load_results():
    df_clf     = _load_required(RESULTS_DIR / 'classifications.csv')
    df_metrics = _load_required(RESULTS_DIR / 'metrics.csv')
    cmp_path   = RESULTS_DIR / 'bodzas_vs_mll23.csv'
    df_cmp     = pd.read_csv(cmp_path, index_col=0) if cmp_path.exists() else None
    return df_clf, df_metrics, df_cmp


# ── Figure helpers ─────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


# ── Figure 1: Confusion matrix ─────────────────────────────────────────────────
def fig_confusion_matrix(df_clf: pd.DataFrame):
    y_true = df_clf['true_idx'].values
    y_pred = df_clf['predicted_idx'].values
    cm     = confusion_matrix(y_true, y_pred, normalize='true',
                              labels=list(range(len(CLASSES))))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.4, ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True',      fontsize=11)
    ax.set_title('Normalised Confusion Matrix — MLL23 External Validation', fontsize=12)
    plt.xticks(rotation=40, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(fig, 'confusion_matrix.png')


# ── Figure 2: Per-class F1 bar chart ──────────────────────────────────────────
def fig_per_class_f1(df_metrics: pd.DataFrame):
    per_cls = df_metrics.loc[
        [c for c in CLASSES if c in df_metrics.index],
        ['precision', 'recall', 'f1-score']
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(per_cls))
    width = 0.28

    ax.bar(x - width,     per_cls['precision'],  width, label='Precision', color='#2980B9', alpha=0.85)
    ax.bar(x,             per_cls['recall'],      width, label='Recall',    color='#27AE60', alpha=0.85)
    ax.bar(x + width,     per_cls['f1-score'],    width, label='F1',        color='#8E44AD', alpha=0.85)

    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='0.9 threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(c, c) for c in per_cls.index], rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.08)
    ax.set_title('Per-class Metrics — MLL23 External Validation', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    _save(fig, 'per_class_f1.png')


# ── Figure 3: Bodzas vs MLL23 F1 grouped bar chart ────────────────────────────
def fig_bodzas_vs_mll23(df_cmp: pd.DataFrame | None):
    if df_cmp is None or 'bodzas_f1' not in df_cmp.columns:
        print('  [SKIP] bodzas_vs_mll23_f1.png — Bodzas results not available')
        return

    classes_present = [c for c in CLASSES if c in df_cmp.index]
    bodzas_f1 = df_cmp.loc[classes_present, 'bodzas_f1'].values
    mll23_f1  = df_cmp.loc[classes_present, 'mll23_f1'].values

    x     = np.arange(len(classes_present))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, bodzas_f1, width, label='Bodzas (test)', color='#2980B9', alpha=0.85)
    ax.bar(x + width / 2, mll23_f1,  width, label='MLL23 (external)', color='#C0392B', alpha=0.85)

    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SHORT_NAMES.get(c, c) for c in classes_present],
        rotation=30, ha='right'
    )
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.08)
    ax.set_title('F1 Score: Bodzas Test Set vs MLL23 External Validation', fontsize=12)
    ax.legend()
    plt.tight_layout()
    _save(fig, 'bodzas_vs_mll23_f1.png')


# ── Figure 4: F1 delta (MLL23 − Bodzas) ───────────────────────────────────────
def fig_f1_delta(df_cmp: pd.DataFrame | None):
    if df_cmp is None or 'f1_delta' not in df_cmp.columns:
        print('  [SKIP] f1_delta.png — Bodzas results not available')
        return

    classes_present = [c for c in CLASSES if c in df_cmp.index]
    deltas          = df_cmp.loc[classes_present, 'f1_delta'].values
    colors          = ['#27AE60' if d >= 0 else '#C0392B' for d in deltas]

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(
        [SHORT_NAMES.get(c, c) for c in classes_present],
        deltas, color=colors, alpha=0.85, width=0.6,
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axhline(-0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
               label='±0.05 band')
    ax.axhline(+0.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_ylabel('Δ F1  (MLL23 − Bodzas)')
    ax.set_title('F1 Change vs Bodzas Test Set — Positive = Improved on MLL23', fontsize=12)
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    _save(fig, 'f1_delta.png')


# ── Figure 5: Confidence by class ─────────────────────────────────────────────
def fig_confidence_by_class(df_clf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    order = CLASSES

    sns.boxplot(
        data=df_clf,
        x='true_class', y='confidence',
        order=[c for c in order if c in df_clf['true_class'].unique()],
        palette='muted', ax=ax,
    )
    ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
               label='0.9 confidence threshold')
    ax.set_xticklabels(
        [SHORT_NAMES.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
        rotation=30, ha='right',
    )
    ax.set_xlabel('')
    ax.set_ylabel('Prediction confidence')
    ax.set_title('Model Confidence per Class — MLL23 External Validation', fontsize=12)
    ax.legend()
    plt.tight_layout()
    _save(fig, 'confidence_by_class.png')


# ── Figure 6: Confidence — correct vs incorrect ────────────────────────────────
def fig_confidence_correct_vs_wrong(df_clf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))

    for label, color, ls in [
        ('Correct',   '#27AE60', '-'),
        ('Incorrect', '#C0392B', '--'),
    ]:
        subset = df_clf[df_clf['correct'] == (label == 'Correct')]['confidence']
        if len(subset) == 0:
            continue
        subset.plot.kde(ax=ax, label=f'{label} (n={len(subset)})',
                        color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel('Prediction confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=12)
    ax.legend()
    ax.set_xlim([0, 1])
    plt.tight_layout()
    _save(fig, 'confidence_correct_vs_wrong.png')


# ── Main ───────────────────────────────────────────────────────────────────────
def run_visualisation():
    print('Loading results...')
    df_clf, df_metrics, df_cmp = load_results()

    print(f'\nGenerating figures ({FIG_DIR}):')
    fig_confusion_matrix(df_clf)
    fig_per_class_f1(df_metrics)
    fig_bodzas_vs_mll23(df_cmp)
    fig_f1_delta(df_cmp)
    fig_confidence_by_class(df_clf)
    fig_confidence_correct_vs_wrong(df_clf)

    print(f'\nDone. All figures saved to {FIG_DIR}')


if __name__ == '__main__':
    run_visualisation()
