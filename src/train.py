"""
train.py
--------
Systematic grid search over EfficientNetB3 training hyperparameters.
Runs headlessly — no Jupyter kernel required.

Usage:
    python src/train.py
    python src/train.py --resume   # skip runs already in grid_search_results.xlsx

Outputs:
    models/checkpoints/            best weights per run
    results/logs/                  TensorBoard logs per run
    results/grid_search_results.xlsx  one row per run, saved after every run
"""

from __future__ import annotations

import gc
import sys
import argparse

from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd
import tensorflow as tf

# Allow running from project root or src/
sys.path.append(str(Path(__file__).resolve().parent))

from dataset import (
    build_dataset,
    get_class_weights,
    CLASSES,
    NUM_CLASSES,
)

from model import (
    build_model,
    compile_phase1,
    compile_phase2,
    unfreeze_top_layers,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / 'data' / 'processed'
RESULTS_DIR = ROOT_DIR / 'results'
CKPT_DIR    = ROOT_DIR / 'models' / 'checkpoints'
LOGS_DIR    = RESULTS_DIR / 'logs'

EXCEL_PATH  = RESULTS_DIR / 'grid_search_results.xlsx'

for d in [RESULTS_DIR, CKPT_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────
BATCH_SIZES          = [32, 64]
UNFREEZE_FROM_LAYERS = [150, 200, 250, 300]

PHASE1_LRS           = [1e-2, 1e-3, 1e-4]
PHASE1_EPOCHS_LIST   = [10, 15]

PHASE2_LRS           = [1e-5, 1e-6, 1e-7]
PHASE2_EPOCHS_LIST   = [40]

USE_CLASS_WEIGHTS    = True


# ─────────────────────────────────────────────────────────────
# TensorFlow 2.10 EfficientNet fix
# ─────────────────────────────────────────────────────────────
def fix_rescaling(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # EfficientNet block
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Rescaling):
                    scale = sublayer.scale

                    if isinstance(scale, tf.Tensor):
                        sublayer.scale = scale.numpy().tolist()
                        print(f"Fixed: {sublayer.name}")
    return model


# ─────────────────────────────────────────────────────────────
# Result utilities
# ─────────────────────────────────────────────────────────────
def build_result_row(
    run_id,
    batch_size,
    unfreeze_from,
    p1_lr,
    p1_epochs,
    p2_lr,
    p2_epochs,
    p1_history,
    p2_history,
    p1_epochs_run,
    p2_epochs_run,
    val_loss,
    val_acc,
    val_auc,
    timestamp,
):

    return {

        'run_id': run_id,
        'timestamp': timestamp,

        'batch_size': batch_size,
        'unfreeze_from_layer': unfreeze_from,

        'phase1_lr': p1_lr,
        'phase1_max_epochs': p1_epochs,

        'phase2_lr': p2_lr,
        'phase2_max_epochs': p2_epochs,

        'use_class_weights': USE_CLASS_WEIGHTS,

        # Phase 1
        'p1_epochs_run': p1_epochs_run,
        'p1_best_val_acc': max(p1_history['val_accuracy']),
        'p1_best_val_loss': min(p1_history['val_loss']),
        'p1_best_val_auc': max(p1_history['val_auc']),
        'p1_final_train_acc': p1_history['accuracy'][-1],

        # Phase 2
        'p2_epochs_run': p2_epochs_run,
        'p2_best_val_acc': max(p2_history['val_accuracy']),
        'p2_best_val_loss': min(p2_history['val_loss']),
        'p2_best_val_auc': max(p2_history['val_auc']),
        'p2_final_train_acc': p2_history['accuracy'][-1],

        # Final evaluation
        'final_val_loss': val_loss,
        'final_val_acc': val_acc,
        'final_val_auc': val_auc,

        # Overfitting
        'p2_overfit_gap':
            p2_history['accuracy'][-1] -
            max(p2_history['val_accuracy']),
    }


def save_results(all_results):

    df = pd.DataFrame(all_results)
    df.to_excel(EXCEL_PATH, index=False)


def load_existing_results():

    if EXCEL_PATH.exists():
        return pd.read_excel(EXCEL_PATH).to_dict('records')

    return []


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing Excel results',
    )

    args = parser.parse_args()

    print(f'TensorFlow : {tf.__version__}')
    print(f'GPUs       : {tf.config.list_physical_devices("GPU")}')

    # ---------------------------------------------------------
    # Load previous results
    # ---------------------------------------------------------
    all_results = []

    completed_configs = set()

    if args.resume and EXCEL_PATH.exists():

        all_results = load_existing_results()

        for row in all_results:

            config_key = (
                row['batch_size'],
                row['unfreeze_from_layer'],
                row['phase1_lr'],
                row['phase1_max_epochs'],
                row['phase2_lr'],
                row['phase2_max_epochs'],
            )

            completed_configs.add(config_key)

        print(f'\nResuming from {len(all_results)} completed runs')

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------
    class_weights = None

    if USE_CLASS_WEIGHTS:

        class_weights = get_class_weights(DATA_DIR / 'train')

        print('\nClass weights:')

        for i, cls in enumerate(CLASSES):
            print(f'  [{i}] {cls:<25s}: {class_weights[i]:.4f}')

    # ---------------------------------------------------------
    # Create full parameter grid
    # ---------------------------------------------------------
    parameter_grid = list(product(
        BATCH_SIZES,
        UNFREEZE_FROM_LAYERS,
        PHASE1_LRS,
        PHASE1_EPOCHS_LIST,
        PHASE2_LRS,
        PHASE2_EPOCHS_LIST,
    ))

    print(f'\nTotal runs: {len(parameter_grid)}')

    # ---------------------------------------------------------
    # Main training loop
    # ---------------------------------------------------------
    for run_id, params in enumerate(parameter_grid, start=1):

        (
            BATCH_SIZE,
            UNFREEZE_FROM_LAYER,
            PHASE1_LR,
            PHASE1_EPOCHS,
            PHASE2_LR,
            PHASE2_EPOCHS,
        ) = params

        config_key = (
            BATCH_SIZE,
            UNFREEZE_FROM_LAYER,
            PHASE1_LR,
            PHASE1_EPOCHS,
            PHASE2_LR,
            PHASE2_EPOCHS,
        )

        if args.resume and config_key in completed_configs:

            print(f'\nSkipping run {run_id:03d} (already completed)')
            continue

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f'\n{"="*72}')
        print(f'RUN {run_id:03d}')
        print(f'{"="*72}')

        print(f'Batch size           : {BATCH_SIZE}')
        print(f'Unfreeze from layer  : {UNFREEZE_FROM_LAYER}')
        print(f'Phase1 LR            : {PHASE1_LR}')
        print(f'Phase1 epochs        : {PHASE1_EPOCHS}')
        print(f'Phase2 LR            : {PHASE2_LR}')
        print(f'Phase2 epochs        : {PHASE2_EPOCHS}')

        # -----------------------------------------------------
        # Build datasets
        # -----------------------------------------------------
        train_ds = build_dataset(
            DATA_DIR / 'train',
            batch_size=BATCH_SIZE,
            augment=True,
            shuffle=True,
        )

        val_ds = build_dataset(
            DATA_DIR / 'val',
            batch_size=BATCH_SIZE,
            augment=False,
            shuffle=False,
        )

        # -----------------------------------------------------
        # Clear session and build fresh model
        # -----------------------------------------------------
        tf.keras.backend.clear_session()

        model = build_model(num_classes=NUM_CLASSES)
        model = fix_rescaling(model)

        # -----------------------------------------------------
        # Phase 1
        # -----------------------------------------------------
        print('\n--- Phase 1 ---')

        model = compile_phase1(
            model,
            learning_rate=PHASE1_LR,
        )

        p1_ckpt = CKPT_DIR / f'run{run_id:03d}_phase1.weights.h5'

        callbacks_p1 = [

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(p1_ckpt),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        history_p1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE1_EPOCHS,
            callbacks=callbacks_p1,
            class_weight=class_weights,
            verbose=1,
        )

        p1_epochs_run = len(history_p1.history['val_accuracy'])

        # -----------------------------------------------------
        # Phase 2
        # -----------------------------------------------------
        print('\n--- Phase 2 ---')

        model.load_weights(str(p1_ckpt))

        model = unfreeze_top_layers(
            model,
            from_layer=UNFREEZE_FROM_LAYER,
        )

        model = compile_phase2(
            model,
            learning_rate=PHASE2_LR,
        )

        p2_ckpt = CKPT_DIR / f'run{run_id:03d}_phase2.weights.h5'

        callbacks_p2 = [

            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(p2_ckpt),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),

            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1,
            ),
        ]

        history_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=PHASE2_EPOCHS,
            callbacks=callbacks_p2,
            class_weight=class_weights,
            verbose=1,
        )

        p2_epochs_run = len(history_p2.history['val_accuracy'])

        # -----------------------------------------------------
        # Final evaluation
        # -----------------------------------------------------
        model.load_weights(str(p2_ckpt))

        val_loss, val_acc, val_auc = model.evaluate(
            val_ds,
            verbose=0,
        )

        print(f'\nFinal results:')
        print(f'  val_acc  : {val_acc:.4f}')
        print(f'  val_auc  : {val_auc:.4f}')
        print(f'  val_loss : {val_loss:.4f}')

        # -----------------------------------------------------
        # Save result row
        # -----------------------------------------------------
        row = build_result_row(
            run_id,
            BATCH_SIZE,
            UNFREEZE_FROM_LAYER,
            PHASE1_LR,
            PHASE1_EPOCHS,
            PHASE2_LR,
            PHASE2_EPOCHS,
            history_p1.history,
            history_p2.history,
            p1_epochs_run,
            p2_epochs_run,
            val_loss,
            val_acc,
            val_auc,
            timestamp,
        )

        all_results.append(row)

        save_results(all_results)

        print(f'\nSaved results to:')
        print(EXCEL_PATH)

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------
        del model
        gc.collect()

    print(f'\n{"="*72}')
    print('GRID SEARCH COMPLETE')
    print(f'{"="*72}')


if __name__ == '__main__':
    main()
