"""
model.py
--------
EfficientNetB3 transfer learning model for 9-class WBC classification.

Two-phase training strategy:
  Phase 1 — Frozen base: train classification head only (fast convergence)
  Phase 2 — Partial unfreeze: fine-tune top N blocks with low LR (precision)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
#from tensorflow.keras.applications.efficientnet import preprocess_input

NUM_CLASSES  = 9
IMG_SIZE     = 300
DROPOUT_RATE = 0.4
L2_REG       = 1e-4

# EfficientNetB3 has 385 layers. We unfreeze the last ~80 (top 2 blocks)
# during Phase 2. Adjust UNFREEZE_FROM if you want more/less fine-tuning.
UNFREEZE_FROM_LAYER = 300


def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build EfficientNetB3 transfer learning model.

    EfficientNetB3 in TF 2.10 has built-in rescaling baked into the
    architecture (no separate include_preprocessing param needed).
    Feed raw [0, 255] float32 pixels — no manual normalisation required.
    Top: GlobalAvgPool → BatchNorm → Dropout → Dense(256, relu) → Dropout → Softmax

    Args:
        num_classes: Number of output classes (default 9)

    Returns:
        Uncompiled Keras model with frozen base (Phase 1 ready)
    """
    # ── Base model ────────────────────────────────────────────────────────────
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        #include_preprocessing=False,
        # Note: include_preprocessing param was added in TF 2.12+
        # TF 2.10 EfficientNet handles normalisation internally — feed [0,255] directly
    )
    base.trainable = False            # frozen for Phase 1

    # ── Classification head ───────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    #x = preprocess_input(inputs)   
    x = base(inputs, training=False)  # training=False keeps BN in inference mode

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
        name="dense_256",
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="EfficientNetB3_WBC")
    return model


def compile_phase1(model: tf.keras.Model, learning_rate: float = 1e-3):
    """
    Compile for Phase 1: frozen base, higher LR, train head only.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),   # multi_label param added in TF 2.12+
        ],
    )
    _print_trainable_summary(model)
    return model


def unfreeze_top_layers(model: tf.keras.Model, from_layer: int = UNFREEZE_FROM_LAYER):
    """
    Unfreeze layers from `from_layer` onwards in the base model for Phase 2.
    BatchNormalization layers are kept frozen (inference mode) to preserve
    ImageNet statistics and prevent instability with small batches.
    """
    base = model.layers[1]    # EfficientNetB3 is the second layer
    base.trainable = True

    for layer in base.layers[:from_layer]:
        layer.trainable = False

    # Always keep BN layers frozen — critical for stable fine-tuning
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    _print_trainable_summary(model)
    return model


def compile_phase2(model: tf.keras.Model, learning_rate: float = 1e-5):
    """
    Re-compile for Phase 2: partially unfrozen base, very low LR.
    Use SGD with momentum for more stable fine-tuning of deep layers.
    """
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def _print_trainable_summary(model: tf.keras.Model):
    total   = sum(tf.size(w).numpy() for w in model.weights)
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.1f}%)")
    print(f"  Frozen params    : {total - trainable:,}")
