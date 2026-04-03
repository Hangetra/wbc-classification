"""
dataset.py
----------
tf.data pipeline for WBC classification.
Handles loading, preprocessing, and augmentation for EfficientNetB3 (300x300).
"""

from __future__ import annotations

import tensorflow as tf
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 300          # EfficientNetB3 native input size
NUM_CLASSES = 9
AUTOTUNE    = tf.data.AUTOTUNE

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
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Resize to 300x300 and apply EfficientNet preprocessing.
    EfficientNetB3 expects pixel values in [0, 255] — the internal
    rescaling is handled inside the model (include_preprocessing=True).
    We only need to ensure dtype is float32.
    """
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)
    return image


# ── Augmentation (train only) ─────────────────────────────────────────────────
def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Standard augmentation for microscopy WBC images.
    - Random horizontal/vertical flips (cells have no canonical orientation)
    - Random rotation via random 90° steps + small continuous rotation
    - Random zoom (simulate different magnification)
    - Random brightness/contrast (stain variation)
    - Random hue/saturation shift (stain batch effects)
    """
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random 90° rotation (k in {0,1,2,3})
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # Colour jitter — important for stain normalisation robustness
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Clip after colour ops to stay in valid range
    image = tf.clip_by_value(image, 0.0, 255.0)

    # Random zoom via crop-and-resize (5–15% zoom range)
    zoom_factor = tf.random.uniform([], 0.85, 1.0)
    crop_size   = tf.cast(IMG_SIZE * zoom_factor, tf.int32)
    image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    return image


# ── Dataset builder ────────────────────────────────────────────────────────────
def load_image_label(filepath: tf.Tensor, label: tf.Tensor):
    """Read, decode, and preprocess a single image."""
    raw  = tf.io.read_file(filepath)
    img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img  = tf.cast(img, tf.float32)
    img  = preprocess_image(img)
    return img, label


def build_dataset(
    data_dir: str | Path,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    cache: bool = True,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from a directory with class subfolders.

    Args:
        data_dir:   Path to split dir, e.g. data/processed/train
        batch_size: Samples per batch
        augment:    Apply augmentation (True for train split only)
        shuffle:    Shuffle the dataset
        cache:      Cache preprocessed images in memory (set False if RAM limited)

    Returns:
        Batched, prefetched tf.data.Dataset yielding (image, one_hot_label) tuples
    """
    data_dir = Path(data_dir)

    filepaths, labels = [], []
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] Class directory missing: {cls_dir}")
            continue
        idx = CLASS_TO_IDX[cls]
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"):
            for img_path in cls_dir.glob(ext):
                filepaths.append(str(img_path))
                labels.append(idx)

    if not filepaths:
        raise ValueError(f"No images found under {data_dir}")

    print(f"[dataset] {data_dir.name}: {len(filepaths)} images across {len(set(labels))} classes")

    # One-hot encode
    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels_oh))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)

    ds = ds.map(load_image_label, num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()

    if augment:
        ds = ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def get_class_weights(data_dir: str | Path) -> dict:
    """
    Compute class weights inversely proportional to class frequency.
    Pass the result to model.fit(class_weight=...) to handle imbalance.
    """
    data_dir = Path(data_dir)
    counts = {}
    for cls in CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            counts[CLASS_TO_IDX[cls]] = 0
            continue
        n = sum(1 for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
                for _ in cls_dir.glob(ext))
        counts[CLASS_TO_IDX[cls]] = n

    total = sum(counts.values())
    n_cls = len(CLASSES)
    weights = {
        idx: total / (n_cls * cnt) if cnt > 0 else 1.0
        for idx, cnt in counts.items()
    }
    return weights
