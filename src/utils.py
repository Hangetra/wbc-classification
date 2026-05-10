"""
utils.py
--------
Shared utilities: metrics reporting, Grad-CAM, confusion matrix plotting.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

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


# ── Metrics ────────────────────────────────────────────────────────────────────
def evaluate_model(model, dataset, save_dir: str | Path = "results"):
    """
    Run full evaluation on a dataset split.
    Saves confusion matrix PNG and classification report CSV.

    Args:
        model:    Trained Keras model
        dataset:  tf.data.Dataset (batched, NOT shuffled for evaluation)
        save_dir: Where to save outputs

    Returns:
        y_true, y_pred arrays
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y_true_all, y_pred_all = [], []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true_all.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_all.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    import pandas as pd
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4, output_dict=True
    )
    pd.DataFrame(report_dict).T.to_csv(save_dir / "classification_report.csv")
    print(f"[utils] Report saved → {save_dir / 'classification_report.csv'}")

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path=save_dir / "confusion_matrix.png")

    return y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, save_path: str | Path = None):
    """Plot and optionally save a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalised Confusion Matrix — WBC Classifier", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Confusion matrix saved → {save_path}")
    plt.show()


def plot_training_history(history, save_path: str | Path = None):
    """Plot accuracy and loss curves for both phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        train_key, val_key = metric
        if train_key in history:
            ax.plot(history[train_key],  label="Train",      linewidth=2)
        if val_key in history:
            ax.plot(history[val_key],    label="Validation", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[utils] Training curves saved → {save_path}")
    plt.show()


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a single preprocessed image.

    Args:
        img_array:          Shape (1, 300, 300, 3), float32 in [0,255]
        model:              Full Keras model

    Returns:
        heatmap: 2D numpy array, values in [0,1]
    """
    # Build a sub-model that outputs feature maps + predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("efficientnetb3").output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Args:
        original_img: (H, W, 3) uint8 image
        heatmap:      2D float array in [0,1]
        alpha:        Heatmap opacity
        colormap:     OpenCV colormap

    Returns:
        Blended image as uint8 numpy array
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def save_gradcam_grid(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    save_dir: str | Path = "results/gradcam",
    n_per_class: int = 3
):
    """
    Generate and save Grad-CAM visualisations for n_per_class samples per class.
    Iterates through the dataset until each class has enough samples.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class_samples = {i: [] for i in range(len(CLASSES))}

    for images, labels in dataset:
        for img, lbl in zip(images.numpy(), labels.numpy()):
            cls_idx = int(np.argmax(lbl))
            if len(class_samples[cls_idx]) < n_per_class:
                class_samples[cls_idx].append(img)
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    for cls_idx, samples in class_samples.items():
        cls_name = CLASSES[cls_idx]
        fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))

        for col, img_arr in enumerate(samples):
            # Original
            axes[0, col].imshow(img_arr.astype(np.uint8))
            axes[0, col].set_title(f"{cls_name}", fontsize=9)
            axes[0, col].axis("off")

            # Grad-CAM
            img_batch = np.expand_dims(img_arr, axis=0)
            heatmap   = make_gradcam_heatmap(img_batch, model)
            cam_img   = overlay_gradcam(img_arr.astype(np.uint8), heatmap)

            pred      = model.predict(img_batch, verbose=0)
            pred_cls  = CLASSES[np.argmax(pred)]
            conf      = np.max(pred)

            axes[1, col].imshow(cam_img)
            axes[1, col].set_title(f"Pred: {pred_cls}\n{conf:.2%}", fontsize=9)
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=10)
        axes[1, 0].set_ylabel("Grad-CAM",  fontsize=10)
        plt.suptitle(f"Grad-CAM — {cls_name}", fontsize=12)
        plt.tight_layout()

        out_path = save_dir / f"gradcam_{cls_name}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[utils] Grad-CAM saved → {out_path}")
