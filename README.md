# WBC Classification — EfficientNetB3 Transfer Learning

Multi-class classification of 9 white blood cell types using transfer learning on the Bodzas et al. dataset.

## Classes
| # | Class |
|---|-------|
| 0 | Basophile |
| 1 | Eosinophile |
| 2 | Lymphoblast |
| 3 | Lymphocyte |
| 4 | Monocyte |
| 5 | Myeloblast |
| 6 | Neutrophile_Band |
| 7 | Neutrophile_Segment |
| 8 | Normoblast |

## Architecture
- **Base model**: EfficientNetB3 pretrained on ImageNet
- **Input size**: 300×300×3
- **Training strategy**: 2-phase fine-tuning (frozen base → partial unfreeze)
- **Split**: 60% train / 20% validation / 20% test (stratified per class)

## Project Structure
```
wbc_classifier/
├── data/
│   ├── raw/                    # Original Bodzas images, one subfolder per class
│   │   ├── Basophile/
│   │   ├── Eosinophile/
│   │   ├── Lymphoblast/
│   │   ├── Lymphocyte/
│   │   ├── Monocyte/
│   │   ├── Myeloblast/
│   │   ├── Neutrophile_Band/
│   │   ├── Neutrophile_Segment/
│   │   └── Normoblast/
│   └── processed/
│       ├── train/              # Stratified 60% split
│       ├── val/                # Stratified 20% split
│       └── test/               # Stratified 20% split (held out — touch last!)
├── EDA/                        # Exploratory data analysis outputs
│   ├── class_distribution.png
│   ├── sample_grid.png
│   └── pixel_stats.csv
├── models/
│   ├── checkpoints/            # Saved .keras weights during training
│   └── final/                  # Final saved model
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.csv
│   └── gradcam/                # Grad-CAM visualisations per class
├── notebooks/
│   ├── 02_EDA.ipynb
│   ├── 03_train.ipynb
│   └── 04_evaluate.ipynb
├── src/
│   ├── split_data.py           # Run this first
│   ├── dataset.py              # tf.data pipeline helpers
│   ├── model.py                # EfficientNetB3 builder
│   └── utils.py                # Metrics, Grad-CAM, plotting
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw images in data/raw/<ClassName>/

# 3. Split dataset
python src/split_data.py

# 4. Open notebooks in order
jupyter lab
```

## Requirements
See `requirements.txt`. Primary dependencies:
- TensorFlow ≥ 2.13
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (metrics + stratified split)
- opencv-python (Grad-CAM)
