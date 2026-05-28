# External Validation — MLL23 Dataset

Tests generalisation of the trained WBC EfficientNetB3 classifier to MLL23,
an independent expert-annotated peripheral blood dataset acquired at the
Munich Leukemia Laboratory.

## Dataset
MLL23 (Shetab Boushehri, Kazeminia, Gruber et al., Sci Data 2025)
DOI: 10.1038/s41597-025-06223-x
Zenodo: https://zenodo.org/uploads/14277609

- 41,906 images · 288×288 px · 18 classes · Pappenheim staining
- Distributed as 18 ZIP files, one per class (e.g. basophil.zip)
- **Only download the 9 ZIPs matching Bodzas classes** (see table below)

## Classes to download

| Download this ZIP          | Folder name after unzip        | Bodzas equivalent     |
|----------------------------|--------------------------------|-----------------------|
| basophil.zip               | basophil                       | Basophile             |
| eosinophil.zip             | eosinophil                     | Eosinophile           |
| lymphoblast.zip            | lymphoblast                    | Lymphoblast           |
| lymphocyte.zip             | lymphocyte                     | Lymphocyte            |
| monocyte.zip               | monocyte                       | Monocyte              |
| myeloblast.zip             | myeloblast                     | Myeloblast            |
| band_neutrophil.zip        | band_neutrophil                | Neutrophile_Band      |
| segmented_neutrophil.zip   | segmented_neutrophil           | Neutrophile_Segment   |
| normoblast.zip             | normoblast                     | Normoblast            |

Unzip each into `data/raw/` so the structure is:
```
data/raw/
├── basophil/
├── eosinophil/
├── lymphoblast/
├── lymphocyte/
├── monocyte/
├── myeloblast/
├── band_neutrophil/
├── segmented_neutrophil/
└── normoblast/
```

## Domain shift context
- Resolution: 288×288 → 300×300 (4% resize, negligible information loss)
- Staining: Pappenheim (MLL23) vs Giemsa (Bodzas) — primary shift source
- Acquisition: automated scanner (MLL23) vs manual microscopy (Bodzas)
- Mitigation: Macenko stain normalisation applied in preprocessing

## Folder structure
```
external_validation/
├── data/
│   ├── raw/                    ← unzip MLL23 class ZIPs here
│   └── preprocessed/           ← output of preprocess.py
├── src/
│   ├── config.py               ← class name mapping, paths, constants
│   ├── preprocess.py           ← stain normalisation + resize
│   ├── evaluate.py             ← inference + per-class metrics
│   └── visualize.py            ← comparison figures
├── results/
│   ├── figures/
│   ├── classifications.csv
│   ├── metrics.csv
│   └── bodzas_vs_mll23.csv
├── notebooks/
│   └── external_validation.ipynb
└── README.md
```

## Workflow
```bash
# 1. Download and unzip the 9 relevant ZIPs into data/raw/

# 2. Preprocess (stain normalise + resize)
python validation/src/preprocess.py

# 3. Evaluate (inference + metrics)
python validation/src/evaluate.py

# 4. Visualise
python validation/src/visualize.py

# 5. Interactive analysis
jupyter lab validation/notebooks/external_validation.ipynb
```
