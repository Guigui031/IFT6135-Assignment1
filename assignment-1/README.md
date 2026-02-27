# IFT6135 Assignment 1 — Practical

Code for IFT6135 Assignment 1, covering MLP/MobileNet classification (Problem 2) and UNet segmentation (Problem 4).

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Problem 2: Classification (PathMNIST)](#problem-2-classification-pathmnist)
- [Problem 4: Segmentation (Retina Blood Vessels)](#problem-4-segmentation-retina-blood-vessels)
- [File Descriptions](#file-descriptions)
- [Results Summary](#results-summary)

---

## Environment Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU support (CUDA 12.6):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

---

## Dataset Preparation

### PathMNIST (Problem 2)
Downloaded automatically on first run.

### Retina Blood Vessel Data (Problem 4)
Download from the course link and place as:
```
Data/
├── train/
│   ├── image/   # 80 retinal fundus images
│   └── mask/    # 80 binary vessel masks
└── test/
    ├── image/   # 20 images
    └── mask/    # 20 masks
```

---

## Problem 2: Classification (PathMNIST)

### Q6: MLP vs MobileNet

**Train:**
```bash
python main_classification.py --model mlp       --logdir logs/mlp       --epochs 15 --batch_size 128 --lr 0.001 --weight_decay 0.0005
python main_classification.py --model mobilenet --logdir logs/mobilenet  --epochs 15 --batch_size 128 --lr 0.001 --weight_decay 0.0005
```

**Plot:**
```bash
python p2_q6_plot_results.py \
    --mlp_log       logs/mlp/results.json \
    --mobilenet_log logs/mobilenet/results.json \
    --out report/training_curves.png
```

**Results:** MLP ~69.7% val accuracy, MobileNet ~93.6% val accuracy.

---

## Problem 4: Segmentation (Retina Blood Vessels)

All segmentation experiments use `smp.Unet` with a ResNet-18 encoder from `segmentation_models_pytorch`. Default optimizer is **AdamW** unless otherwise noted.

### Q1: Skip Connections

Train UNet with and without skip connections:
```bash
python main_segmentation.py --dataset Data --model smp_unet       --logdir logs/p4_q1_smp_unet       --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet_noskip --logdir logs/p4_q1_smp_unet_noskip --epochs 40 --batch_size 2 --lr 1e-4
```

Plot:
```bash
python p4_q1_plot_results.py \
    --unet_log   logs/p4_q1_smp_unet/results.json \
    --noskip_log logs/p4_q1_smp_unet_noskip/results.json \
    --out report/p4_q1_curves.png
```

**Results:** UNet with skip: **0.8103** Dice, no skip: **0.6106** Dice (~20 pp gap).

---

### Q2: Learning Rate Sweep (Adam optimizer)

```bash
python main_segmentation.py --dataset Data --model smp_unet --optimizer adam --lr 0.1     --logdir logs/p4_lr_0.1     --epochs 40 --batch_size 2
python main_segmentation.py --dataset Data --model smp_unet --optimizer adam --lr 0.01    --logdir logs/p4_lr_0.01    --epochs 40 --batch_size 2
python main_segmentation.py --dataset Data --model smp_unet --optimizer adam --lr 0.001   --logdir logs/p4_lr_0.001   --epochs 40 --batch_size 2
python main_segmentation.py --dataset Data --model smp_unet --optimizer adam --lr 0.0001  --logdir logs/p4_lr_0.0001  --epochs 40 --batch_size 2
python main_segmentation.py --dataset Data --model smp_unet --optimizer adam --lr 0.00001 --logdir logs/p4_lr_0.00001 --epochs 40 --batch_size 2
```

Plot:
```bash
python p4_q2_plot_results.py \
    --log_dirs logs/p4_lr_0.1/results.json \
               logs/p4_lr_0.01/results.json \
               logs/p4_lr_0.001/results.json \
               logs/p4_lr_0.0001/results.json \
               logs/p4_lr_0.00001/results.json \
    --lrs 0.1 0.01 0.001 0.0001 0.00001 \
    --out report/p4_q2_lr_curves.png
```

**Results:**

| LR | Best val Dice |
|---|---|
| 0.1 | 0.2946 (diverges) |
| 0.01 | 0.8098 |
| **0.001** | **0.8383** |
| 0.0001 | 0.8099 |
| 0.00001 | 0.3824 (too slow) |

---

### Q3: Data Augmentation Strategies

Four strategies are implemented in `p4_q3_augmentations.py`:
- `geometric` — random flip + rotation ±30°
- `photometric` — brightness & contrast jitter
- `gaussian_noise` — additive Gaussian noise (σ ∈ [5, 25])
- `zoom` — random crop + resize (scale ∈ [0.7, 1.0])
- `combined` — all four in sequence

```bash
python main_segmentation.py --dataset Data --model smp_unet --augmentation none          --logdir logs/p4_q3_none        --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet --augmentation geometric     --logdir logs/p4_q3_geometric   --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet --augmentation photometric   --logdir logs/p4_q3_photometric --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet --augmentation gaussian_noise --logdir logs/p4_q3_gaussian    --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet --augmentation zoom          --logdir logs/p4_q3_zoom        --epochs 40 --batch_size 2 --lr 1e-4
python main_segmentation.py --dataset Data --model smp_unet --augmentation combined      --logdir logs/p4_q3_combined    --epochs 40 --batch_size 2 --lr 1e-4
```

Generate augmentation example images:
```bash
python p4_q3_aug_examples.py
# saves report/p4_q3_aug_examples.png
```

Plot training curves:
```bash
python p4_q3_plot_results.py \
    --log_dirs logs/p4_q3_none/results.json \
               logs/p4_q3_geometric/results.json \
               logs/p4_q3_photometric/results.json \
               logs/p4_q3_gaussian/results.json \
               logs/p4_q3_zoom/results.json \
    --out report/p4_q3_aug_curves.png
```

**Results:**

| Strategy | Best val Dice |
|---|---|
| None (baseline) | 0.8057 |
| **Geometric** | **0.8141** |
| Photometric | 0.8117 |
| Gaussian noise | 0.7958 |
| Zoom | 0.7787 |

---

### Q4: Pretrained Model Fine-tuning

```bash
python main_segmentation.py --dataset Data --model pretrained_unet --logdir logs/p4_q4_pretrained_unet --epochs 40 --batch_size 2 --lr 1e-4
```

Plot (uses Q1 scratch run as baseline):
```bash
python p4_q4_plot_results.py \
    --scratch_log    logs/p4_q1_smp_unet/results.json \
    --pretrained_log logs/p4_q4_pretrained_unet/results.json \
    --out report/p4_q4_curves.png
```

**Results:** From scratch: 0.8103, ImageNet fine-tuned: **0.8296** (+1.9 pp).

---

## File Descriptions

### Training Scripts
| File | Description |
|---|---|
| `main_classification.py` | Train MLP / MobileNet on PathMNIST |
| `main_segmentation.py` | Train UNet variants on retina vessel data |

### Model Implementations
| File | Description |
|---|---|
| `mlp.py` | MLP with Glorot init |
| `mobileNet.py` | MobileNet with depthwise separable convolutions |
| `unet.py` | Custom UNet (used for Gradescope autograder) |
| `p4_q1_unet_no_skip.py` | Custom UNet without skip connections |

> `main_segmentation.py` also defines `SmpUNetNoSkip` (zeroes out encoder skip features for a fair no-skip baseline with the smp architecture).

### Augmentations
| File | Description |
|---|---|
| `p4_q3_augmentations.py` | `geometric`, `photometric`, `gaussian_noise`, `zoom`, `combined` |
| `p4_q3_aug_examples.py` | Generates `report/p4_q3_aug_examples.png` |

### Plotting Scripts
| File | Output |
|---|---|
| `p2_q6_plot_results.py` | `report/training_curves.png` |
| `p4_q1_plot_results.py` | `report/p4_q1_curves.png` |
| `p4_q2_plot_results.py` | `report/p4_q2_lr_curves.png` |
| `p4_q3_plot_results.py` | `report/p4_q3_aug_curves.png` |
| `p4_q4_plot_results.py` | `report/p4_q4_curves.png` |

### Utilities
| File | Description |
|---|---|
| `utils.py` | `DiceLoss`, `DiceCELoss` |

---

## Results Summary

| Problem | Model | Best val metric |
|---|---|---|
| P2 Q6 | MLP | 69.7% accuracy |
| P2 Q6 | MobileNet | 93.6% accuracy |
| P4 Q1 | UNet (with skip) | 0.8103 Dice |
| P4 Q1 | UNet (no skip) | 0.6106 Dice |
| P4 Q2 | Best LR (0.001, Adam) | 0.8383 Dice |
| P4 Q3 | Best augmentation (geometric) | 0.8141 Dice |
| P4 Q4 | From scratch | 0.8103 Dice |
| P4 Q4 | ImageNet pretrained | 0.8296 Dice |

---

## Notes

- **Reproducibility**: Seeds fixed (`np.random.seed(42)`, `torch.manual_seed(42)`).
- **Q4 no code submission**: Problem 4 requires only the PDF report; `segmentation_models_pytorch` is used for reliable results.
- **Windows**: `num_workers=0` is hardcoded in DataLoaders to avoid multiprocessing issues.
- **GPU**: ~3–4 s/epoch for segmentation experiments on a modern GPU (40 epochs ≈ 2–3 min per run).
