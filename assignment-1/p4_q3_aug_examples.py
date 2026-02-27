"""
Generate example augmented images for Problem 4, Question 3.
Saves a 2-row × 6-column figure (top: images, bottom: masks).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.insert(0, ".")
from p4_q3_augmentations import aug_geometric, aug_photometric, aug_gaussian_noise, aug_zoom, aug_combined

STRATEGIES = [
    ("No augmentation", lambda img, msk: (img.copy(), msk.copy())),
    ("Geometric",        aug_geometric),
    ("Photometric",      aug_photometric),
    ("Gaussian noise",   aug_gaussian_noise),
    ("Zoom",             aug_zoom),
]

# Use a fixed seed per augmentation for reproducibility
SEEDS = [0, 42, 42, 42, 42]

image = cv2.imread("Data/train/image/3.png", cv2.IMREAD_COLOR)
mask  = cv2.imread("Data/train/mask/3.png",  cv2.IMREAD_GRAYSCALE)

fig, axes = plt.subplots(2, len(STRATEGIES), figsize=(15, 6))

for col, ((name, fn), seed) in enumerate(zip(STRATEGIES, SEEDS)):
    np.random.seed(seed)
    aug_img, aug_msk = fn(image.copy(), mask.copy())

    axes[0, col].imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    axes[0, col].set_title(name, fontsize=9, fontweight="bold")
    axes[0, col].axis("off")

    axes[1, col].imshow(aug_msk, cmap="gray")
    axes[1, col].axis("off")

# Row labels
for row, label in enumerate(["Image", "Mask"]):
    axes[row, 0].set_ylabel(label, fontsize=10, labelpad=4)
    axes[row, 0].yaxis.set_visible(True)
    axes[row, 0].tick_params(left=False, labelleft=False)
    for spine in axes[row, 0].spines.values():
        spine.set_visible(False)

plt.tight_layout(pad=0.5)
plt.savefig("report/p4_q3_aug_examples.png", dpi=150, bbox_inches="tight")
print("Saved report/p4_q3_aug_examples.png")
