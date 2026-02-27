"""
Problem 2, Question 2 — Blurring kernel
Applies a Gaussian kernel via discrete_2d_convolution and saves the figure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from utils import discrete_2d_convolution


# ─── Helper: build a normalized Gaussian kernel ───────────────────────────────
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a (size x size) Gaussian kernel with std = sigma, sum = 1."""
    k = size // 2
    xs = np.arange(-k, k + 1, dtype=np.float64)
    x, y = np.meshgrid(xs, xs)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


# ─── Load a grayscale test image ──────────────────────────────────────────────
image_bgr = cv2.imread("1995-fiat-multipla-minivan.jpg")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

# ─── Build the kernel ─────────────────────────────────────────────────────────
KERNEL_SIZE = 15   # 15 x 15 window
SIGMA       = 3.0  # standard deviation in pixels

kernel = gaussian_kernel(KERNEL_SIZE, SIGMA)

print(f"Kernel size : {KERNEL_SIZE}x{KERNEL_SIZE}")
print(f"Sigma       : {SIGMA}")
print(f"Kernel sum  : {kernel.sum():.6f}  (should be 1.0)")
print()
print("Kernel (rounded to 4 dp):")
print(np.round(kernel, 4))

# ─── Apply the kernel ─────────────────────────────────────────────────────────
blurred = discrete_2d_convolution(image, kernel)

# Clip to valid display range
blurred = np.clip(blurred, 0, 255)

# ─── Save the side-by-side figure ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original image", fontsize=13)
axes[0].axis("off")

axes[1].imshow(blurred, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(
    f"After Gaussian blur  (size={KERNEL_SIZE}×{KERNEL_SIZE}, σ={SIGMA})",
    fontsize=13,
)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("report/blur_result.png", dpi=150, bbox_inches="tight")
print("\nFigure saved -> report/blur_result.png")

# ─── Save kernel visualisation ────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(4, 3.5))
im = ax.imshow(kernel, cmap="hot", interpolation="nearest")
ax.set_title(f"Gaussian kernel  (size={KERNEL_SIZE}, σ={SIGMA})", fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("report/gaussian_kernel.png", dpi=150, bbox_inches="tight")
print("Figure saved -> report/gaussian_kernel.png")
