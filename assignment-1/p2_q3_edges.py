"""
Problem 2, Question 3 — Horizontal and vertical edge detection
Uses Sobel kernels via discrete_2d_convolution.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from utils import discrete_2d_convolution

# ─── Load grayscale image ─────────────────────────────────────────────────────
image_bgr = cv2.imread("1995-fiat-multipla-minivan.jpg")
image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

# ─── Sobel kernels ────────────────────────────────────────────────────────────
# Horizontal edges: responds to intensity changes along the vertical axis (dy).
# Rows are weighted finite-difference operators; columns apply Gaussian smoothing.
sobel_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
], dtype=np.float64)

# Vertical edges: responds to intensity changes along the horizontal axis (dx).
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float64)

print("Sobel horizontal kernel:")
print(sobel_horizontal)
print("\nSobel vertical kernel:")
print(sobel_vertical)

# ─── Apply kernels ────────────────────────────────────────────────────────────
edges_h = discrete_2d_convolution(image, sobel_horizontal)
edges_v = discrete_2d_convolution(image, sobel_vertical)

# Take absolute value: edges appear on both gradient signs
edges_h_abs = np.abs(edges_h)
edges_v_abs = np.abs(edges_v)

# ─── Save 3-panel figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original image", fontsize=12)
axes[0].axis("off")

axes[1].imshow(edges_h_abs, cmap="gray")
axes[1].set_title("Horizontal edges  (Sobel $G_y$)", fontsize=12)
axes[1].axis("off")

axes[2].imshow(edges_v_abs, cmap="gray")
axes[2].set_title("Vertical edges  (Sobel $G_x$)", fontsize=12)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("report/edge_detection_result.png", dpi=150, bbox_inches="tight")
print("\nFigure saved -> report/edge_detection_result.png")

# ─── Save kernel visualisations ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(7, 3))

for ax, kernel, title in zip(
    axes2,
    [sobel_horizontal, sobel_vertical],
    ["Sobel $G_y$ (horizontal edges)", "Sobel $G_x$ (vertical edges)"],
):
    im = ax.imshow(kernel, cmap="RdBu_r", interpolation="nearest",
                   vmin=-2, vmax=2)
    ax.set_title(title, fontsize=10)
    for (r, c), val in np.ndenumerate(kernel):
        ax.text(c, r, f"{int(val):+d}", ha="center", va="center",
                fontsize=13, fontweight="bold")
    ax.axis("off")

plt.colorbar(im, ax=axes2.ravel().tolist(), shrink=0.8)
plt.tight_layout()
plt.savefig("report/sobel_kernels.png", dpi=150, bbox_inches="tight")
print("Figure saved -> report/sobel_kernels.png")
