"""
Benchmark: discrete_2d_convolution vs scipy.signal.convolve2d
Problem 2, Question 1 — IFT6135 Assignment 1
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from utils import discrete_2d_convolution

# ─── Kernels ──────────────────────────────────────────────────────────────────
KERNELS = {
    "3x3": np.ones((3, 3), dtype=np.float64) / 9,
    "7x7": np.ones((7, 7), dtype=np.float64) / 49,
    "15x15": np.ones((15, 15), dtype=np.float64) / 225,
}

# ─── Image sizes to sweep ─────────────────────────────────────────────────────
IMAGE_SIZES = [64, 128, 256, 512]

N_REPEATS = 5  # number of timed repetitions per configuration


def time_function(fn, *args, repeats=N_REPEATS):
    """Return mean elapsed time (seconds) over `repeats` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def scipy_convolve2d_same(image, kernel):
    """Wrapper: scipy with mode='same' to match our implementation's output size."""
    # scipy.signal.convolve2d flips the kernel (true convolution).
    # For a fair apples-to-apples speed comparison we still use convolve2d;
    # the flip is O(1) and negligible.
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)


# ─── Run benchmarks ───────────────────────────────────────────────────────────
print(f"{'Image':>10}  {'Kernel':>6}  {'Custom (s)':>12}  {'SciPy (s)':>12}  {'Speedup':>8}")
print("-" * 60)

results = []  # (image_size, kernel_label, t_custom, t_scipy)

for size in IMAGE_SIZES:
    image = np.random.rand(size, size).astype(np.float64)
    for k_label, kernel in KERNELS.items():
        t_custom, _ = time_function(discrete_2d_convolution, image, kernel)
        t_scipy, _ = time_function(scipy_convolve2d_same, image, kernel)
        speedup = t_custom / t_scipy
        results.append((size, k_label, t_custom, t_scipy))
        print(
            f"{size:>10}  {k_label:>6}  {t_custom:>12.4f}  {t_scipy:>12.4f}  {speedup:>7.1f}x"
        )

# ─── Plot: execution time vs image size for each kernel ───────────────────────
fig, axes = plt.subplots(1, len(KERNELS), figsize=(14, 4), sharey=False)

for ax, (k_label, _) in zip(axes, KERNELS.items()):
    sizes_k = [r[0] for r in results if r[1] == k_label]
    t_custom_k = [r[2] for r in results if r[1] == k_label]
    t_scipy_k = [r[3] for r in results if r[1] == k_label]

    ax.plot(sizes_k, t_custom_k, "o-", label="discrete_2d_convolution", color="steelblue")
    ax.plot(sizes_k, t_scipy_k, "s-", label="scipy.signal.convolve2d", color="darkorange")
    ax.set_title(f"Kernel {k_label}")
    ax.set_xlabel("Image size (px)")
    ax.set_ylabel("Time (s)")
    ax.legend(fontsize=8)
    ax.set_xticks(sizes_k)
    ax.grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Convolution speed: custom vs SciPy", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("report/convolution_benchmark.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> report/convolution_benchmark.png")

# ─── Summary table (for the report) ──────────────────────────────────────────
print("\n=== LaTeX table snippet ===")
print(r"\begin{tabular}{cccc}")
print(r"  Image & Kernel & Custom (s) & SciPy (s) & Speedup \\")
print(r"  \hline")
for size, k_label, t_c, t_s in results:
    print(
        f"  {size}\\times{size} & {k_label} & {t_c:.4f} & {t_s:.4f} & {t_c/t_s:.1f}x \\\\"
    )
print(r"\end{tabular}")
