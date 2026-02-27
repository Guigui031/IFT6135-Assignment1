"""
Problem 4, Question 3 — Plot augmentation strategy comparison.

Usage:
    python p4_q3_plot_results.py \
        --log_dirs logs/aug_none/results.json \
                   logs/aug_geometric/results.json \
                   logs/aug_photometric/results.json \
                   logs/aug_elastic/results.json \
                   logs/aug_combined/results.json
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ['No augmentation', 'Geometric', 'Photometric', 'Gaussian noise', 'Zoom', 'Combined']
COLORS = ['#333333', '#e41a1c', '#ff7f00', '#4daf4a', '#984ea3', '#377eb8']
STYLES = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,2))]


def load(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dirs", nargs="+", required=True)
    p.add_argument("--out", default="report/p4_q3_aug_curves.png")
    return p.parse_args()


def main():
    args  = parse_args()
    logs  = [load(p) for p in args.log_dirs]
    epochs = range(1, len(logs[0]["valid_accs"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ME = 4  # mark every 4 epochs

    for log, label, color in zip(logs, LABELS, COLORS):
        axes[0].plot(epochs, log["valid_losses"], "s-", color=color, markevery=ME, label=label)
        axes[1].plot(epochs, log["valid_accs"],   "s-", color=color, markevery=ME, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("DiceCE Loss (val)")
    axes[0].set_title("Validation Loss — Augmentation strategies")
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice Score (val)")
    axes[1].set_title("Validation Dice Score — Augmentation strategies")
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.4)

    print(f"\n{'Strategy':<20} {'Best val Dice':>14}")
    print("-" * 36)
    for log, label in zip(logs, LABELS):
        print(f"{label:<20} {max(log['valid_accs']):>14.4f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {args.out}")


if __name__ == "__main__":
    main()
