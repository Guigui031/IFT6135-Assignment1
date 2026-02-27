"""
Problem 4, Question 4 — Plot UNet from scratch vs pretrained fine-tuned UNet.

Usage:
    python p4_q4_plot_results.py \
        --scratch_log    logs/unet/results.json \
        --pretrained_log logs/pretrained_unet/results.json
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch_log",    required=True)
    p.add_argument("--pretrained_log", required=True)
    p.add_argument("--out", default="report/p4_q4_curves.png")
    return p.parse_args()


def main():
    args      = parse_args()
    scratch   = load(args.scratch_log)
    pretrained = load(args.pretrained_log)

    epochs = range(1, len(scratch["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ME = 4  # show a marker every 4 epochs to avoid clutter

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, scratch["train_losses"],    "o-", color="steelblue",  markevery=ME, label="UNet (scratch) train")
    ax.plot(epochs, scratch["valid_losses"],    "s-", color="steelblue",  markevery=ME, label="UNet (scratch) val")
    ax.plot(epochs, pretrained["train_losses"], "o-", color="darkorange", markevery=ME, label="ResNet18-UNet (pretrained) train")
    ax.plot(epochs, pretrained["valid_losses"], "s-", color="darkorange", markevery=ME, label="ResNet18-UNet (pretrained) val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("DiceCE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.5)

    # ── Dice score ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, scratch["train_accs"],    "o-", color="steelblue",  markevery=ME, label="UNet (scratch) train")
    ax.plot(epochs, scratch["valid_accs"],    "s-", color="steelblue",  markevery=ME, label="UNet (scratch) val")
    ax.plot(epochs, pretrained["train_accs"], "o-", color="darkorange", markevery=ME, label="ResNet18-UNet (pretrained) train")
    ax.plot(epochs, pretrained["valid_accs"], "s-", color="darkorange", markevery=ME, label="ResNet18-UNet (pretrained) val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
    ax.set_title("Training and Validation Dice Score")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.5)

    print(f"\n{'Model':<30} {'Best val Dice':>14}")
    print("-" * 46)
    for name, log in [("UNet (from scratch)", scratch),
                      ("ResNet18-UNet (pretrained)", pretrained)]:
        print(f"{name:<30} {max(log['valid_accs']):>14.4f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {args.out}")


if __name__ == "__main__":
    main()
