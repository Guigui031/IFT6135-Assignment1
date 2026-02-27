"""
Problem 4, Question 1 — Plot UNet vs UNetNoSkip training curves.

Usage:
    python p4_q1_plot_results.py --unet_log     logs/unet/results.json \
                                  --noskip_log   logs/unet_noskip/results.json
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
    p.add_argument("--unet_log",   required=True, help="Path to UNet results.json")
    p.add_argument("--noskip_log", required=True, help="Path to UNetNoSkip results.json")
    p.add_argument("--out", default="report/p4_q1_curves.png")
    return p.parse_args()


def main():
    args   = parse_args()
    unet   = load(args.unet_log)
    noskip = load(args.noskip_log)

    epochs = range(1, len(unet["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ME = 4  # show a marker every 4 epochs to avoid clutter

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, unet["train_losses"],   "o-", color="steelblue",  markevery=ME, label="UNet train")
    ax.plot(epochs, unet["valid_losses"],   "s-", color="steelblue",  markevery=ME, label="UNet val")
    ax.plot(epochs, noskip["train_losses"], "o-", color="darkorange", markevery=ME, label="UNet (no skip) train")
    ax.plot(epochs, noskip["valid_losses"], "s-", color="darkorange", markevery=ME, label="UNet (no skip) val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("DiceCE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    # ── Dice score ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, unet["train_accs"],   "o-", color="steelblue",  markevery=ME, label="UNet train")
    ax.plot(epochs, unet["valid_accs"],   "s-", color="steelblue",  markevery=ME, label="UNet val")
    ax.plot(epochs, noskip["train_accs"], "o-", color="darkorange", markevery=ME, label="UNet (no skip) train")
    ax.plot(epochs, noskip["valid_accs"], "s-", color="darkorange", markevery=ME, label="UNet (no skip) val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice Score")
    ax.set_title("Training and Validation Dice Score")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    print(f"{'Model':<20} {'Best val Dice':>14}")
    print("-" * 36)
    for name, log in [("UNet", unet), ("UNet (no skip)", noskip)]:
        print(f"{name:<20} {max(log['valid_accs']):>14.4f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {args.out}")


if __name__ == "__main__":
    main()
