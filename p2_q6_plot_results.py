"""
Problem 2, Question 6 — Plot training curves for MLP vs MobileNet.

Usage (after both training runs finish):
    python p6_plot_results.py --mlp_log logs/mlp/results.json \
                               --mobilenet_log logs/mobilenet/results.json
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
    p.add_argument("--mlp_log",       required=True,  help="Path to MLP results.json")
    p.add_argument("--mobilenet_log", required=True,  help="Path to MobileNet results.json")
    p.add_argument("--out", default="report/training_curves.png",
                   help="Output figure path (default: report/training_curves.png)")
    return p.parse_args()


def main():
    args = parse_args()
    mlp  = load(args.mlp_log)
    mob  = load(args.mobilenet_log)

    epochs = range(1, len(mlp["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, mlp["train_losses"],  "o-",  color="steelblue",   label="MLP train")
    ax.plot(epochs, mlp["valid_losses"],  "o--", color="steelblue",   label="MLP val")
    ax.plot(epochs, mob["train_losses"],  "s-",  color="darkorange",  label="MobileNet train")
    ax.plot(epochs, mob["valid_losses"],  "s--", color="darkorange",  label="MobileNet val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, mlp["train_accs"],  "o-",  color="steelblue",   label="MLP train")
    ax.plot(epochs, mlp["valid_accs"],  "o--", color="steelblue",   label="MLP val")
    ax.plot(epochs, mob["train_accs"],  "s-",  color="darkorange",  label="MobileNet train")
    ax.plot(epochs, mob["valid_accs"],  "s--", color="darkorange",  label="MobileNet val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # ── Print final numbers ───────────────────────────────────────────────────
    print(f"{'Model':<12} {'Best val acc':>14} {'Test acc':>10} {'Test loss':>12}")
    print("-" * 50)
    for name, log in [("MLP", mlp), ("MobileNet", mob)]:
        print(f"{name:<12} {max(log['valid_accs']):>14.4f} "
              f"{log['test_acc']:>10.4f} {log['test_loss']:>12.4f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {args.out}")


if __name__ == "__main__":
    main()
