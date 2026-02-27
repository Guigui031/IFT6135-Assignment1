"""
Problem 4, Question 2 — Plot learning-rate sweep results.

Usage (after all 5 training runs finish):
    python p4_q2_plot_results.py \
        --log_dirs logs/lr_0.1/results.json \
                   logs/lr_0.01/results.json \
                   logs/lr_0.001/results.json \
                   logs/lr_0.0001/results.json \
                   logs/lr_0.00001/results.json \
        --lrs 0.1 0.01 0.001 0.0001 0.00001
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


def load(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dirs", nargs="+", required=True,
                   help="One results.json per LR, in the same order as --lrs")
    p.add_argument("--lrs", nargs="+", type=float, required=True,
                   help="Learning rate values matching --log_dirs order")
    p.add_argument("--out", default="report/p4_q2_lr_curves.png")
    return p.parse_args()


def main():
    args = parse_args()
    assert len(args.log_dirs) == len(args.lrs), \
        "Number of log files must match number of LR values"

    logs = [load(p) for p in args.log_dirs]
    epochs = range(1, len(logs[0]["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ME = 4  # mark every 4 epochs

    for log, lr, color in zip(logs, args.lrs, LR_COLORS):
        label = f"lr={lr}"
        axes[0].plot(epochs, log["valid_losses"], "s-", color=color, markevery=ME, label=label)
        axes[1].plot(epochs, log["valid_accs"],   "s-", color=color, markevery=ME, label=label)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("DiceCE Loss (val)")
    axes[0].set_title("Validation Loss — LR sweep")
    axes[0].set_ylim(0, 2.0)   # clip spike at lr=0.1 epoch 11 (~24.5) for readability
    axes[0].legend(); axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice Score (val)")
    axes[1].set_title("Validation Dice Score — LR sweep")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(); axes[1].grid(True, linestyle="--", alpha=0.5)

    # Summary table
    print(f"\n{'LR':<12} {'Best val Dice':>14}")
    print("-" * 28)
    for log, lr in zip(logs, args.lrs):
        print(f"{lr:<12} {max(log['valid_accs']):>14.4f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {args.out}")


if __name__ == "__main__":
    main()
