"""Plot training loss curve and Precision-Recall curve from evaluation results.

Usage:
    python plot_results.py                          # uses eval_results.json + pr_data_iou07.json
    python plot_results.py --loss-log result-latest.txt  # plot loss from training log
"""

import argparse
import json
import re

import matplotlib.pyplot as plt


def parse_loss_log(path):
    """Extract (epochs, train_losses, val_aps) from training log file."""
    epochs, train_losses, val_aps = [], [], []
    val_epochs = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"Epoch\s+(\d+)/\d+.*train_loss=([\d.]+)", line)
            if m:
                ep = int(m.group(1))
                loss = float(m.group(2))
                epochs.append(ep)
                train_losses.append(loss)

                vm = re.search(r"val_AP@0\.5=([\d.]+)", line)
                if vm:
                    val_epochs.append(ep)
                    val_aps.append(float(vm.group(1)))

    return epochs, train_losses, val_epochs, val_aps


def plot_loss_curve(epochs, train_losses, val_epochs, val_aps, save_path="loss_curve.png"):
    """Plot training loss and validation AP curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    ax1.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curve")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Validation AP@0.5
    if val_epochs:
        ax2.plot(val_epochs, val_aps, "r-o", linewidth=1.5, markersize=4, label="Val AP@0.5")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AP@0.5")
        ax2.set_title("Validation AP@0.5")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")
    plt.close()


def plot_pr_curve(precisions, recalls, iou_thresh, ap, save_path="pr_curve.png"):
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(8, 6))

    # Interpolate for smooth curve
    plt.plot(recalls, precisions, "b-", linewidth=2, label=f"AP = {ap:.4f}")

    # Also plot filled area
    plt.fill_between(recalls, precisions, alpha=0.15)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (IoU = {iou_thresh})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"PR curve saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss-log", default="result-latest.txt")
    parser.add_argument("--eval-results", default="eval_results.json")
    parser.add_argument("--pr-data", default="pr_data_iou07.json")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Loss curve ──
    epochs, losses, val_epochs, val_aps = parse_loss_log(args.loss_log)
    if epochs:
        plot_loss_curve(
            epochs, losses, val_epochs, val_aps,
            save_path=os.path.join(args.output_dir, "loss_curve.png"),
        )
        # Also save numerical data
        with open(os.path.join(args.output_dir, "loss_data.txt"), "w") as f:
            for ep, loss in zip(epochs, losses):
                f.write(f"{ep} {loss}\n")

    # ── PR curve ──
    try:
        with open(args.pr_data) as f:
            pr_data = json.load(f)
        precisions = pr_data["precisions"]
        recalls = pr_data["recalls"]

        # Load AP value from eval results
        ap_07 = 0.0
        try:
            with open(args.eval_results) as f:
                eval_data = json.load(f)
            ap_07 = eval_data.get("AP@0.70", 0.0)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        if precisions and recalls:
            plot_pr_curve(
                precisions, recalls, 0.7, ap_07,
                save_path=os.path.join(args.output_dir, "pr_curve_iou07.png"),
            )
    except FileNotFoundError:
        print(f"PR data not found at {args.pr_data}, skipping PR curve.")

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
