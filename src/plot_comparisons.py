import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_losses(path):
    path = Path(path)
    if not path.exists():
        print(f"[Skip] missing {path}")
        return None

    try:
        losses = torch.load(path, weights_only=True)
    except TypeError:
        losses = torch.load(path)

    out = []
    for x in losses:
        if hasattr(x, "detach"):
            out.append(float(x.detach().cpu()))
        else:
            out.append(float(x))
    return np.asarray(out, dtype=np.float32)


def load_returns(data_dir):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))

    returns = []
    successes = []

    for f in files:
        data = np.load(f)
        returns.append(float(data["rewards"].sum()))
        successes.append(float(data["success"]))

    return np.asarray(returns, dtype=np.float32), np.asarray(successes, dtype=np.float32)


def save_bar(labels, values, ylabel, title, path, ylim=None):
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)


def save_line(losses_dict, ylabel, title, path, logy=False):
    plt.figure(figsize=(8, 5))

    for name, losses in losses_dict.items():
        if losses is None:
            continue
        x = np.arange(1, len(losses) + 1)
        plt.plot(x, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)


def save_boxplot(data, labels, ylabel, title, path):
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)


def save_hist_overlay(demo_returns, rollout_returns, path):
    plt.figure(figsize=(8, 5))
    plt.hist(demo_returns, bins=30, alpha=0.6, label="Expert Demos")
    plt.hist(rollout_returns, bins=30, alpha=0.6, label="BC Rollouts")
    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    plt.title("Return Distribution: Expert Demos vs BC Rollouts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/comparison_metrics.json")
    parser.add_argument("--demo_dir", default="data/demos")
    parser.add_argument("--rollout_dir", default="data/rollouts_bc")
    parser.add_argument("--out_dir", default="reports/figures_compare")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(args.metrics)

    demo_returns, demo_successes = load_returns(args.demo_dir)
    rollout_returns, rollout_successes = load_returns(args.rollout_dir)

    # 1. Method success-rate comparison
    method_labels = ["BC Eval", "Planning-WV"]
    method_success = [
        metrics["bc_eval"]["success_rate"],
        metrics["planning_wv_eval"]["success_rate"],
    ]
    save_bar(
        method_labels,
        method_success,
        "Success Rate",
        "Policy Evaluation Success Rate",
        out_dir / "method_success_rate_comparison.png",
        ylim=(0, 1.05),
    )

    # 2. Method average return comparison
    method_return = [
        metrics["bc_eval"]["avg_return"],
        metrics["planning_wv_eval"]["avg_return"],
    ]
    save_bar(
        method_labels,
        method_return,
        "Average Return",
        "Policy Evaluation Average Return",
        out_dir / "method_avg_return_comparison.png",
    )

    # 3. Dataset success-rate comparison
    dataset_labels = ["Expert Demos", "BC Rollouts"]
    dataset_success = [
        float(np.mean(demo_successes)),
        float(np.mean(rollout_successes)),
    ]
    save_bar(
        dataset_labels,
        dataset_success,
        "Success Rate",
        "Dataset Success Rate Comparison",
        out_dir / "dataset_success_rate_comparison.png",
        ylim=(0, 1.05),
    )

    # 4. Dataset average return comparison
    dataset_return = [
        float(np.mean(demo_returns)),
        float(np.mean(rollout_returns)),
    ]
    save_bar(
        dataset_labels,
        dataset_return,
        "Average Return",
        "Dataset Average Return Comparison",
        out_dir / "dataset_avg_return_comparison.png",
    )

    # 5. Return distribution boxplot
    save_boxplot(
        [demo_returns, rollout_returns],
        ["Expert Demos", "BC Rollouts"],
        "Episode Return",
        "Return Distribution Boxplot",
        out_dir / "return_distribution_boxplot.png",
    )

    # 6. Return distribution overlay
    save_hist_overlay(
        demo_returns,
        rollout_returns,
        out_dir / "return_distribution_overlay.png",
    )

    # 7. Loss comparison, raw
    policy_losses = load_losses("logs/policy_losses.pt")
    world_losses = load_losses("logs/world_losses.pt")
    value_losses = load_losses("logs/value_losses.pt")
    q_losses = load_losses("logs/q_losses.pt")

    save_line(
        {
            "Policy": policy_losses,
            "World": world_losses,
            "Value": value_losses,
            "Q": q_losses,
        },
        "Loss",
        "Training Loss Comparison",
        out_dir / "training_loss_comparison_raw.png",
        logy=False,
    )

    # 8. Loss comparison, log scale
    save_line(
        {
            "Policy": policy_losses,
            "World": world_losses,
            "Value": value_losses,
            "Q": q_losses,
        },
        "Loss, log scale",
        "Training Loss Comparison, Log Scale",
        out_dir / "training_loss_comparison_log.png",
        logy=True,
    )

    # 9. Video example comparison
    video_labels = ["BC Video", "Planning-WV Video"]
    video_returns = [
        metrics["video_examples"]["bc_video"]["return"],
        metrics["video_examples"]["planning_wv_video_seed42"]["return"],
    ]
    save_bar(
        video_labels,
        video_returns,
        "Single-Rollout Return",
        "Video Example Return Comparison",
        out_dir / "video_example_return_comparison.png",
    )

    # 10. Summary CSV
    rows = [
        {
            "category": "dataset",
            "name": "expert_demos",
            "num": len(demo_returns),
            "success_rate": float(np.mean(demo_successes)),
            "avg_return": float(np.mean(demo_returns)),
            "min_return": float(np.min(demo_returns)),
            "max_return": float(np.max(demo_returns)),
        },
        {
            "category": "dataset",
            "name": "bc_rollouts",
            "num": len(rollout_returns),
            "success_rate": float(np.mean(rollout_successes)),
            "avg_return": float(np.mean(rollout_returns)),
            "min_return": float(np.min(rollout_returns)),
            "max_return": float(np.max(rollout_returns)),
        },
        {
            "category": "evaluation",
            "name": "bc_eval",
            "num": metrics["bc_eval"]["num_episodes"],
            "success_rate": metrics["bc_eval"]["success_rate"],
            "avg_return": metrics["bc_eval"]["avg_return"],
            "min_return": "",
            "max_return": "",
        },
        {
            "category": "evaluation",
            "name": "planning_wv_eval",
            "num": metrics["planning_wv_eval"]["num_episodes"],
            "success_rate": metrics["planning_wv_eval"]["success_rate"],
            "avg_return": metrics["planning_wv_eval"]["avg_return"],
            "min_return": "",
            "max_return": "",
        },
        {
            "category": "video",
            "name": "bc_video",
            "num": 1,
            "success_rate": 1.0 if metrics["video_examples"]["bc_video"]["success"] else 0.0,
            "avg_return": metrics["video_examples"]["bc_video"]["return"],
            "min_return": "",
            "max_return": "",
        },
        {
            "category": "video",
            "name": "planning_wv_video_seed42",
            "num": 1,
            "success_rate": 1.0 if metrics["video_examples"]["planning_wv_video_seed42"]["success"] else 0.0,
            "avg_return": metrics["video_examples"]["planning_wv_video_seed42"]["return"],
            "min_return": "",
            "max_return": "",
        },
    ]

    df = pd.DataFrame(rows)
    csv_path = Path("reports/comparison_summary.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("=" * 80)
    print("Comparison figures generated.")
    print("Output directory:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
