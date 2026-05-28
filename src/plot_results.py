import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_losses(path):
    path = Path(path)
    if not path.exists():
        print(f"[Skip] Missing loss file: {path}")
        return None

    try:
        losses = torch.load(path, weights_only=True)
    except TypeError:
        losses = torch.load(path)

    values = []
    for x in losses:
        if hasattr(x, "detach"):
            values.append(float(x.detach().cpu()))
        else:
            values.append(float(x))
    return values


def plot_loss_curve(losses, title, save_path):
    if losses is None or len(losses) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def load_traj_stats(data_dir):
    files = sorted(Path(data_dir).glob("*.npz"))

    returns = []
    successes = []

    for f in files:
        data = np.load(f)
        returns.append(float(data["rewards"].sum()))
        successes.append(float(data["success"]))

    if len(files) == 0:
        return {
            "num_files": 0,
            "returns": np.array([]),
            "successes": np.array([]),
            "avg_return": np.nan,
            "min_return": np.nan,
            "max_return": np.nan,
            "success_rate": np.nan,
        }

    returns = np.asarray(returns, dtype=np.float32)
    successes = np.asarray(successes, dtype=np.float32)

    return {
        "num_files": len(files),
        "returns": returns,
        "successes": successes,
        "avg_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "success_rate": float(np.mean(successes)),
    }


def plot_return_hist(returns, title, save_path):
    if returns is None or len(returns) == 0:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(returns, bins=30)
    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def plot_bar(labels, values, ylabel, title, save_path):
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/metrics_expert_200.json")
    parser.add_argument("--out_dir", default="reports/figures")
    parser.add_argument("--demo_dir", default="data/demos")
    parser.add_argument("--rollout_dir", default="data/rollouts_bc")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Loss curves
    loss_files = {
        "Policy Loss": "logs/policy_losses.pt",
        "World Model Loss": "logs/world_losses.pt",
        "Value Model Loss": "logs/value_losses.pt",
        "Q Model Loss": "logs/q_losses.pt",
    }

    for title, path in loss_files.items():
        losses = load_losses(path)
        file_name = title.lower().replace(" ", "_") + ".png"
        plot_loss_curve(losses, title, out_dir / file_name)

    # 2. BC vs Planning performance
    methods = ["BC", "Planning-WV"]
    success_rates = [
        metrics["bc_policy"]["success_rate"],
        metrics["planning_wv"]["success_rate"],
    ]
    avg_returns = [
        metrics["bc_policy"]["avg_return"],
        metrics["planning_wv"]["avg_return"],
    ]

    plot_bar(
        methods,
        success_rates,
        "Success Rate",
        "BC vs Planning Success Rate",
        out_dir / "bc_vs_planning_success_rate.png",
    )

    plot_bar(
        methods,
        avg_returns,
        "Average Return",
        "BC vs Planning Average Return",
        out_dir / "bc_vs_planning_avg_return.png",
    )

    # 3. Dataset statistics
    demo_stats = load_traj_stats(args.demo_dir)
    rollout_stats = load_traj_stats(args.rollout_dir)

    plot_return_hist(
        demo_stats["returns"],
        "Expert Demo Return Distribution",
        out_dir / "expert_demo_return_distribution.png",
    )

    plot_return_hist(
        rollout_stats["returns"],
        "BC Rollout Return Distribution",
        out_dir / "bc_rollout_return_distribution.png",
    )

    plot_bar(
        ["Expert Demos", "BC Rollouts"],
        [demo_stats["success_rate"], rollout_stats["success_rate"]],
        "Success Rate",
        "Dataset Success Rate",
        out_dir / "dataset_success_rate.png",
    )

    plot_bar(
        ["Expert Demos", "BC Rollouts"],
        [demo_stats["avg_return"], rollout_stats["avg_return"]],
        "Average Return",
        "Dataset Average Return",
        out_dir / "dataset_avg_return.png",
    )

    # 4. Save summary CSV
    rows = [
        {
            "name": "expert_demos",
            "num": demo_stats["num_files"],
            "success_rate": demo_stats["success_rate"],
            "avg_return": demo_stats["avg_return"],
            "min_return": demo_stats["min_return"],
            "max_return": demo_stats["max_return"],
        },
        {
            "name": "bc_rollouts",
            "num": rollout_stats["num_files"],
            "success_rate": rollout_stats["success_rate"],
            "avg_return": rollout_stats["avg_return"],
            "min_return": rollout_stats["min_return"],
            "max_return": rollout_stats["max_return"],
        },
        {
            "name": "bc_eval",
            "num": 100,
            "success_rate": metrics["bc_policy"]["success_rate"],
            "avg_return": metrics["bc_policy"]["avg_return"],
            "min_return": "",
            "max_return": "",
        },
        {
            "name": "planning_wv_eval",
            "num": 100,
            "success_rate": metrics["planning_wv"]["success_rate"],
            "avg_return": metrics["planning_wv"]["avg_return"],
            "min_return": "",
            "max_return": "",
        },
    ]

    df = pd.DataFrame(rows)
    csv_path = Path("reports/summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("=" * 80)
    print("Visualization finished.")
    print("Figures saved to:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
