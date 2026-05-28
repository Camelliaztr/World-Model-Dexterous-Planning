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

    values = []
    for x in losses:
        if hasattr(x, "detach"):
            values.append(float(x.detach().cpu()))
        else:
            values.append(float(x))
    return np.asarray(values, dtype=np.float32)


def load_returns(data_dir):
    files = sorted(Path(data_dir).glob("*.npz"))
    returns = []
    successes = []

    for f in files:
        data = np.load(f)
        returns.append(float(data["rewards"].sum()))
        successes.append(float(data["success"]))

    return np.asarray(returns, dtype=np.float32), np.asarray(successes, dtype=np.float32)


def save_bar(labels, values, ylabel, title, save_path, ylim=None):
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def save_grouped_bar(groups, series_names, values, ylabel, title, save_path, logy=False):
    values = np.asarray(values, dtype=np.float64)
    x = np.arange(len(groups))
    width = 0.8 / len(series_names)

    plt.figure(figsize=(8, 5))

    for i, name in enumerate(series_names):
        offset = (i - (len(series_names) - 1) / 2) * width
        plt.bar(x + offset, values[:, i], width, label=name)

    plt.xticks(x, groups)
    plt.ylabel(ylabel)
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def save_loss_plot(loss_dict, save_path, logy=True):
    plt.figure(figsize=(8, 5))

    for name, losses in loss_dict.items():
        if losses is None:
            continue
        x = np.arange(1, len(losses) + 1)
        plt.plot(x, losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def save_boxplot(data, labels, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def save_hist_overlay(demo_returns, rollout_returns, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(demo_returns, bins=40, alpha=0.6, label="Expert Demos")
    plt.hist(rollout_returns, bins=40, alpha=0.6, label="BC Rollouts")
    plt.xlabel("Episode Return")
    plt.ylabel("Count")
    plt.title("Return Distribution: Expert Demos vs BC Rollouts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def main():
    metrics_path = Path("reports/metrics_expert1000.json")
    out_dir = Path("reports/figures_expert1000")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(metrics_path)

    # 1. Online BC vs Planning
    methods = ["BC Policy", "Planning-WV"]
    success = [
        metrics["online_eval"]["BC Policy"]["success_rate"],
        metrics["online_eval"]["Planning-WV"]["success_rate"],
    ]
    returns = [
        metrics["online_eval"]["BC Policy"]["avg_return"],
        metrics["online_eval"]["Planning-WV"]["avg_return"],
    ]

    save_bar(
        methods,
        success,
        "Success Rate",
        "Online Evaluation Success Rate",
        out_dir / "online_success_rate_bc_vs_planning.png",
        ylim=(0, 1.05),
    )

    save_bar(
        methods,
        returns,
        "Average Return",
        "Online Evaluation Average Return",
        out_dir / "online_avg_return_bc_vs_planning.png",
    )

    improvement = returns[1] - returns[0]
    relative = improvement / returns[0] * 100.0

    save_bar(
        ["Planning-WV - BC"],
        [improvement],
        "Average Return Improvement",
        f"Planning Return Improvement: +{relative:.2f}%",
        out_dir / "online_return_improvement.png",
    )

    # 2. Split statistics
    split_names = ["train", "val", "test"]

    demo_success = [metrics["expert_demos"][s]["success_rate"] for s in split_names]
    rollout_success = [metrics["bc_rollouts"][s]["success_rate"] for s in split_names]
    demo_return = [metrics["expert_demos"][s]["avg_return"] for s in split_names]
    rollout_return = [metrics["bc_rollouts"][s]["avg_return"] for s in split_names]

    save_grouped_bar(
        split_names,
        ["Expert Demos", "BC Rollouts"],
        np.column_stack([demo_success, rollout_success]),
        "Success Rate",
        "Train / Val / Test Success Rate",
        out_dir / "split_success_rate_comparison.png",
        logy=False,
    )

    save_grouped_bar(
        split_names,
        ["Expert Demos", "BC Rollouts"],
        np.column_stack([demo_return, rollout_return]),
        "Average Return",
        "Train / Val / Test Average Return",
        out_dir / "split_avg_return_comparison.png",
        logy=False,
    )

    # 3. Supervised MSE
    model_names = ["BC Policy", "World Model", "Value Model", "Q Model"]
    mse_values = []

    for model in model_names:
        mse_values.append([
            metrics["supervised_mse"][model]["train"],
            metrics["supervised_mse"][model]["val"],
            metrics["supervised_mse"][model]["test"],
        ])

    save_grouped_bar(
        model_names,
        ["train", "val", "test"],
        mse_values,
        "MSE, log scale",
        "Supervised Train / Val / Test MSE",
        out_dir / "supervised_mse_train_val_test_log.png",
        logy=True,
    )

    # 4. Training losses
    policy_losses = load_losses("logs_expert1000/policy_losses.pt")
    world_losses = load_losses("logs_expert1000/world_losses.pt")
    value_losses = load_losses("logs_expert1000/value_losses.pt")
    q_losses = load_losses("logs_expert1000/q_losses.pt")

    save_loss_plot(
        {
            "BC Policy": policy_losses,
            "World Model": world_losses,
            "Value Model": value_losses,
            "Q Model": q_losses,
        },
        out_dir / "training_loss_comparison_log.png",
        logy=True,
    )

    # 5. Return distributions
    demo_returns, demo_successes = load_returns("data/demos_expert1000")
    rollout_returns, rollout_successes = load_returns("data/rollouts_bc_expert1000_raw")

    save_boxplot(
        [demo_returns, rollout_returns],
        ["Expert Demos", "BC Rollouts"],
        "Episode Return",
        "Return Distribution Boxplot",
        out_dir / "return_distribution_boxplot.png",
    )

    save_hist_overlay(
        demo_returns,
        rollout_returns,
        out_dir / "return_distribution_overlay.png",
    )

    # 6. Save summary csv
    rows = [
        {
            "name": "BC Policy",
            "success_rate": success[0],
            "avg_return": returns[0],
            "avg_planning_score": "",
        },
        {
            "name": "Planning-WV",
            "success_rate": success[1],
            "avg_return": returns[1],
            "avg_planning_score": metrics["online_eval"]["Planning-WV"]["avg_planning_score"],
        },
        {
            "name": "Planning improvement",
            "success_rate": "",
            "avg_return": improvement,
            "avg_planning_score": "",
        },
        {
            "name": "Planning relative improvement %",
            "success_rate": "",
            "avg_return": relative,
            "avg_planning_score": "",
        },
    ]

    df = pd.DataFrame(rows)
    summary_path = Path("reports/expert1000_visual_summary.csv")
    df.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    print("=" * 80)
    print("Expert1000 visualization finished.")
    print("Output directory:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
