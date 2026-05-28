from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_losses(path):
    losses = torch.load(path)

    values = []
    for x in losses:
        if hasattr(x, "detach"):
            values.append(float(x.detach().cpu()))
        else:
            values.append(float(x))

    return np.asarray(values, dtype=np.float32)


def get_mse(df, model, split):
    row = df[(df["model"] == model) & (df["split"] == split)]
    if len(row) == 0:
        return None
    return float(row["mse"].iloc[0])


def plot_one_model(ax, model_name, loss_file, mse_df, title):
    train_losses = load_losses(loss_file)
    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(
        epochs,
        train_losses,
        linewidth=2.2,
        label="Train loss during training",
    )

    train_mse = get_mse(mse_df, model_name, "train")
    val_mse = get_mse(mse_df, model_name, "val")
    test_mse = get_mse(mse_df, model_name, "test")

    if train_mse is not None:
        ax.axhline(
            train_mse,
            linestyle="--",
            linewidth=1.6,
            label=f"Final train MSE = {train_mse:.3g}",
        )

    if val_mse is not None:
        ax.axhline(
            val_mse,
            linestyle="--",
            linewidth=1.6,
            label=f"Final val MSE = {val_mse:.3g}",
        )

    if test_mse is not None:
        ax.axhline(
            test_mse,
            linestyle="--",
            linewidth=1.6,
            label=f"Final test MSE = {test_mse:.3g}",
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    out_dir = Path("reports/figures_expert1000")
    out_dir.mkdir(parents=True, exist_ok=True)

    mse_csv = Path("reports/supervised_split_eval_expert1000.csv")
    mse_df = pd.read_csv(mse_csv)

    configs = [
        {
            "model": "BC Policy",
            "loss_file": "logs_expert1000/policy_losses.pt",
            "title": "BC Policy",
        },
        {
            "model": "World Model",
            "loss_file": "logs_expert1000/world_losses.pt",
            "title": "World Model",
        },
        {
            "model": "Value Model",
            "loss_file": "logs_expert1000/value_losses.pt",
            "title": "Value Model",
        },
        {
            "model": "Q Model",
            "loss_file": "logs_expert1000/q_losses.pt",
            "title": "Q Model",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, cfg in zip(axes, configs):
        plot_one_model(
            ax=ax,
            model_name=cfg["model"],
            loss_file=cfg["loss_file"],
            mse_df=mse_df,
            title=cfg["title"],
        )

    fig.suptitle(
        "Train / Val / Test Convergence Comparison",
        fontsize=18,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / "train_val_test_convergence_comparison.png"
    plt.savefig(out_path, dpi=240)
    plt.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
