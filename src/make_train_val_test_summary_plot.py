from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_dir = Path("reports/figures_expert1000")
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_csv = Path("data/splits/demos_expert1000/demos_expert1000_split_stats.csv")
    rollout_csv = Path("data/splits/rollouts_bc_expert1000/rollouts_bc_expert1000_split_stats.csv")
    mse_csv = Path("reports/supervised_split_eval_expert1000.csv")

    demo = pd.read_csv(demo_csv)
    rollout = pd.read_csv(rollout_csv)
    mse = pd.read_csv(mse_csv)

    # 1. 整理 demo 和 rollout 指标
    demo_small = demo[[
        "split", "num", "success_rate", "avg_return", "min_return", "max_return"
    ]].rename(columns={
        "num": "expert_num",
        "success_rate": "expert_success_rate",
        "avg_return": "expert_avg_return",
        "min_return": "expert_min_return",
        "max_return": "expert_max_return",
    })

    rollout_small = rollout[[
        "split", "num", "success_rate", "avg_return", "min_return", "max_return"
    ]].rename(columns={
        "num": "rollout_num",
        "success_rate": "rollout_success_rate",
        "avg_return": "rollout_avg_return",
        "min_return": "rollout_min_return",
        "max_return": "rollout_max_return",
    })

    summary = pd.merge(demo_small, rollout_small, on="split", how="outer")

    # 2. 整理 MSE 指标
    mse_pivot = mse.pivot(index="split", columns="model", values="mse").reset_index()
    mse_pivot = mse_pivot.rename(columns={
        "BC Policy": "bc_policy_mse",
        "World Model": "world_model_mse",
        "Value Model": "value_model_mse",
        "Q Model": "q_model_mse",
    })

    summary = pd.merge(summary, mse_pivot, on="split", how="outer")

    # 3. 固定 split 顺序
    split_order = {"train": 0, "val": 1, "test": 2}
    summary["split_order"] = summary["split"].map(split_order)
    summary = summary.sort_values("split_order").drop(columns=["split_order"])

    # 4. 保存总表
    out_csv = Path("reports/train_val_test_full_comparison_expert1000.csv")
    out_md = Path("reports/train_val_test_full_comparison_expert1000.md")

    summary.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Train / Validation / Test Full Comparison\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print("Saved table:", out_csv)
    print("Saved table:", out_md)

    # 5. 平均回报曲线
    x = summary["split"].tolist()

    plt.figure(figsize=(8, 5))
    plt.plot(x, summary["expert_avg_return"], marker="o", label="Expert Demos")
    plt.plot(x, summary["rollout_avg_return"], marker="o", label="BC Rollouts")
    plt.xlabel("Split")
    plt.ylabel("Average Return")
    plt.title("Train / Val / Test Average Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_test_avg_return_curve.png", dpi=220)
    plt.close()

    # 6. 成功率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(x, summary["expert_success_rate"], marker="o", label="Expert Demos")
    plt.plot(x, summary["rollout_success_rate"], marker="o", label="BC Rollouts")
    plt.xlabel("Split")
    plt.ylabel("Success Rate")
    plt.title("Train / Val / Test Success Rate")
    plt.ylim(0.9, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_test_success_rate_curve.png", dpi=220)
    plt.close()

    # 7. 监督学习 MSE 曲线，log scale
    plt.figure(figsize=(8, 5))
    plt.plot(x, summary["bc_policy_mse"], marker="o", label="BC Policy")
    plt.plot(x, summary["world_model_mse"], marker="o", label="World Model")
    plt.plot(x, summary["value_model_mse"], marker="o", label="Value Model")
    plt.plot(x, summary["q_model_mse"], marker="o", label="Q Model")
    plt.xlabel("Split")
    plt.ylabel("MSE, log scale")
    plt.title("Train / Val / Test Supervised MSE")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "train_val_test_supervised_mse_curve_log.png", dpi=220)
    plt.close()

    print("Saved figure:", out_dir / "train_val_test_avg_return_curve.png")
    print("Saved figure:", out_dir / "train_val_test_success_rate_curve.png")
    print("Saved figure:", out_dir / "train_val_test_supervised_mse_curve_log.png")

    print("=" * 80)
    print(summary)
    print("=" * 80)


if __name__ == "__main__":
    main()
