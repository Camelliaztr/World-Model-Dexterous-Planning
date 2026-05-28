import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_result_from_stdout(stdout):
    """
    evaluate_planning.py 最后会打印类似：
    {'mode': 'wv', 'success_rate': 0.84, 'avg_return': 2993.68, ...}

    这里从 stdout 里找到最后一个 Python dict。
    """
    result = None

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                result = ast.literal_eval(line)
            except Exception:
                pass

    if result is None:
        raise RuntimeError("Could not parse result dict from evaluate_planning output.")

    return result


def run_one(config_path, noise_std, mode, episodes, out_dir):
    cfg = load_config(config_path)
    cfg["noise_std"] = float(noise_std)

    if episodes is not None:
        cfg["num_eval_episodes"] = int(episodes)

    tmp_config = out_dir / f"tmp_noise_{noise_std}.yaml"
    save_config(cfg, tmp_config)

    cmd = [
        sys.executable,
        "-m",
        "src.evaluate_planning",
        "--config",
        str(tmp_config),
    ]

    print("=" * 80)
    print("Running planning evaluation")
    print("noise_std:", noise_std)
    print("mode:", mode)
    print("config:", tmp_config)
    print("=" * 80)

    completed = subprocess.run(
        cmd,
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(completed.stdout)

    if completed.returncode != 0:
        raise RuntimeError(f"evaluate_planning failed for noise_std={noise_std}")

    result = parse_result_from_stdout(completed.stdout)

    row = {
        "noise_std": float(noise_std),
        "mode": result.get("mode", mode),
        "success_rate": float(result["success_rate"]),
        "avg_return": float(result["avg_return"]),
        "avg_planning_score": float(result.get("avg_planning_score", 0.0)),
        "num_eval_episodes": int(cfg.get("num_eval_episodes", -1)),
        "horizon": int(cfg.get("horizon", -1)),
        "execute_steps": int(cfg.get("execute_steps", -1)),
        "num_candidates": int(cfg.get("num_candidates", -1)),
    }

    return row


def plot_line(df, x_col, y_col, title, ylabel, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print("Saved:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--out_dir", default="reports/planning_ablation")
    parser.add_argument("--mode", default="wv")
    parser.add_argument(
        "--noise_values",
        nargs="+",
        type=float,
        default=[0.00, 0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="覆盖配置里的 num_eval_episodes。默认不覆盖。",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for noise_std in args.noise_values:
        row = run_one(
            config_path=args.config,
            noise_std=noise_std,
            mode=args.mode,
            episodes=args.episodes,
            out_dir=out_dir,
        )
        rows.append(row)

        # 每跑完一个点就保存一次，防止中途失败数据丢失
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(out_dir / "noise_std_ablation_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df = df.sort_values("noise_std").reset_index(drop=True)

    csv_path = out_dir / "noise_std_ablation.csv"
    json_path = out_dir / "noise_std_ablation.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)

    plot_line(
        df,
        "noise_std",
        "success_rate",
        "Planning Ablation: noise_std vs Success Rate",
        "Success Rate",
        out_dir / "noise_std_success_rate.png",
    )

    plot_line(
        df,
        "noise_std",
        "avg_return",
        "Planning Ablation: noise_std vs Average Return",
        "Average Return",
        out_dir / "noise_std_avg_return.png",
    )

    plot_line(
        df,
        "noise_std",
        "avg_planning_score",
        "Planning Ablation: noise_std vs Planning Score",
        "Average Planning Score",
        out_dir / "noise_std_planning_score.png",
    )

    print("=" * 80)
    print("Ablation finished.")
    print(df)
    print("=" * 80)


if __name__ == "__main__":
    main()
