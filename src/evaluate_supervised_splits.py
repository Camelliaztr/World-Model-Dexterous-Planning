import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils import load_config, device
from src.models.mlp import MLP


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        cleaned[new_k] = v
    return cleaned


def infer_dims(state_dict):
    weight_keys = sorted([k for k in state_dict.keys() if k.endswith(".weight")])
    bias_keys = sorted([k for k in state_dict.keys() if k.endswith(".bias")])
    in_dim = int(state_dict[weight_keys[0]].shape[1])
    out_dim = int(state_dict[bias_keys[-1]].shape[0])
    return in_dim, out_dim


def load_mlp(path, hidden_dim, num_layers, dev):
    ckpt = torch.load(path, map_location=dev)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {path}")

    state_dict = clean_state_dict(state_dict)
    in_dim, out_dim = infer_dims(state_dict)

    model = MLP(
        in_dim,
        out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)

    model.load_state_dict(state_dict)
    model.eval()

    return model


def batched_mse(model, xs, ys, dev, batch_size=4096):
    if len(xs) == 0:
        return np.nan

    total_se = 0.0
    total_numel = 0

    with torch.no_grad():
        for start in range(0, len(xs), batch_size):
            xb = torch.as_tensor(xs[start:start + batch_size], dtype=torch.float32, device=dev)
            yb = torch.as_tensor(ys[start:start + batch_size], dtype=torch.float32, device=dev)

            pred = model(xb)
            total_se += torch.sum((pred - yb) ** 2).item()
            total_numel += yb.numel()

    return total_se / max(total_numel, 1)


def collect_policy_samples(data_dir, horizon):
    xs, ys = [], []
    files = sorted(Path(data_dir).glob("*.npz"))

    for f in files:
        data = np.load(f)
        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)

        max_t = len(actions) - horizon + 1

        for t in range(max_t):
            xs.append(obs[t])
            ys.append(actions[t:t + horizon].reshape(-1))

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect_world_samples(data_dir, horizon):
    xs, ys = [], []
    files = sorted(Path(data_dir).glob("*.npz"))

    for f in files:
        data = np.load(f)
        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)

        max_t = len(actions) - horizon + 1

        for t in range(max_t):
            action_chunk = actions[t:t + horizon].reshape(-1)
            x = np.concatenate([obs[t], action_chunk], axis=0)
            y = obs[t + horizon]
            xs.append(x)
            ys.append(y)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect_value_samples(data_dir, horizon):
    xs, ys = [], []
    files = sorted(Path(data_dir).glob("*.npz"))

    for f in files:
        data = np.load(f)
        obs = data["obs"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)

        max_t = len(rewards) - horizon + 1

        for t in range(max_t):
            target = np.asarray([rewards[t:t + horizon].sum()], dtype=np.float32)
            xs.append(obs[t])
            ys.append(target)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def collect_q_samples(data_dir, horizon):
    xs, ys = [], []
    files = sorted(Path(data_dir).glob("*.npz"))

    for f in files:
        data = np.load(f)
        obs = data["obs"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)

        max_t = len(actions) - horizon + 1

        for t in range(max_t):
            action_chunk = actions[t:t + horizon].reshape(-1)
            x = np.concatenate([obs[t], action_chunk], axis=0)
            target = np.asarray([rewards[t:t + horizon].sum()], dtype=np.float32)
            xs.append(x)
            ys.append(target)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def evaluate_one_model(model_name, model, split_dirs, collector, horizon, dev):
    rows = []

    for split_name, data_dir in split_dirs.items():
        xs, ys = collector(data_dir, horizon)
        mse = batched_mse(model, xs, ys, dev)

        row = {
            "model": model_name,
            "split": split_name,
            "num_samples": len(xs),
            "mse": mse,
        }

        rows.append(row)

        print(
            model_name,
            split_name,
            "num_samples:",
            len(xs),
            "mse:",
            mse,
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate_expert1000_wm.yaml")
    parser.add_argument("--demo_root", default="data/splits/demos_expert1000")
    parser.add_argument("--rollout_root", default="data/splits/rollouts_bc_expert1000")
    parser.add_argument("--out_csv", default="reports/supervised_split_eval_expert1000.csv")
    parser.add_argument("--out_md", default="reports/supervised_split_eval_expert1000.md")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev = device()

    hidden_dim = int(cfg["hidden_dim"])
    num_layers = int(cfg["num_layers"])
    horizon = int(cfg["horizon"])

    ckpt_dir = Path(cfg["checkpoint_dir"])

    policy = load_mlp(ckpt_dir / "policy.pt", hidden_dim, num_layers, dev)
    world = load_mlp(ckpt_dir / "world.pt", hidden_dim, num_layers, dev)
    value = load_mlp(ckpt_dir / "value.pt", hidden_dim, num_layers, dev)
    q_model = load_mlp(ckpt_dir / "q.pt", hidden_dim, num_layers, dev)

    demo_split_dirs = {
        "train": Path(args.demo_root) / "train",
        "val": Path(args.demo_root) / "val",
        "test": Path(args.demo_root) / "test",
    }

    rollout_split_dirs = {
        "train": Path(args.rollout_root) / "train",
        "val": Path(args.rollout_root) / "val",
        "test": Path(args.rollout_root) / "test",
    }

    rows = []

    rows += evaluate_one_model(
        "BC Policy",
        policy,
        demo_split_dirs,
        collect_policy_samples,
        horizon,
        dev,
    )

    rows += evaluate_one_model(
        "World Model",
        world,
        rollout_split_dirs,
        collect_world_samples,
        horizon,
        dev,
    )

    rows += evaluate_one_model(
        "Value Model",
        value,
        rollout_split_dirs,
        collect_value_samples,
        horizon,
        dev,
    )

    rows += evaluate_one_model(
        "Q Model",
        q_model,
        rollout_split_dirs,
        collect_q_samples,
        horizon,
        dev,
    )

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Supervised Train / Val / Test Evaluation\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write("说明：BC Policy 评估 action chunk prediction MSE；World Model 评估 future observation prediction MSE；Value/Q Model 这里使用 horizon 内累计 reward 作为 chunk-level return target，用于评估价值打分模型的泛化趋势。\n")

    print("=" * 80)
    print("Saved:", out_csv)
    print("Saved:", out_md)
    print(df)
    print("=" * 80)


if __name__ == "__main__":
    main()
