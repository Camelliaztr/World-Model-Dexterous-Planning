import argparse
from pathlib import Path

import minari
import numpy as np
from tqdm import tqdm


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def flatten_obs(obs):
    """
    D4RL/relocate/expert-v2 的 observation 正常应为 ndarray，形状类似：
    (T + 1, 39)

    这里也兼容 dict observation。
    """
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            value = np.asarray(obs[key])
            parts.append(value.reshape(value.shape[0], -1))
        return np.concatenate(parts, axis=-1)

    return np.asarray(obs)


def extract_success(episode):
    """
    尽量从 infos 中提取 success。
    如果没有 success / is_success 字段，就退化为 0.0。

    注意：
    对 D4RL/Adroit 数据，success 字段不一定总是存在。
    后续正式评估仍以环境 rollout 的 success_rate 为准。
    """
    infos = getattr(episode, "infos", None)

    if isinstance(infos, dict):
        for key in ["success", "is_success", "goal_achieved"]:
            if key in infos:
                value = np.asarray(infos[key])
                return float(np.any(value))

    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="D4RL/relocate/expert-v2",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/demos",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=200,
        help="最多转换多少条 episode。设为 -1 表示全部转换。",
    )
    parser.add_argument(
        "--clear_old",
        action="store_true",
        help="转换前删除 out_dir 里的旧 .npz 文件。",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_old:
        for p in out_dir.glob("*.npz"):
            p.unlink()

    print("=" * 80)
    print("Loading Minari dataset:", args.dataset_id)
    print("=" * 80)

    dataset = minari.load_dataset(args.dataset_id)

    print("Dataset loaded.")
    print("Output dir:", out_dir)
    print("Max episodes:", args.max_episodes)

    converted = 0
    skipped = 0

    for episode in tqdm(dataset.iterate_episodes(), desc="convert episodes"):
        if args.max_episodes > 0 and converted >= args.max_episodes:
            break

        obs = flatten_obs(episode.observations).astype(np.float32)
        actions = to_numpy(episode.actions).astype(np.float32)
        rewards = to_numpy(episode.rewards).astype(np.float32)

        if obs.ndim != 2:
            print("Skip episode because obs ndim != 2:", obs.shape)
            skipped += 1
            continue

        if actions.ndim != 2:
            print("Skip episode because actions ndim != 2:", actions.shape)
            skipped += 1
            continue

        if rewards.ndim != 1:
            rewards = rewards.reshape(-1).astype(np.float32)

        if len(obs) != len(actions) + 1:
            print(
                "Skip episode because len(obs) != len(actions) + 1:",
                "obs =", len(obs),
                "actions =", len(actions),
            )
            skipped += 1
            continue

        if len(actions) != len(rewards):
            print(
                "Skip episode because len(actions) != len(rewards):",
                "actions =", len(actions),
                "rewards =", len(rewards),
            )
            skipped += 1
            continue

        success = np.asarray(extract_success(episode), dtype=np.float32)

        save_path = out_dir / f"expert_ep_{converted:05d}.npz"

        np.savez_compressed(
            save_path,
            obs=obs,
            actions=actions,
            rewards=rewards,
            success=success,
        )

        converted += 1

    print("=" * 80)
    print("Conversion finished.")
    print("Converted episodes:", converted)
    print("Skipped episodes:", skipped)
    print("Saved to:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
