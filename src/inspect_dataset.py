import argparse
from pathlib import Path
import numpy as np

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data if args.data is not None else cfg["demo_dir"])
    horizon = int(cfg["horizon"])

    files = sorted(data_dir.glob("*.npz"))

    print("=" * 60)
    print("Inspect Dataset")
    print("=" * 60)
    print("data_dir:", data_dir)
    print("num files:", len(files))
    print("horizon:", horizon)

    if len(files) == 0:
        print("No .npz files found.")
        return

    data = np.load(files[0])
    print("-" * 60)
    print("first file:", files[0])

    for k in data.files:
        arr = data[k]
        print(k, arr.shape, arr.dtype)

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    success = data["success"]

    print("-" * 60)
    print("Trajectory consistency check")
    print("obs length:", len(obs))
    print("actions length:", len(actions))
    print("rewards length:", len(rewards))
    print("success:", float(success))

    assert len(obs) == len(actions) + 1, "obs should be one longer than actions"
    assert len(actions) == len(rewards), "actions and rewards should have same length"

    if len(actions) >= horizon:
        s_t = obs[0]
        action_chunk = actions[0:horizon]
        s_future = obs[horizon]
        return_to_go = rewards[0:horizon].sum()

        print("-" * 60)
        print("Policy sample:")
        print("input s_t:", s_t.shape)
        print("target action_chunk:", action_chunk.shape)

        print("-" * 60)
        print("World model sample:")
        print("input s_t:", s_t.shape)
        print("input action_chunk:", action_chunk.reshape(-1).shape)
        print("target s_future:", s_future.shape)

        print("-" * 60)
        print("Value/Q sample:")
        print("state:", s_t.shape)
        print("score/return:", float(return_to_go))

    print("=" * 60)
    print("Dataset inspection passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
