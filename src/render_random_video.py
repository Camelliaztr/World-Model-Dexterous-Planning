import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="AdroitHandRelocate-v1")
    parser.add_argument("--out", default="reports/videos/random_relocate.mp4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)

    env = gym.make(
        args.env_id,
        render_mode="rgb_array",
        width=640,
        height=480,
    )

    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    frames = []

    for t in range(args.steps):
        frame = env.render()
        frames.append(frame)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_path, frames, fps=30)

    print("Saved video to:", out_path)
    print("num frames:", len(frames))


if __name__ == "__main__":
    main()
