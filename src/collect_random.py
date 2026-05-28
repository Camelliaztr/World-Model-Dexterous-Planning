# src/collect_random.py
import argparse
import numpy as np
from tqdm import trange
from src.utils import load_config, set_seed
from src.envs.adroit_env import make_env, flatten_obs, get_success
from src.datasets.trajectory import save_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    env = make_env(cfg["env_id"], cfg["seed"])

    for ep in trange(args.episodes):
        obs, _ = env.reset(seed=cfg["seed"] + ep)
        obs_list, act_list, rew_list = [flatten_obs(obs)], [], []
        done, success = False, False
        while not done and len(act_list) < cfg["max_episode_steps"]:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            success = success or get_success(info, reward, cfg["success_reward_threshold"])
            act_list.append(action)
            rew_list.append(reward)
            obs_list.append(flatten_obs(next_obs))
        save_trajectory(
            f"{cfg['random_dir']}/random_ep_{ep:04d}.npz",
            obs_list,
            act_list,
            rew_list,
            success,
        )
    env.close()


if __name__ == "__main__":
    main()