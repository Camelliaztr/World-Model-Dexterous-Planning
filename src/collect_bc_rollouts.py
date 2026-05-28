# src/collect_bc_rollouts.py
import argparse
import os
import torch
from tqdm import trange
from src.utils import load_config, set_seed, device
from src.envs.adroit_env import make_env, flatten_obs, get_success
from src.datasets.trajectory import save_trajectory
from src.evaluate_bc import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dev = device()
    env = make_env(cfg["env_id"], cfg["seed"])
    policy, _ = load_policy(cfg, dev)

    for ep in trange(args.episodes):
        obs, _ = env.reset(seed=cfg["seed"] + 2000 + ep)
        obs_list, act_list, rew_list = [flatten_obs(obs)], [], []
        done, success, steps = False, False, 0
        while not done and steps < cfg["max_episode_steps"]:
            s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=dev)
            with torch.no_grad():
                chunk = policy(s.unsqueeze(0)).view(cfg["horizon"], -1).cpu().numpy()
            for i in range(min(cfg["execute_steps"], len(chunk))):
                obs, reward, terminated, truncated, info = env.step(chunk[i])
                done = terminated or truncated
                success = success or get_success(info, reward, cfg["success_reward_threshold"])
                act_list.append(chunk[i])
                rew_list.append(reward)
                obs_list.append(flatten_obs(obs))
                steps += 1
                if done or steps >= cfg["max_episode_steps"]:
                    break
        tag = "success" if success else "fail"
        save_trajectory(f"{cfg['rollout_dir']}/bc_{tag}_{ep:04d}.npz", obs_list, act_list,
    rew_list, success)
    env.close()


if __name__ == "__main__":
    main()