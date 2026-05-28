# src/evaluate_bc.py
import argparse
import os
import numpy as np
import torch
from tqdm import trange
from src.utils import load_config, set_seed, device
from src.envs.adroit_env import make_env, flatten_obs, get_success, obs_dim, action_dim
from src.models.mlp import GaussianBCPolicy


def load_policy(cfg, dev):
    env = make_env(cfg["env_id"], cfg["seed"])
    odim, adim = obs_dim(env), action_dim(env)
    env.close()
    policy = GaussianBCPolicy(odim, adim, cfg["horizon"], cfg["hidden_dim"],
    cfg["num_layers"]).to(dev)
    ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "policy.pt"), map_location=dev)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, adim


def run_episode(env, policy, cfg, dev, seed=0):
    obs, _ = env.reset(seed=seed)
    done, total_reward, success, steps = False, 0.0, False, 0
    while not done and steps < cfg["max_episode_steps"]:
        s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=dev)
        with torch.no_grad():
            chunk = policy(s.unsqueeze(0)).view(cfg["horizon"], -1).cpu().numpy()
        for i in range(min(cfg["execute_steps"], len(chunk))):
            obs, reward, terminated, truncated, info = env.step(chunk[i])
            done = terminated or truncated
            total_reward += reward
            success = success or get_success(info, reward, cfg["success_reward_threshold"])
            steps += 1
            if done or steps >= cfg["max_episode_steps"]:
                break
    return total_reward, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dev = device()
    policy, _ = load_policy(cfg, dev)
    env = make_env(cfg["env_id"], cfg["seed"])

    returns, successes = [], []
    for ep in trange(args.episodes):
        ret, suc = run_episode(env, policy, cfg, dev, cfg["seed"] + 1000 + ep)
        returns.append(ret); successes.append(float(suc))
    print({"success_rate": float(np.mean(successes)), "avg_return": float(np.mean(returns))})
    env.close()


if __name__ == "__main__":
    main()