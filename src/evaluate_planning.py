# src/evaluate_planning.py
import argparse
import numpy as np
import torch
from tqdm import trange
from src.utils import load_config, set_seed, device
from src.envs.adroit_env import make_env, flatten_obs, get_success, obs_dim, action_dim
from src.planner import load_all_models, plan_with_q, plan_with_world_value, plan_random


def run_episode(env, models, cfg, dev, mode, seed):
    policy, world, value, q = models
    obs, _ = env.reset(seed=seed)
    done, total_reward, success, steps = False, 0.0, False, 0
    planning_scores = []
    while not done and steps < cfg["max_episode_steps"]:
        s = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=dev)
        if mode == "q":
            chunk, score = plan_with_q(s, policy, q, cfg)
        elif mode == "wv":
            chunk, score = plan_with_world_value(s, policy, world, value, cfg)
        elif mode == "random":
            chunk, score = plan_random(s, policy, cfg)
        else:
            raise ValueError(mode)
        planning_scores.append(score)
        chunk = chunk.cpu().numpy()
        for i in range(min(cfg["execute_steps"], len(chunk))):
            obs, reward, terminated, truncated, info = env.step(chunk[i])
            done = terminated or truncated
            total_reward += reward
            success = success or get_success(info, reward, cfg["success_reward_threshold"])
            steps += 1
            if done or steps >= cfg["max_episode_steps"]:
                break
    return total_reward, success, float(np.mean(planning_scores) if planning_scores else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--mode", choices=["q", "wv", "random"], default="wv")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dev = device()
    env = make_env(cfg["env_id"], cfg["seed"])
    odim, adim = obs_dim(env), action_dim(env)
    models = load_all_models(cfg, odim, adim, dev)

    returns, successes, scores = [], [], []
    for ep in trange(args.episodes, desc=f"eval {args.mode}"):
        ret, suc, score = run_episode(env, models, cfg, dev, args.mode, cfg["seed"] + 3000 + ep)
        returns.append(ret); successes.append(float(suc)); scores.append(score)
    result = {
        "mode": args.mode,
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "avg_planning_score": float(np.mean(scores)),
    }
    print(result)
    env.close()


if __name__ == "__main__":
    main()