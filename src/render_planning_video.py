import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio
import numpy as np
import torch

from src.utils import load_config, set_seed, device
from src.envs.adroit_env import flatten_obs
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


def infer_mlp_dims_from_state_dict(state_dict):
    weight_keys = sorted([k for k in state_dict.keys() if k.endswith(".weight")])
    bias_keys = sorted([k for k in state_dict.keys() if k.endswith(".bias")])

    first_weight = state_dict[weight_keys[0]]
    last_bias = state_dict[bias_keys[-1]]

    in_dim = int(first_weight.shape[1])
    out_dim = int(last_bias.shape[0])

    return in_dim, out_dim


def load_mlp_checkpoint(path, dev, hidden_dim, num_layers):
    ckpt = torch.load(path, map_location=dev)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)
    in_dim, out_dim = infer_mlp_dims_from_state_dict(state_dict)

    model = MLP(
        in_dim,
        out_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)

    model.load_state_dict(state_dict)
    model.eval()

    return model, in_dim, out_dim


def plan_action_chunk(
    obs_vec,
    policy,
    world,
    value,
    q_model,
    env_act_dim,
    horizon,
    num_candidates,
    noise_std,
    mode,
    dev,
    action_low,
    action_high,
):
    obs_tensor = torch.as_tensor(
        obs_vec,
        dtype=torch.float32,
        device=dev,
    ).unsqueeze(0)

    with torch.no_grad():
        base_chunk = policy(obs_tensor).cpu().numpy().reshape(horizon, env_act_dim)

    candidates = np.repeat(base_chunk[None, :, :], num_candidates, axis=0)

    if noise_std > 0:
        noise = np.random.randn(num_candidates, horizon, env_act_dim).astype(np.float32)
        candidates = candidates + noise_std * noise

    # 保留第 0 个候选为原始 BC action chunk，避免 planning 完全偏离 BC
    candidates[0] = base_chunk

    candidates = np.clip(candidates, action_low, action_high)

    cand_flat = candidates.reshape(num_candidates, horizon * env_act_dim).astype(np.float32)

    obs_batch = np.repeat(obs_vec[None, :], num_candidates, axis=0).astype(np.float32)

    obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=dev)
    cand_flat_t = torch.as_tensor(cand_flat, dtype=torch.float32, device=dev)

    with torch.no_grad():
        scores = torch.zeros(num_candidates, dtype=torch.float32, device=dev)

        if "w" in mode or "v" in mode:
            world_input = torch.cat([obs_batch_t, cand_flat_t], dim=-1)
            pred_future_obs = world(world_input)

        if "v" in mode:
            v_score = value(pred_future_obs).reshape(-1)
            scores = scores + v_score

        if "q" in mode:
            q_input = torch.cat([obs_batch_t, cand_flat_t], dim=-1)
            q_score = q_model(q_input).reshape(-1)
            scores = scores + q_score

        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())

    return candidates[best_idx], best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--out", default="reports/videos/planning_wv_relocate.mp4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="wv", choices=["v", "q", "wv", "wq", "wvq"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    dev = device()
    hidden_dim = int(cfg["hidden_dim"])
    num_layers = int(cfg["num_layers"])

    horizon = int(cfg["horizon"])
    execute_steps = int(cfg["execute_steps"])
    num_candidates = int(cfg["num_candidates"])
    noise_std = float(cfg["noise_std"])

    gym.register_envs(gymnasium_robotics)

    env = gym.make(
        cfg["env_id"],
        render_mode="rgb_array",
        width=640,
        height=480,
    )

    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_vec = flatten_obs(obs)
    env_obs_dim = int(obs_vec.shape[0])
    env_act_dim = int(np.prod(env.action_space.shape))

    checkpoint_dir = Path(cfg["checkpoint_dir"])

    policy, policy_in_dim, policy_out_dim = load_mlp_checkpoint(
        checkpoint_dir / "policy.pt",
        dev,
        hidden_dim,
        num_layers,
    )

    world, world_in_dim, world_out_dim = load_mlp_checkpoint(
        checkpoint_dir / "world.pt",
        dev,
        hidden_dim,
        num_layers,
    )

    value, value_in_dim, value_out_dim = load_mlp_checkpoint(
        checkpoint_dir / "value.pt",
        dev,
        hidden_dim,
        num_layers,
    )

    q_model, q_in_dim, q_out_dim = load_mlp_checkpoint(
        checkpoint_dir / "q.pt",
        dev,
        hidden_dim,
        num_layers,
    )

    inferred_horizon = policy_out_dim // env_act_dim

    if inferred_horizon != horizon:
        print(f"Warning: config horizon={horizon}, inferred policy horizon={inferred_horizon}")
        horizon = inferred_horizon

    print("=" * 80)
    print("Planning video config")
    print("=" * 80)
    print("mode:", args.mode)
    print("env_obs_dim:", env_obs_dim)
    print("env_act_dim:", env_act_dim)
    print("horizon:", horizon)
    print("execute_steps:", execute_steps)
    print("num_candidates:", num_candidates)
    print("noise_std:", noise_std)
    print("policy dims:", policy_in_dim, "->", policy_out_dim)
    print("world dims:", world_in_dim, "->", world_out_dim)
    print("value dims:", value_in_dim, "->", value_out_dim)
    print("q dims:", q_in_dim, "->", q_out_dim)
    print("=" * 80)

    frames = []
    total_reward = 0.0
    success = False
    planning_scores = []

    t = 0

    while t < args.steps:
        obs_vec = flatten_obs(obs)

        action_chunk, planning_score = plan_action_chunk(
            obs_vec=obs_vec,
            policy=policy,
            world=world,
            value=value,
            q_model=q_model,
            env_act_dim=env_act_dim,
            horizon=horizon,
            num_candidates=num_candidates,
            noise_std=noise_std,
            mode=args.mode,
            dev=dev,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        )

        planning_scores.append(planning_score)

        steps_to_execute = min(execute_steps, horizon, args.steps - t)

        for j in range(steps_to_execute):
            frame = env.render()
            frames.append(frame)

            action = action_chunk[j]
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)

            if isinstance(info, dict) and "success" in info:
                success = success or bool(info["success"])

            t += 1

            if terminated or truncated or t >= args.steps:
                break

        if terminated or truncated:
            break

    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_path, frames, fps=30)

    avg_planning_score = float(np.mean(planning_scores)) if planning_scores else 0.0

    print("Saved video to:", out_path)
    print("num frames:", len(frames))
    print("return:", total_reward)
    print("success:", success)
    print("avg_planning_score:", avg_planning_score)


if __name__ == "__main__":
    main()
