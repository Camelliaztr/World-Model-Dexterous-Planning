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
    """
    兼容 checkpoint 中带前缀的权重名：
    model.net.0.weight -> net.0.weight
    module.net.0.weight -> net.0.weight
    """
    cleaned = {}

    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]

        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]

        cleaned[new_k] = v

    return cleaned


def infer_mlp_dims_from_state_dict(state_dict):
    """
    从 MLP 权重自动推断输入维度和输出维度。
    例如：
    net.0.weight: [hidden_dim, obs_dim]
    net.6.bias: [out_dim]
    """
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight")]
    bias_keys = [k for k in state_dict.keys() if k.endswith(".bias")]

    weight_keys = sorted(weight_keys)
    bias_keys = sorted(bias_keys)

    first_weight = state_dict[weight_keys[0]]
    last_bias = state_dict[bias_keys[-1]]

    obs_dim = int(first_weight.shape[1])
    out_dim = int(last_bias.shape[0])

    return obs_dim, out_dim


def load_policy_checkpoint(ckpt_path, dev):
    ckpt = torch.load(ckpt_path, map_location=dev)

    if not isinstance(ckpt, dict):
        raise ValueError("Unsupported checkpoint format: checkpoint is not a dict.")

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)
    obs_dim, out_dim = infer_mlp_dims_from_state_dict(state_dict)

    return state_dict, obs_dim, out_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    parser.add_argument("--out", default="reports/videos/bc_relocate.mp4")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    dev = device()

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

    ckpt_path = Path(cfg["checkpoint_dir"]) / "policy.pt"
    state_dict, policy_obs_dim, policy_out_dim = load_policy_checkpoint(ckpt_path, dev)

    if policy_out_dim % env_act_dim != 0:
        raise ValueError(
            f"Policy output dim {policy_out_dim} is not divisible by env action dim {env_act_dim}."
        )

    horizon = policy_out_dim // env_act_dim

    print("env_obs_dim:", env_obs_dim)
    print("env_act_dim:", env_act_dim)
    print("policy_obs_dim:", policy_obs_dim)
    print("policy_out_dim:", policy_out_dim)
    print("inferred_horizon:", horizon)

    policy = MLP(
        policy_obs_dim,
        policy_out_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
    ).to(dev)

    policy.load_state_dict(state_dict)
    policy.eval()

    frames = []
    total_reward = 0.0
    success = False

    for t in range(args.steps):
        frame = env.render()
        frames.append(frame)

        obs_vec = flatten_obs(obs)

        obs_tensor = torch.as_tensor(
            obs_vec,
            dtype=torch.float32,
            device=dev,
        ).unsqueeze(0)

        with torch.no_grad():
            action_chunk = policy(obs_tensor).cpu().numpy().reshape(horizon, env_act_dim)

        action = action_chunk[0]
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)

        if isinstance(info, dict) and "success" in info:
            success = success or bool(info["success"])

        if terminated or truncated:
            break

    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_path, frames, fps=30)

    print("Saved video to:", out_path)
    print("num frames:", len(frames))
    print("return:", total_reward)
    print("success:", success)


if __name__ == "__main__":
    main()
