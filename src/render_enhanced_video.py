import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

    return model, in_dim, out_dim


def draw_text_box(draw, xy, text, font, fill=(255, 255, 255), box_fill=(0, 0, 0, 150), pad=8):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle(
        [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
        radius=8,
        fill=box_fill,
    )
    draw.text((x, y), text, font=font, fill=fill)


def overlay_frame(
    frame,
    title,
    step,
    max_steps,
    total_reward,
    success,
    extra_lines,
):
    img = Image.fromarray(frame).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()
    big_font = ImageFont.load_default()

    # top translucent bar
    draw.rectangle([0, 0, w, 72], fill=(0, 0, 0, 145))
    draw.text((24, 16), title, font=big_font, fill=(255, 255, 255, 255))

    status = f"Step {step:03d}/{max_steps} | Return {total_reward:8.2f} | Success {success}"
    draw.text((24, 42), status, font=font, fill=(230, 230, 230, 255))

    # progress bar
    bar_x = 24
    bar_y = h - 36
    bar_w = w - 48
    bar_h = 12
    ratio = min(max(step / max_steps, 0.0), 1.0)
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=6, fill=(40, 40, 40, 180))
    draw.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * ratio), bar_y + bar_h], radius=6, fill=(220, 220, 220, 230))

    # information boxes
    y = 92
    for line in extra_lines:
        draw_text_box(draw, (24, y), line, font)
        y += 34

    out = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(out)


def make_card(width, height, title, lines):
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((40, 50), title, font=font, fill=(255, 255, 255))

    y = 100
    for line in lines:
        draw.text((40, y), line, font=font, fill=(230, 230, 230))
        y += 28

    return np.asarray(img)


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
    planner_mode,
    dev,
    action_low,
    action_high,
):
    obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=dev).unsqueeze(0)

    with torch.no_grad():
        base_chunk = policy(obs_tensor).cpu().numpy().reshape(horizon, env_act_dim)

    candidates = np.repeat(base_chunk[None, :, :], num_candidates, axis=0)

    if noise_std > 0:
        noise = np.random.randn(num_candidates, horizon, env_act_dim).astype(np.float32)
        candidates = candidates + noise_std * noise

    candidates[0] = base_chunk
    candidates = np.clip(candidates, action_low, action_high)

    cand_flat = candidates.reshape(num_candidates, horizon * env_act_dim).astype(np.float32)
    obs_batch = np.repeat(obs_vec[None, :], num_candidates, axis=0).astype(np.float32)

    obs_batch_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=dev)
    cand_flat_t = torch.as_tensor(cand_flat, dtype=torch.float32, device=dev)

    with torch.no_grad():
        scores = torch.zeros(num_candidates, dtype=torch.float32, device=dev)

        if "w" in planner_mode or "v" in planner_mode:
            world_input = torch.cat([obs_batch_t, cand_flat_t], dim=-1)
            pred_future_obs = world(world_input)

        if "v" in planner_mode:
            scores = scores + value(pred_future_obs).reshape(-1)

        if "q" in planner_mode:
            q_input = torch.cat([obs_batch_t, cand_flat_t], dim=-1)
            scores = scores + q_model(q_input).reshape(-1)

        best_idx = int(torch.argmax(scores).item())
        best_score = float(scores[best_idx].item())

    return candidates[best_idx], best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate_expert1000_wm.yaml")
    parser.add_argument("--mode", choices=["random", "bc", "planning"], required=True)
    parser.add_argument("--planner_mode", default="wv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    dev = device()
    hidden_dim = int(cfg["hidden_dim"])
    num_layers = int(cfg["num_layers"])

    gym.register_envs(gymnasium_robotics)

    env = gym.make(
        cfg["env_id"],
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
    )

    obs, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_vec = flatten_obs(obs)
    env_obs_dim = int(obs_vec.shape[0])
    env_act_dim = int(np.prod(env.action_space.shape))

    checkpoint_dir = Path(cfg["checkpoint_dir"])

    policy = None
    world = None
    value = None
    q_model = None

    horizon = int(cfg["horizon"])
    execute_steps = int(cfg["execute_steps"])
    num_candidates = int(cfg["num_candidates"])
    noise_std = float(cfg["noise_std"])

    if args.mode in ["bc", "planning"]:
        policy, policy_in_dim, policy_out_dim = load_mlp(
            checkpoint_dir / "policy.pt",
            hidden_dim,
            num_layers,
            dev,
        )
        horizon = policy_out_dim // env_act_dim

    if args.mode == "planning":
        world, _, _ = load_mlp(checkpoint_dir / "world.pt", hidden_dim, num_layers, dev)
        value, _, _ = load_mlp(checkpoint_dir / "value.pt", hidden_dim, num_layers, dev)
        q_model, _, _ = load_mlp(checkpoint_dir / "q.pt", hidden_dim, num_layers, dev)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = {
        "random": "Random Policy | AdroitHandRelocate-v1",
        "bc": "BC Policy | Expert1000",
        "planning": "Planning-WV | Expert1000",
    }[args.mode]

    intro_lines = [
        f"Environment: {cfg['env_id']}",
        f"Observation dim: {env_obs_dim}",
        f"Action dim: {env_act_dim}",
        f"Mode: {args.mode}",
        f"Seed: {args.seed}",
    ]

    if args.mode == "planning":
        intro_lines += [
            f"Planner mode: {args.planner_mode}",
            f"Horizon: {horizon}",
            f"Execute steps: {execute_steps}",
            f"Candidates: {num_candidates}",
            f"Noise std: {noise_std}",
        ]

    total_reward = 0.0
    success = False
    planning_scores = []

    with imageio.get_writer(out_path, fps=args.fps, quality=8, macro_block_size=16) as writer:
        intro = make_card(args.width, args.height, title, intro_lines)
        for _ in range(args.fps * 2):
            writer.append_data(intro)

        t = 0

        if args.mode in ["random", "bc"]:
            while t < args.steps:
                frame = env.render()

                if args.mode == "random":
                    action = env.action_space.sample()
                    extra = [
                        "Policy: random action",
                        "Purpose: baseline / failure case",
                    ]
                else:
                    obs_vec = flatten_obs(obs)
                    obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=dev).unsqueeze(0)

                    with torch.no_grad():
                        action_chunk = policy(obs_tensor).cpu().numpy().reshape(horizon, env_act_dim)

                    action = action_chunk[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    extra = [
                        "Policy: Behavior Cloning",
                        f"Action chunk horizon: {horizon}",
                        "Execution: first action of chunk",
                    ]

                frame = overlay_frame(
                    frame,
                    title,
                    t + 1,
                    args.steps,
                    total_reward,
                    success,
                    extra,
                )
                writer.append_data(frame)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)

                if isinstance(info, dict) and "success" in info:
                    success = success or bool(info["success"])

                t += 1

                if terminated or truncated:
                    break

        else:
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
                    planner_mode=args.planner_mode,
                    dev=dev,
                    action_low=env.action_space.low,
                    action_high=env.action_space.high,
                )

                planning_scores.append(planning_score)
                steps_to_execute = min(execute_steps, horizon, args.steps - t)

                for j in range(steps_to_execute):
                    frame = env.render()

                    extra = [
                        f"Planner: {args.planner_mode}",
                        f"Horizon: {horizon} | Execute: {execute_steps}",
                        f"Candidates: {num_candidates} | Noise std: {noise_std}",
                        f"Current planning score: {planning_score:.2f}",
                    ]

                    frame = overlay_frame(
                        frame,
                        title,
                        t + 1,
                        args.steps,
                        total_reward,
                        success,
                        extra,
                    )
                    writer.append_data(frame)

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

        avg_planning_score = float(np.mean(planning_scores)) if planning_scores else 0.0

        outro_lines = [
            f"Frames: {t}",
            f"Return: {total_reward:.2f}",
            f"Success: {success}",
        ]

        if args.mode == "planning":
            outro_lines.append(f"Average planning score: {avg_planning_score:.2f}")

        outro = make_card(args.width, args.height, "Rollout Summary", outro_lines)
        for _ in range(args.fps * 2):
            writer.append_data(outro)

    env.close()

    print("Saved video to:", out_path)
    print("mode:", args.mode)
    print("num frames:", t)
    print("return:", total_reward)
    print("success:", success)

    if args.mode == "planning":
        print("avg_planning_score:", float(np.mean(planning_scores)) if planning_scores else 0.0)


if __name__ == "__main__":
    main()
