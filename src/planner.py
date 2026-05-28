# src/planner.py
import os
import torch
from src.models.mlp import MLP, GaussianBCPolicy


def load_all_models(cfg, obs_dim, action_dim, dev):
    policy = GaussianBCPolicy(obs_dim, action_dim, cfg["horizon"], cfg["hidden_dim"],
    cfg["num_layers"]).to(dev)
    policy_ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "policy.pt"), map_location=dev)
    policy.load_state_dict(policy_ckpt["model"])
    policy.eval()

    world_in = obs_dim + action_dim * cfg["horizon"]
    world = MLP(world_in, obs_dim, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    world_ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "world.pt"), map_location=dev)
    world.load_state_dict(world_ckpt["model"])
    world.eval()

    value = MLP(obs_dim, 1, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    value_ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "value.pt"), map_location=dev)
    value.load_state_dict(value_ckpt["model"])
    value.eval()

    q = MLP(world_in, 1, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    q_ckpt = torch.load(os.path.join(cfg["checkpoint_dir"], "q.pt"), map_location=dev)
    q.load_state_dict(q_ckpt["model"])
    q.eval()
    return policy, world, value, q


@torch.no_grad()
def plan_with_q(s, policy, q_model, cfg):
    candidates = policy.sample_chunks(s, cfg["num_candidates"], cfg["noise_std"])
    s_rep = s.unsqueeze(0).repeat(cfg["num_candidates"], 1)
    q_in = torch.cat([s_rep, candidates], dim=-1)
    scores = q_model(q_in).squeeze(-1)
    best = torch.argmax(scores).item()
    return candidates[best].view(cfg["horizon"], -1), scores[best].item()


@torch.no_grad()
def plan_with_world_value(s, policy, world_model, value_model, cfg):
    candidates = policy.sample_chunks(s, cfg["num_candidates"], cfg["noise_std"])
    s_rep = s.unsqueeze(0).repeat(cfg["num_candidates"], 1)
    world_in = torch.cat([s_rep, candidates], dim=-1)
    s_future = world_model(world_in)
    scores = value_model(s_future).squeeze(-1)
    best = torch.argmax(scores).item()
    return candidates[best].view(cfg["horizon"], -1), scores[best].item()


@torch.no_grad()
def plan_random(s, policy, cfg):
    candidates = policy.sample_chunks(s, cfg["num_candidates"], cfg["noise_std"])
    idx = torch.randint(0, cfg["num_candidates"], (1,)).item()
    return candidates[idx].view(cfg["horizon"], -1), 0.0