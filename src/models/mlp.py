# src/models/mlp.py
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianBCPolicy(nn.Module):
    """第一版 deterministic MLP + 采样噪声。
    train 时输出 action chunk；eval/planning 时可加噪声得到 N 个候选。
    """
    def __init__(self, obs_dim, action_dim, horizon, hidden_dim=256, num_layers=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.model = MLP(obs_dim, action_dim * horizon, hidden_dim, num_layers)

    def forward(self, obs):
        return self.model(obs)

    @torch.no_grad()
    def sample_chunks(self, obs, num_candidates=32, noise_std=0.08):
        """obs: [obs_dim] or [1, obs_dim]，返回 [N, H*action_dim]。"""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        mean = self.forward(obs).repeat(num_candidates, 1)
        noise = torch.randn_like(mean) * noise_std
        return torch.clamp(mean + noise, -1.0, 1.0)