# src/train_policy.py
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from src.utils import load_config, set_seed, device, ensure_dir
from src.datasets.chunk_dataset import ChunkDataset
from src.envs.adroit_env import make_env, obs_dim, action_dim
from src.models.mlp import GaussianBCPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dev = device()

    env = make_env(cfg["env_id"], cfg["seed"])
    odim, adim = obs_dim(env), action_dim(env)
    env.close()

    ds = ChunkDataset(cfg["demo_dir"], cfg["horizon"], mode="policy")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    policy = GaussianBCPolicy(odim, adim, cfg["horizon"], cfg["hidden_dim"],
    cfg["num_layers"]).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    losses = []
    for epoch in trange(cfg["policy_epochs"], desc="train policy"):
        epoch_loss = 0.0
        for s, a_demo in dl:
            s, a_demo = s.to(dev), a_demo.to(dev)
            a_pred = policy(s)
            loss = F.mse_loss(a_pred, a_demo)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dl))

    ensure_dir(cfg["checkpoint_dir"])
    torch.save({"model": policy.state_dict(), "obs_dim": odim, "action_dim": adim},
               os.path.join(cfg["checkpoint_dir"], "policy.pt"))
    torch.save(torch.tensor(losses), os.path.join(cfg["log_dir"], "policy_losses.pt"))


if __name__ == "__main__":
    main()