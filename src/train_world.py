# src/train_world.py
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from src.utils import load_config, set_seed, device, ensure_dir
from src.datasets.chunk_dataset import ChunkDataset
from src.envs.adroit_env import make_env, obs_dim, action_dim
from src.models.mlp import MLP


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
    in_dim = odim + adim * cfg["horizon"]
    out_dim = odim

    # 使用 demos + BC rollouts；如果第 7 天还没有 rollouts，可以先只放 demo_dir
    roots = [cfg["demo_dir"], cfg["rollout_dir"]]
    ds = ChunkDataset(roots, cfg["horizon"], mode="world")
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    world = MLP(in_dim, out_dim, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    opt = torch.optim.Adam(world.parameters(), lr=cfg["lr"])

    losses = []
    for epoch in trange(cfg["world_epochs"], desc="train world"):
        epoch_loss = 0.0
        for x, s_future in dl:
            x, s_future = x.to(dev), s_future.to(dev)
            pred = world(x)
            loss = F.mse_loss(pred, s_future)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dl))

    ensure_dir(cfg["checkpoint_dir"])
    torch.save({"model": world.state_dict(), "in_dim": in_dim, "out_dim": out_dim},
               os.path.join(cfg["checkpoint_dir"], "world.pt"))
    torch.save(torch.tensor(losses), os.path.join(cfg["log_dir"], "world_losses.pt"))


if __name__ == "__main__":
    main()