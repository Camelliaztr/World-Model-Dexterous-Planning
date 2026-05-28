# src/train_value_q.py
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


def train_model(model, dataloader, epochs, lr, dev, loss_type="mse"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in trange(epochs, desc="train score model"):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(dev), y.to(dev)
            pred = model(x)
            if loss_type == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, y)
            else:
                loss = F.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
    return losses


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
    roots = [cfg["demo_dir"], cfg["rollout_dir"]]

    value_ds = ChunkDataset(roots, cfg["horizon"], mode="value")
    value_dl = DataLoader(value_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    value = MLP(odim, 1, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    value_losses = train_model(value, value_dl, cfg["value_epochs"], cfg["lr"], dev,
    loss_type="mse")

    q_ds = ChunkDataset(roots, cfg["horizon"], mode="q")
    q_dl = DataLoader(q_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    q_in_dim = odim + adim * cfg["horizon"]
    q = MLP(q_in_dim, 1, cfg["hidden_dim"], cfg["num_layers"]).to(dev)
    q_losses = train_model(q, q_dl, cfg["value_epochs"], cfg["lr"], dev, loss_type="mse")

    ensure_dir(cfg["checkpoint_dir"])
    torch.save({"model": value.state_dict(), "in_dim": odim},
               os.path.join(cfg["checkpoint_dir"], "value.pt"))
    torch.save({"model": q.state_dict(), "in_dim": q_in_dim},
               os.path.join(cfg["checkpoint_dir"], "q.pt"))
    ensure_dir(cfg["log_dir"])
    torch.save(torch.tensor(value_losses), os.path.join(cfg["log_dir"], "value_losses.pt"))
    torch.save(torch.tensor(q_losses), os.path.join(cfg["log_dir"], "q_losses.pt"))


if __name__ == "__main__":
    main()