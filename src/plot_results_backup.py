# src/plot_results.py
import argparse
import os
import torch
import matplotlib.pyplot as plt
from src.utils import moving_average, ensure_dir


def plot_loss(path, title, out_path):
    if not os.path.exists(path):
        print(f"skip missing {path}")
        return
    losses = torch.load(path).cpu().numpy().tolist()
    plt.figure(figsize=(6, 4))
    plt.plot(losses, label="loss")
    if len(losses) >= 10:
        plt.plot(range(9, len(losses)), moving_average(losses, 10), label="ma10")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()
    ensure_dir(args.log_dir)
    plot_loss(os.path.join(args.log_dir, "policy_losses.pt"), "BC Policy Loss",
              os.path.join(args.log_dir, "policy_loss.png"))
    plot_loss(os.path.join(args.log_dir, "world_losses.pt"), "World Model Loss",
              os.path.join(args.log_dir, "world_loss.png"))
    plot_loss(os.path.join(args.log_dir, "value_losses.pt"), "Value Model Loss",
              os.path.join(args.log_dir, "value_loss.png"))
    plot_loss(os.path.join(args.log_dir, "q_losses.pt"), "Q Model Loss",
              os.path.join(args.log_dir, "q_loss.png"))


if __name__ == "__main__":
    main()