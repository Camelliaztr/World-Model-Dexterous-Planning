# src/utils.py
import os
import random
import yaml
import numpy as np
import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_tensor(x, dev=None):
    if dev is None:
        dev = device()
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def save_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)


def moving_average(x, window=10):
    if len(x) < window:
        return np.asarray(x)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(x), kernel, mode="valid")