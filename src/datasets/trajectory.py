# src/datasets/trajectory.py
from pathlib import Path
import numpy as np


def save_trajectory(path, obs, actions, rewards, success):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=np.asarray(obs, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        success=np.asarray(success, dtype=np.float32),
    )


def load_trajectory(path):
    data = np.load(path)
    return {
        "obs": data["obs"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "success": float(data["success"]),
    }


def list_npz_files(root):
    return sorted(str(p) for p in Path(root).glob("*.npz"))