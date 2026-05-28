# src/datasets/chunk_dataset.py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.datasets.trajectory import load_trajectory


class ChunkDataset(Dataset):
    """从轨迹中构造三类样本：
    1) policy: s_t -> a_{t:t+H-1}
    2) world: (s_t, a_chunk) -> s_{t+H}
    3) value/q: state 或 state+action -> score
    """
    def __init__(self, roots, horizon=10, mode="policy"):
        if isinstance(roots, str):
            roots = [roots]
        self.horizon = horizon
        self.mode = mode
        self.samples = []
        for root in roots:
            for path in sorted(Path(root).glob("*.npz")):
                traj = load_trajectory(path)
                obs = traj["obs"]
                actions = traj["actions"]
                rewards = traj["rewards"]
                success = traj["success"]
                T = len(actions)
                for t in range(0, T - horizon):
                    s_t = obs[t]
                    a_chunk = actions[t:t + horizon].reshape(-1)
                    s_future = obs[t + horizon]
                    # dense score：未来 H 步的折扣回报；若 reward 不稳定可替换成 -distance
                    ret = float(np.sum(rewards[t:t + horizon]))
                    label = max(ret, float(success))
                    self.samples.append((s_t, a_chunk, s_future, label))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {roots}; check data path and horizon")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s, a, sf, y = self.samples[idx]
        if self.mode == "policy":
            x, target = s, a
        elif self.mode == "world":
            x, target = np.concatenate([s, a], axis=0), sf
        elif self.mode == "value":
            x, target = sf, np.asarray([y], dtype=np.float32)
        elif self.mode == "q":
            x, target = np.concatenate([s, a], axis=0), np.asarray([y], dtype=np.float32)
        else:
            raise ValueError(self.mode)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)