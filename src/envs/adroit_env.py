# src/envs/adroit_env.py
import gymnasium as gym
import gymnasium_robotics
import numpy as np


def register_robotics_envs():
    """
    注册 Gymnasium-Robotics 提供的机器人环境。
    如果不注册，gym.make("AdroitHandRelocate-v1") 会找不到环境。
    """
    gym.register_envs(gymnasium_robotics)


def make_env(env_id: str, seed: int = 0):
    """
    创建 Adroit / Gymnasium-Robotics 环境。
    """
    register_robotics_envs()

    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


def flatten_obs(obs):
    """
    把 observation 转成一维 float32 向量。
    兼容 ndarray observation 和 dict observation。
    """
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                parts.append(value.reshape(-1))
        return np.concatenate(parts, axis=0).astype(np.float32)

    return np.asarray(obs, dtype=np.float32).reshape(-1)


def get_success(info, reward=None, threshold=0.0):
    """
    从 info 或 reward 中判断任务是否成功。
    不同环境的 success 字段可能不同，因此做兼容处理。
    """
    if isinstance(info, dict):
        for key in ["success", "is_success", "goal_achieved"]:
            if key in info:
                return bool(info[key])

    if reward is not None:
        return bool(reward > threshold)

    return False


def obs_dim(env):
    obs, _ = env.reset()
    return int(flatten_obs(obs).shape[0])


def action_dim(env):
    return int(np.prod(env.action_space.shape))