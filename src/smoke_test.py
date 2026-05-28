import argparse

from src.utils import load_config, set_seed
from src.envs.adroit_env import make_env, flatten_obs, obs_dim, action_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adroit_relocate.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    env_id = cfg["env_id"]
    seed = cfg["seed"]

    print("=" * 60)
    print("Smoke Test")
    print("=" * 60)
    print("env_id:", env_id)
    print("seed:", seed)

    env = make_env(env_id, seed=seed)

    obs, info = env.reset(seed=seed)
    obs_vec = flatten_obs(obs)

    print("Environment created successfully.")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Flatten obs shape:", obs_vec.shape)
    print("obs_dim:", obs_dim(env))
    print("action_dim:", action_dim(env))

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    next_obs_vec = flatten_obs(next_obs)

    print("-" * 60)
    print("One-step interaction successful.")
    print("Action shape:", action.shape)
    print("Next obs shape:", next_obs_vec.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info keys:", list(info.keys()) if isinstance(info, dict) else type(info))

    env.close()

    print("=" * 60)
    print("Smoke test passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()