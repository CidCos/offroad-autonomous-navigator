from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from offroad_autonomous_navigator.envs.offroad_env import OffroadEnv
from offroad_autonomous_navigator.envs.schemas import EnvConfig, RewardConfig
from offroad_autonomous_navigator.utils.config_loader import load_settings_from_yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"


def build_env() -> OffroadEnv:
    """Builds an OffroadEnv instance using the configuration files."""
    env_config = load_settings_from_yaml(EnvConfig, CONFIG_DIR / "env_config.yaml")
    reward_config = load_settings_from_yaml(RewardConfig, CONFIG_DIR / "reward_config.yaml")
    return OffroadEnv(config=env_config, reward_config=reward_config)

if __name__ == "__main__":
    env = build_env()
    check_env(env, warn=True) 
    print("Environment check passed.")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log= None
    )

    model.learn(total_timesteps=20_000)
    print("Training completed.")