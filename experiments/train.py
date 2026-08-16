from pathlib import Path

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env
from wandb.integration.sb3 import WandbCallback

from experiments.callbacks import RewardBreakdownCallback
from offroad_autonomous_navigator.envs.kinematic.offroad_env import OffroadEnv
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

    total_timesteps = 5_000

    run = wandb.init(
        project="offroad-autonomous-navigator",
        config={
            "total_timesteps": total_timesteps,
            "env_config": env.config.model_dump(),
            "reward_config": env.reward_config.model_dump(),
        },
        sync_tensorboard=True,
    )

    # 
    run.define_metric("global_step")
    run.define_metric("reward/*", step_metric="global_step")
    run.define_metric("distance_to_goal", step_metric="global_step")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log= f"runs/{run.id}",
    )

    model.learn(total_timesteps=total_timesteps,
                callback=CallbackList([
                    WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=f"checkpoints/{run.id}", 
                        verbose=2, 
                    ),
                    RewardBreakdownCallback(),
                ]))

    run.finish()
    print("Training completed.")