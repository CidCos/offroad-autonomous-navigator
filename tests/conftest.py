import pytest

from offroad_autonomous_navigator.envs.offroad_env import OffroadEnv
from offroad_autonomous_navigator.envs.schemas import EnvConfig


@pytest.fixture
def default_config() -> EnvConfig:
    """Global fixture that provides a reusable base configuration."""
    return EnvConfig(
        wheelbase=2.5,
        max_steering_angle=0.5,
        max_acceleration=3.0,
        max_speed=10.0,
        map_min_x=-50.0,
        map_max_x=50.0,
        map_min_y=-50.0,
        map_max_y=50.0,
        goal_x=20.0,
        goal_y=20.0,
        goal_tolerance=1.0,
        dt=0.1,
        max_episode_steps=100,
    )


@pytest.fixture
def default_env(default_config: EnvConfig) -> OffroadEnv:
    """Global fixture that provides a ready-to-use OffroadEnv instance."""
    return OffroadEnv(config=default_config)