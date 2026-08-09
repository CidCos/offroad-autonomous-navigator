import numpy as np
import pytest

from offroad_autonomous_navigator.envs.offroad_env import OffroadEnv
from offroad_autonomous_navigator.envs.schemas import EnvConfig, RewardConfig


def test_reset_returns_valid_observation(default_env: OffroadEnv) -> None:
    """Verifies that reset() returns an observation contained in observation_space."""
    obs, info = default_env.reset()

    assert default_env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_before_reset_raises_runtime_error(default_env: OffroadEnv) -> None:
    """Verifies that calling step() before reset() raises RuntimeError."""
    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Cannot call step\\(\\) before reset\\(\\)."):
        default_env.step(zero_action)


def test_truncation_on_max_episode_steps(default_config: EnvConfig, 
                                        default_reward_config: RewardConfig) -> None:
    """Verifies that reaching max_episode_steps returns truncated=True and terminated=False."""
    config = default_config.model_copy(update={"max_episode_steps": 3})
    env = OffroadEnv(config=config, reward_config=default_reward_config)
    env.reset()
    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    # Steps 1 and 2
    _, _, terminated_1, truncated_1, _ = env.step(zero_action)
    _, _, terminated_2, truncated_2, _ = env.step(zero_action)
    assert not terminated_1 and not truncated_1
    assert not terminated_2 and not truncated_2

    # Step 3: limit reached
    _, _, terminated_3, truncated_3, _ = env.step(zero_action)
    assert not terminated_3
    assert truncated_3


def test_out_of_bounds_produces_terminated(default_config: EnvConfig, 
                                        default_reward_config: RewardConfig) -> None:
    """Verifies that leaving the map boundaries terminates the episode,
    regardless of orientation."""
    # Extremely tight map bounds in X and Y to force an immediate exit in any direction
    config = default_config.model_copy(
        update={
            "map_min_x": -0.01,
            "map_max_x": 0.01,
            "map_min_y": -0.01,
            "map_max_y": 0.01,
            "dt": 0.1,
        }
    )
    env = OffroadEnv(config=config, reward_config=default_reward_config)
    env.reset()

    # Forward acceleration; it will leave in 1 step regardless of theta
    max_accel_action = np.array([0.0, config.max_acceleration], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(max_accel_action)

    assert terminated
    assert not truncated


def test_goal_reached_produces_terminated(default_config: EnvConfig, 
                                        default_reward_config: RewardConfig) -> None:
    """Verifies that being within goal_tolerance produces terminated=True."""
    config = default_config.model_copy(
        update={
            "goal_x": 0.0,
            "goal_y": 0.0,
            "goal_tolerance": 5.0,
        }
    )
    env = OffroadEnv(config=config, reward_config=default_reward_config)
    env.reset()

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, terminated, truncated, _ = env.step(zero_action)

    assert terminated
    assert not truncated

def test_normal_step_continues_episode(default_env: OffroadEnv) -> None:
    """Verifies that a normal step within bounds keeps the episode active."""
    default_env.reset()
    normal_action = np.array([0.1, 1.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = default_env.step(normal_action)

    assert not terminated
    assert not truncated
    assert default_env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(info, dict)