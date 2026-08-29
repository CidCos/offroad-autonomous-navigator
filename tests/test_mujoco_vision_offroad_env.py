from typing import Any

import numpy as np
import pytest

from offroad_autonomous_navigator.envs.mujoco.mujoco_vision_offroad_env import (
    MujocoOffroadEnvVision,
)
from offroad_autonomous_navigator.envs.schemas import EnvConfig, RewardConfig


@pytest.fixture
def vision_env(
    default_config: EnvConfig, 
    default_reward_config: RewardConfig
    ) -> MujocoOffroadEnvVision:
    """Provides a ready-to-use MujocoOffroadEnvVision instance."""
    return MujocoOffroadEnvVision(config=default_config, reward_config=default_reward_config)


# 1
def test_reset_returns_valid_observation(vision_env: MujocoOffroadEnvVision) -> None:
    """Verifies that reset() returns a Dict observation contained in observation_space,
    with the expected keys and shapes."""
    obs, info = vision_env.reset()

    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"depth", "vector"}
    assert vision_env.observation_space.contains(obs)
    assert obs["depth"].shape == (64, 64)
    assert obs["depth"].dtype == np.float32
    assert obs["vector"].shape == (4,)
    assert obs["vector"].dtype == np.float32
    assert isinstance(info, dict)


# 2
def test_depth_values_within_normalized_range(vision_env: MujocoOffroadEnvVision) -> None:
    """Verifies that the depth image is normalized to [0, 1] as designed."""
    obs, _ = vision_env.reset()

    assert obs["depth"].min() >= 0.0
    assert obs["depth"].max() <= 1.0


# 3
def test_step_before_reset_raises_runtime_error(vision_env: MujocoOffroadEnvVision) -> None:
    """Verifies that calling step() before reset() raises RuntimeError."""
    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Cannot call step\\(\\) before reset\\(\\)."):
        vision_env.step(zero_action)


# 4
def test_normal_step_continues_episode(vision_env: MujocoOffroadEnvVision) -> None:
    """Verifies that a normal step within bounds keeps the episode active and
    returns a valid Dict observation."""
    vision_env.reset()
    normal_action = np.array([0.1, 1.0], dtype=np.float32)

    obs, reward, terminated, truncated, info = vision_env.step(normal_action)

    assert not terminated
    assert not truncated
    assert vision_env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    assert info["is_goal_reached"] is False
    assert info["is_out_of_bounds"] is False


# 5
def test_step_returns_correct_info(vision_env: MujocoOffroadEnvVision) -> None:
    """Verifies that the info dictionary contains expected keys after a step."""
    vision_env.reset()
    normal_action = np.array([0.1, 1.0], dtype=np.float32)

    _, _, _, _, info = vision_env.step(normal_action)

    expected_keys = {
        "reward_breakdown",
        "distance_to_goal",
        "is_goal_reached",
        "is_out_of_bounds",
        "current_step",
    }
    assert expected_keys.issubset(info.keys())
    assert info["current_step"] == 1
    assert isinstance(info["distance_to_goal"], float)


# 6
def test_truncation_on_max_episode_steps(
    default_config: EnvConfig, default_reward_config: RewardConfig
) -> None:
    """Verifies that reaching max_episode_steps returns truncated=True and terminated=False."""
    config = default_config.model_copy(update={"max_episode_steps": 3})
    env = MujocoOffroadEnvVision(config=config, reward_config=default_reward_config)
    env.reset()
    zero_action = np.array([0.0, 0.0], dtype=np.float32)

    _, _, terminated_1, truncated_1, info_1 = env.step(zero_action)
    _, _, terminated_2, truncated_2, info_2 = env.step(zero_action)
    assert not terminated_1 and not truncated_1
    assert info_1["current_step"] == 1
    assert not terminated_2 and not truncated_2
    assert info_2["current_step"] == 2

    _, _, terminated_3, truncated_3, info_3 = env.step(zero_action)
    assert not terminated_3
    assert info_3["current_step"] == 3
    assert truncated_3


# 7
def test_out_of_bounds_produces_terminated(
    default_config: EnvConfig, default_reward_config: RewardConfig
) -> None:
    """Verifies that leaving the map boundaries eventually terminates the episode."""
    config = default_config.model_copy(
        update={
            "map_min_x": -0.5,
            "map_max_x": 0.5,
            "map_min_y": -0.5,
            "map_max_y": 0.5,
            "max_episode_steps": 100,
        }
    )
    env = MujocoOffroadEnvVision(config=config, reward_config=default_reward_config)
    env.reset()

    max_accel_action = np.array([0.0, 1.0], dtype=np.float32)

    terminated = False
    info: dict[str, Any] = {}
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(max_accel_action)
        if terminated:
            break

    assert terminated
    assert info["is_out_of_bounds"]
    assert not info["is_goal_reached"]


# 8
def test_goal_reached(default_config: EnvConfig, default_reward_config: RewardConfig) -> None:
    """Verifies that being within goal_tolerance produces terminated=True."""
    config = default_config.model_copy(
        update={
            "goal_x": 0.0,
            "goal_y": 0.0,
            "goal_tolerance": 5.0,
        }
    )
    env = MujocoOffroadEnvVision(config=config, reward_config=default_reward_config)
    env.reset()

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, terminated, truncated, info = env.step(zero_action)

    assert terminated
    assert not truncated
    assert info["is_goal_reached"]
    assert not info["is_out_of_bounds"]