import math

import numpy as np
import pytest

from offroad_autonomous_navigator.envs.kinematic.observation import (
    scale_action,
    state_to_observation,
)
from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleState
from offroad_autonomous_navigator.utils.geometry import euclidean_distance, normalize_angle


# 0. Check that the observation array has the correct shape and type
def test_state_to_observation_shape_and_type(default_config: EnvConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    obs = state_to_observation(state, default_config)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4,)

# 1. Check speed and orientation
def test_state_to_observation_speed_and_orientation(default_config: EnvConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=math.pi/4, v=5.0)
    obs = state_to_observation(state, default_config)
    assert obs[0] == pytest.approx(5.0)  # Speed
    assert obs[1] == pytest.approx(math.pi/4)  # Orientation

# 2. Check distance to goal
def test_state_to_observation_distance_to_goal(default_config: EnvConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    obs = state_to_observation(state, default_config)
    expected_distance = euclidean_distance(0.0, 0.0, default_config.goal_x, default_config.goal_y)
    assert obs[2] == pytest.approx(expected_distance)

# 3. Check relative angle to goal
def test_state_to_observation_relative_angle_to_goal(default_config: EnvConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    obs = state_to_observation(state, default_config)
    expected_angle = math.atan2(default_config.goal_y - 0.0, default_config.goal_x - 0.0) - 0.0
    expected_angle = normalize_angle(expected_angle)
    assert obs[3] == pytest.approx(expected_angle)

# 4. Check negative coordinates
def test_state_to_observation_negative_coordinates(default_config: EnvConfig) -> None:
    state = VehicleState(x=-10.0, y=-10.0, theta=math.pi/2, v=3.0)
    obs = state_to_observation(state, default_config)
    expected_distance = euclidean_distance(-10.0, -10.0, 
                                        default_config.goal_x, default_config.goal_y)
    expected_angle = math.atan2(default_config.goal_y - (-10.0), 
                                default_config.goal_x - (-10.0)) - math.pi/2
    expected_angle = normalize_angle(expected_angle)
    
    assert obs[0] == pytest.approx(3.0)  # Speed
    assert obs[1] == pytest.approx(math.pi/2)  # Orientation
    assert obs[2] == pytest.approx(expected_distance)  # Distance to goal
    assert obs[3] == pytest.approx(expected_angle)  # Relative angle to goal

# 5. Check observation when at goal
def test_state_to_observation_at_goal() -> None:
    state = VehicleState(x=20.0, y=20.0, theta=0.0, v=0.0)
    config = EnvConfig(
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
    obs = state_to_observation(state, config)
    assert obs[2] == pytest.approx(0.0)  # Distance to goal

# 6. Check scale_action function
def test_scale_action(default_config: EnvConfig) -> None:
    normalized_action = np.array([0.5, -0.5], dtype=np.float32)
    vehicle_action = scale_action(normalized_action, default_config)
    
    expected_steering_angle = 0.5 * default_config.max_steering_angle
    expected_acceleration = -0.5 * default_config.max_acceleration
    
    assert vehicle_action.steering_angle == pytest.approx(expected_steering_angle)
    assert vehicle_action.acceleration == pytest.approx(expected_acceleration)