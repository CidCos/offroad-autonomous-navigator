
import math
from typing import Any

import gymnasium as gym
import numpy as np

from offroad_autonomous_navigator.envs.kinematics import BicycleModel
from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleAction, VehicleState
from offroad_autonomous_navigator.utils.geometry import normalize_angle


class OffroadEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, config: EnvConfig) -> None:
        super().__init__()
        self.config = config
        self.vehicle = BicycleModel(
            wheelbase=config.wheelbase,
            max_steering_angle=config.max_steering_angle,
            max_acceleration=config.max_acceleration,
            max_speed=config.max_speed,
        )
        self._state: VehicleState | None = None
        self.current_step = 0

        # Action Space: [steering_angle, acceleration]
        self.action_space = gym.spaces.Box(
            low=np.array([-config.max_steering_angle, -config.max_acceleration], dtype=np.float32),
            high=np.array([config.max_steering_angle, config.max_acceleration], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation Space: [v, theta, distance_to_goal, relative_angle_to_goal]
        max_dist = math.hypot(config.map_max_x - config.map_min_x, 
                            config.map_max_y - config.map_min_y)
        self.observation_space = gym.spaces.Box(
            low=np.array([-config.max_speed, -math.pi, 0, -math.pi], dtype=np.float32),
            high=np.array([config.max_speed, math.pi, max_dist, math.pi], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int|None = None, 
            options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        # Reset vehicle state to the origin with zero velocity and orientation
        self._state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)

        return self._state_to_obs(self._state), {}

    def _state_to_obs(self, state: VehicleState) -> np.ndarray:
        '''Convert the VehicleState to an observation array for the RL agent.'''
        distance_to_goal = math.hypot(
            self.config.goal_x - state.x,
            self.config.goal_y - state.y
        )
        raw_angle = (
            math.atan2(self.config.goal_y - state.y,
                        self.config.goal_x - state.x)
            - state.theta
            )
        relative_angle_to_goal = normalize_angle(raw_angle)

        return np.array([state.v, state.theta, distance_to_goal, relative_angle_to_goal]
                        , dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        '''Update the environment state based on the action taken by the agent.'''
        if self._state is None:
            raise RuntimeError("Cannot call step() before reset().")

        vehicle_action = VehicleAction(
            steering_angle=float(action[0]), 
            acceleration=float(action[1])
            )

        new_state = self.vehicle.step(self._state, vehicle_action, self.config.dt)
        self._state = new_state
        self.current_step += 1

        reward = 0.0 #TODO: Implement a reward function in phase 2

        truncated = self.current_step >= self.config.max_episode_steps
        terminated = self._is_out_of_bounds() or self._is_goal_reached()

        return self._state_to_obs(new_state), reward, terminated, truncated, {}

    def _is_out_of_bounds(self) -> bool:
        if self._state is None:
            raise RuntimeError("Cannot check bounds before reset().")
        return (
            self._state.x < self.config.map_min_x or
            self._state.x > self.config.map_max_x or
            self._state.y < self.config.map_min_y or
            self._state.y > self.config.map_max_y
        )

    def _is_goal_reached(self) -> bool:
        if self._state is None:
            raise RuntimeError("Cannot check goal before reset().")
        distance = math.hypot(self._state.x - self.config.goal_x, 
                            self._state.y - self.config.goal_y)
        return distance <= self.config.goal_tolerance


