
import math
from typing import Any

import gymnasium as gym
import numpy as np

from offroad_autonomous_navigator.envs.kinematic.kinematics import BicycleModel
from offroad_autonomous_navigator.envs.kinematic.observation import (
    scale_action,
    state_to_observation,
)
from offroad_autonomous_navigator.envs.kinematic.reward_fn import compute_reward
from offroad_autonomous_navigator.envs.schemas import (
    EnvConfig,
    RewardConfig,
    VehicleState,
)
from offroad_autonomous_navigator.utils.geometry import euclidean_distance


class OffroadEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, config: EnvConfig, reward_config: RewardConfig) -> None:
        super().__init__()
        self.config = config
        self.reward_config = reward_config
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
                                        low=-1.0, 
                                        high=1.0,
                                        shape=(2,), 
                                        dtype=np.float32
                                        )

        # Observation Space: [v, theta, distance_to_goal, relative_angle_to_goal]
        max_dist = euclidean_distance(config.map_min_x, config.map_min_y, 
                                    config.map_max_x, config.map_max_y)
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
        return state_to_observation(state, self.config)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        '''Update the environment state based on the action taken by the agent.'''

        if self._state is None:
            raise RuntimeError("Cannot call step() before reset().")

        # Action and state update
        prev_state = self._state
        vehicle_action = scale_action(action, self.config)
        new_state = self.vehicle.step(self._state, vehicle_action, self.config.dt)
        self._state = new_state
        # Step count and truncation check
        self.current_step += 1
        truncated = self.current_step >= self.config.max_episode_steps

        # Check for termination conditions: out of bounds or goal reached
        is_goal_reached = self._is_goal_reached()
        is_out_of_bounds = self._is_out_of_bounds()
        terminated = is_out_of_bounds or is_goal_reached

        # Compute reward
        reward, reward_breakdown = compute_reward(
            state=prev_state,
            new_state=new_state,
            action=vehicle_action,
            config=self.config,
            reward_config=self.reward_config,
            is_goal_reached=is_goal_reached,
            is_out_of_bounds=is_out_of_bounds
        )

        distance_to_goal = euclidean_distance(new_state.x, new_state.y, 
                                            self.config.goal_x, self.config.goal_y)

        return (self._state_to_obs(new_state), 
                reward,
                terminated,
                truncated, 
                {"reward_breakdown": reward_breakdown,
                "distance_to_goal": distance_to_goal,
                "is_goal_reached": is_goal_reached,
                "is_out_of_bounds": is_out_of_bounds,
                "current_step": self.current_step}
                )

    def _is_out_of_bounds(self) -> bool:
        '''Check if the vehicle is out of the defined map boundaries.'''
        if self._state is None:
            raise RuntimeError("Cannot check bounds before reset().")
        return (
            self._state.x < self.config.map_min_x or
            self._state.x > self.config.map_max_x or
            self._state.y < self.config.map_min_y or
            self._state.y > self.config.map_max_y
        )

    def _is_goal_reached(self) -> bool:
        '''Check if the vehicle is within the goal tolerance.'''
        if self._state is None:
            raise RuntimeError("Cannot check goal before reset().")
        distance = euclidean_distance(self._state.x, self._state.y, 
                                    self.config.goal_x, self.config.goal_y)
        return distance <= self.config.goal_tolerance


