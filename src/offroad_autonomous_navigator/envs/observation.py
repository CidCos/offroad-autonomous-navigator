import math

import numpy as np

from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleState
from offroad_autonomous_navigator.utils.geometry import euclidean_distance, normalize_angle


def state_to_observation(state: VehicleState, config: EnvConfig) -> np.ndarray:
    '''Convert the VehicleState to an observation array for the RL agent.'''
    distance_to_goal = euclidean_distance(state.x, state.y,
                                        config.goal_x, config.goal_y)
    raw_angle = (
        math.atan2(config.goal_y - state.y,
                    config.goal_x - state.x)
        - state.theta
        )
    relative_angle_to_goal = normalize_angle(raw_angle)

    return np.array([state.v, state.theta, distance_to_goal, relative_angle_to_goal]
                    , dtype=np.float32)