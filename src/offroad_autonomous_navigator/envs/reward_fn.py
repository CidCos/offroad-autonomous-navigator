

from offroad_autonomous_navigator.envs.schemas import (
    EnvConfig,
    RewardBreakdown,
    RewardConfig,
    VehicleAction,
    VehicleState,
)
from offroad_autonomous_navigator.utils.geometry import euclidean_distance


# Reward function components for the OffroadEnv environment
def penalty_border( 
            state: VehicleState, 
            config: EnvConfig,
            reward_config: RewardConfig
            ) -> float:
    """Compute Border Penalty: Penalize proximity to map bounds"""

    map_min_x, map_max_x = config.map_min_x, config.map_max_x
    map_min_y, map_max_y = config.map_min_y, config.map_max_y
    border_margin = reward_config.border_margin

    distance_to_border = (
        # Left border
        max(0, border_margin - (state.x - map_min_x)),
        # Right border
        max(0, border_margin - (map_max_x - state.x)),
        # Bottom border
        max(0, border_margin - (state.y - map_min_y)),
        # Top border
        max(0, border_margin - (map_max_y - state.y))   
    )

    # Closest border penetration (if any) determines the penalty
    penetration = max(distance_to_border)
    normalized_distance_to_border = penetration / border_margin

    # Quadratic penalty
    border_penalty = reward_config.w_border_penalty * min(1.0, normalized_distance_to_border**3)

    return -border_penalty


def reward_progress(
            state: VehicleState,
            new_state: VehicleState, 
            config: EnvConfig,
            reward_config: RewardConfig
            ) -> float:
    '''Compute Progress Reward: Reward for moving closer to the goal.'''

    # Distance to goal before and after the step
    distance_to_goal_state = euclidean_distance(
                                                state.x, state.y, 
                                                config.goal_x, config.goal_y
                                                )
    distance_to_goal_new_state = euclidean_distance(
                                                    new_state.x, new_state.y, 
                                                    config.goal_x, config.goal_y
                                                    )
    # Positive reward if vehicle is closer to the goal
    progress = distance_to_goal_state - distance_to_goal_new_state  # Positive if moving closer
    progress_reward = reward_config.w_progress * progress
    return progress_reward

def penalty_energy(action: VehicleAction, reward_config: RewardConfig) -> float:
    '''Compute Energy Penalty: Penalize control effort (steering and acceleration).'''
    # Quadratic penalty (more sensitive to larger actions)
    energy = action.steering_angle**2 + action.acceleration**2  
    energy_penalty = reward_config.w_energy_penalty * energy
    return -energy_penalty

def penalty_step(reward_config: RewardConfig) -> float:
    '''Compute Step Penalty: Penalize each step to encourage faster goal reaching.'''
    return -reward_config.w_step_penalty

# Reward function 
def compute_reward(
    state: VehicleState,
    new_state: VehicleState,
    action: VehicleAction,
    config: EnvConfig,
    reward_config: RewardConfig,
    is_goal_reached: bool,
    is_out_of_bounds: bool,
) -> tuple[float, RewardBreakdown]:
    '''Compute the total reward for a step in the OffroadEnv environment.'''
    r_progress = reward_progress(state, new_state, config, reward_config)
    r_border = penalty_border(new_state, config, reward_config)
    r_energy = penalty_energy(action, reward_config)
    r_step = penalty_step(reward_config)
    r_goal = 0.0
    r_collision = 0.0

    if is_goal_reached:
        r_goal = reward_config.goal_reward

    elif is_out_of_bounds:
        r_collision = -reward_config.collision_penalty

    # Regular step reward
    total_reward = sum([r_progress, r_border, r_energy, r_step, r_goal, r_collision])

    return (total_reward, RewardBreakdown(
        reward_progress=r_progress,
        penalty_border=r_border,
        penalty_energy=r_energy,
        penalty_step=r_step,
        reward_goal=r_goal,
        penalty_collision=r_collision,
        total_reward=total_reward
    ))