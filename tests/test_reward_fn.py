
import pytest

from offroad_autonomous_navigator.envs.kinematic.reward_fn import (
    compute_reward,
    penalty_border,
    penalty_energy,
    penalty_step,
    reward_progress,
)
from offroad_autonomous_navigator.envs.schemas import (
    EnvConfig,
    RewardConfig,
    VehicleAction,
    VehicleState,
)


# 1. Border Penalty: Vehicle far from border and next to border.
@pytest.mark.parametrize(
    "state, expected_penalty",
    [
        #Vehicle far from border, no penalty
        (VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0), 
        0.0), 
        #Vehicle at right border, max penalty
        (VehicleState(x=50.0, y=0.0, theta=0.0, v=5.0), 
        -1.0), 
        #Vehicle at top border, max penalty
        (VehicleState(x=0.0, y=50.0, theta=0.0, v=5.0),
        -1.0), 
        #Vehicle at left border, max penalty
        (VehicleState(x=-50.0, y=0.0, theta=0.0, v=5.0), 
        -1.0),
        #Vehicle at bottom border, max penalty
        (VehicleState(x=0.0, y=-50.0, theta=0.0, v=5.0), 
        -1.0),                                                        
    ]
)
def test_penalty_border(default_config: EnvConfig, 
                        default_reward_config: RewardConfig, 
                        state: VehicleState, 
                        expected_penalty: float
                        ) -> None:
    penalty = penalty_border(state, default_config, default_reward_config)
    assert penalty == pytest.approx(expected_penalty, rel=1e-4)

# 2. Border Penalty: Close to border but not exceeding margin
def test_penalty_border_close_to_border(default_config: EnvConfig, 
                                        default_reward_config: RewardConfig
                                        ) -> None:
    state = VehicleState(x=46.0, y=0.0, theta=0.0, v=5.0)
    penalty = penalty_border(state, default_config, default_reward_config)
    assert penalty < 0.0  # Should be a negative penalty

# 3. Border Penalty: Exceeding border margin
def test_penalty_border_exceeding_margin(default_config: EnvConfig,
                                        default_reward_config: RewardConfig
                                        ) -> None:
    state = VehicleState(x=51.0, y=0.0, theta=0.0, v=5.0)
    penalty = penalty_border(state, default_config, default_reward_config)
    assert penalty == pytest.approx(-1.0, rel=1e-4)  # Should be the maximum penalty

# 4. Reward Progress: prev_state vs new_state
@pytest.mark.parametrize(
    "prev_state, new_state, expected_reward",
    [
        # Moving closer to goal
        (VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0), 
        VehicleState(x=1.0, y=1.0, theta=0.0, v=5.0), 
        pytest.approx(1.4142, rel=1e-4)), 
        # Moving away from goal
        (VehicleState(x=10.0, y=10.0, theta=0.0, v=5.0), 
        VehicleState(x=9.0, y=9.0, theta=0.0, v=5.0), 
        pytest.approx(-1.4142, rel=1e-4)),  
        # No movement
        (VehicleState(x=5.0, y=5.0, theta=0.0, v=5.0), 
        VehicleState(x=5.0, y=5.0, theta=0.0, v=5.0), 
        0.0),  
    ]
)
def test_reward_progress(
    default_config: EnvConfig,
    default_reward_config: RewardConfig,
    prev_state: VehicleState,
    new_state: VehicleState,
    expected_reward: float
) -> None:
    reward = reward_progress(prev_state, new_state, default_config, default_reward_config)
    assert reward == pytest.approx(expected_reward, rel=1e-4)

# 5. Energy Penalty: Check different actions and their corresponding penalties

@pytest.mark.parametrize(
    "action, expected_penalty",
    [
        # No action, no penalty
        (VehicleAction(steering_angle=0.0, acceleration=0.0), 0.0),  
        # Steering only
        (VehicleAction(steering_angle=1.0, acceleration=0.0), -0.1),
        # Acceleration only  
        (VehicleAction(steering_angle=0.0, acceleration=1.0), -0.1), 
        # Both actions 
        (VehicleAction(steering_angle=1.0, acceleration=1.0), -0.2),  
    ]
)
def test_penalty_energy(
    default_reward_config: RewardConfig,
    action: VehicleAction,
    expected_penalty: float
) -> None:
    penalty = penalty_energy(action, default_reward_config)
    assert penalty == pytest.approx(expected_penalty, rel=1e-4)

# 6. Energy Penalty: Compare penalties for different magnitudes of actions
def test_penalty_energy_magnitude(default_reward_config: RewardConfig) -> None:
    small_action = VehicleAction(steering_angle=0.1, acceleration=0.1)
    large_action = VehicleAction(steering_angle=1.0, acceleration=1.0)

    small_penalty = penalty_energy(small_action, default_reward_config)
    large_penalty = penalty_energy(large_action, default_reward_config)

    assert abs(large_penalty) > abs(small_penalty)

# 7. Step Penalty: Check that the step penalty is constant and negative
def test_penalty_step(default_reward_config: RewardConfig) -> None:
    penalty = penalty_step(default_reward_config)
    assert penalty == pytest.approx(-default_reward_config.w_step_penalty, rel=1e-4)

# 8. Compute Reward: is_goal_reached
def test_compute_reward_goal_reached(default_config: EnvConfig, 
                                    default_reward_config: RewardConfig) -> None:

    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    new_state = VehicleState(x=5.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=0.1)

    r_progress = reward_progress(state, new_state, default_config, default_reward_config)
    r_border = penalty_border(new_state, default_config, default_reward_config)
    r_energy = penalty_energy(action, default_reward_config)
    r_step = penalty_step(default_reward_config)

    total_reward, reward_breakdown = compute_reward(
        state, new_state, action, default_config, default_reward_config,
        is_goal_reached=True, is_out_of_bounds=False
    )

    assert total_reward == pytest.approx(r_progress + r_border + r_energy + r_step + 100.0) 
    assert reward_breakdown.reward_progress == pytest.approx(r_progress)
    assert reward_breakdown.penalty_border == pytest.approx(r_border)
    assert reward_breakdown.penalty_energy == pytest.approx(r_energy)
    assert reward_breakdown.penalty_step == pytest.approx(r_step)
    assert reward_breakdown.reward_goal == pytest.approx(default_reward_config.goal_reward)
    assert reward_breakdown.penalty_collision == pytest.approx(0.0)
    assert reward_breakdown.total_reward == pytest.approx(total_reward)

# 9. Compute Reward: is_out_of_bounds
def test_compute_reward_out_of_bounds(default_config: EnvConfig, 
                                    default_reward_config: RewardConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    new_state = VehicleState(x=5.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=0.1)

    r_progress = reward_progress(state, new_state, default_config, default_reward_config)
    r_border = penalty_border(new_state, default_config, default_reward_config)
    r_energy = penalty_energy(action, default_reward_config)
    r_step = penalty_step(default_reward_config)

    total_reward, reward_breakdown = compute_reward(
        state, new_state, action, default_config, default_reward_config,
        is_goal_reached=False, is_out_of_bounds=True
    )

    assert total_reward == pytest.approx(r_progress + r_border + r_energy + r_step - 100.0) 
    assert reward_breakdown.reward_progress == pytest.approx(r_progress)
    assert reward_breakdown.penalty_border == pytest.approx(r_border)
    assert reward_breakdown.penalty_energy == pytest.approx(r_energy)
    assert reward_breakdown.penalty_step == pytest.approx(r_step)
    assert reward_breakdown.reward_goal == pytest.approx(0.0)
    assert reward_breakdown.penalty_collision == pytest.approx(
                                                -default_reward_config.collision_penalty
                                                )
    assert reward_breakdown.total_reward == pytest.approx(total_reward)

# 11. Compute Reward: Normal step (not goal reached, not out of bounds)
def test_compute_reward_normal_step(default_config: EnvConfig, 
                                    default_reward_config: RewardConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    new_state = VehicleState(x=5.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=0.1)

    r_progress = reward_progress(state, new_state, default_config, default_reward_config)
    r_border = penalty_border(new_state, default_config, default_reward_config)
    r_energy = penalty_energy(action, default_reward_config)
    r_step = penalty_step(default_reward_config)

    total_reward, _ = compute_reward(
        state, new_state, action, default_config, default_reward_config,
        is_goal_reached=False, is_out_of_bounds=False
    )

    assert total_reward == pytest.approx(r_progress + r_border + r_energy + r_step)  

# 12. Compute Reward: Check that the reward is a sum of all components
def test_compute_reward_components(default_config: EnvConfig, 
                                    default_reward_config: RewardConfig) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    new_state = VehicleState(x=5.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=0.1)

    r_progress = reward_progress(state, new_state, default_config, default_reward_config)
    r_border = penalty_border(new_state, default_config, default_reward_config)
    r_energy = penalty_energy(action, default_reward_config)
    r_step = penalty_step(default_reward_config)

    total_reward, reward_breakdown = compute_reward(
        state, new_state, action, default_config, default_reward_config,
        is_goal_reached=False, is_out_of_bounds=False
    )

    assert total_reward == pytest.approx(r_progress + r_border + r_energy + r_step)
    assert reward_breakdown.reward_progress == pytest.approx(r_progress)
    assert reward_breakdown.penalty_border == pytest.approx(r_border)
    assert reward_breakdown.penalty_energy == pytest.approx(r_energy)
    assert reward_breakdown.penalty_step == pytest.approx(r_step)
    assert reward_breakdown.reward_goal == pytest.approx(0.0)
    assert reward_breakdown.penalty_collision == pytest.approx(0.0)