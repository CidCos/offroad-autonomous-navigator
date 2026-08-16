import math

import pytest

from offroad_autonomous_navigator.envs.kinematic.kinematics import BicycleModel
from offroad_autonomous_navigator.envs.schemas import VehicleAction, VehicleState


# Test cases for the BicycleModel class
@pytest.fixture
def default_model() -> BicycleModel:
    return BicycleModel(wheelbase=2.0, max_steering_angle=0.5, max_acceleration=3.0, max_speed=10.0)

# 1. Vehicle with v=0 and no acceleration should remain stationary.
def test_stationary_state(default_model: BicycleModel) -> None:
    initial = VehicleState(x=1.0, y=2.0, theta=0.5, v=0.0)
    action = VehicleAction(steering_angle=0.0, acceleration=0.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.x == initial.x
    assert new_state.y == initial.y
    assert new_state.theta == initial.theta
    assert new_state.v == initial.v

# 2. Vehicle with v=0 and positive acceleration should start moving.
def test_stationary_state_with_acceleration(default_model: BicycleModel) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    action = VehicleAction(steering_angle=0.0, acceleration=2.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.v > 0
    assert new_state.x > initial.x or new_state.y > initial.y  # Should have moved

# 3. Vehicle with v>0, no acceleration and no steering should maintain its speed and orientation.
def test_constant_speed(default_model: BicycleModel) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.0, acceleration=0.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.v == initial.v
    assert new_state.x > initial.x  # Should have moved forward
    assert new_state.theta == initial.theta  # Should have maintained orientation

# 4. Vehicle with v>0 and negative acceleration should slow down.
def test_deceleration(default_model: BicycleModel) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.0, acceleration=-2.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.v < initial.v
    assert new_state.x > initial.x  # Should have moved forward

# 5. Vehicle with v>0 and positive acceleration should speed up.
def test_acceleration(default_model: BicycleModel) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.0, acceleration=2.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.v > initial.v
    assert new_state.x > initial.x  # Should have moved forward

# 6. Vehicle with a positive steering angle should change its orientation to the left.
def test_left_turn(default_model: BicycleModel) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.3, acceleration=0.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.theta > initial.theta  # Should have turned left
    assert new_state.x > initial.x  # Should have moved forward
    assert new_state.y > initial.y  # Should have moved upward

# 7. Vehicle with a negative steering angle should change its orientation to the right.
def test_right_turn(default_model: BicycleModel ) -> None:
    initial = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=-0.3, acceleration=0.0)
    new_state = default_model.step(initial, action, dt=1.0)

    assert new_state.theta < initial.theta  # Should have turned right
    assert new_state.x > initial.x  # Should have moved forward
    assert new_state.y < initial.y  # Should have moved downward

# 8. Check that the vehicle's speed maintains within the maximum and minimum speed limit.
@pytest.mark.parametrize(
    "initial_v, acceleration, expected_v",
    [
        (5.0, 0.0, 5.0),      # No acceleration, speed should remain the same
        (9.0, 5.0, 10.0),     # exceeds max_speed, should clamp to 10.0
        (-9.0, -5.0, -10.0),  # exceeds -max_speed, should clamp to -10.0
    ],
)
def test_speed_clamping(default_model: BicycleModel, initial_v: float, 
                        acceleration: float, expected_v: float) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=initial_v)
    action = VehicleAction(steering_angle=0.0, acceleration=acceleration)
    new_state = default_model.step(state, action, dt=1.0)
    assert new_state.v == pytest.approx(expected_v)

# 9. Check that the vehicle's steering angle maintains within the maximum 
# and minimum steering angle limit.
@pytest.mark.parametrize(
    "excessive_steering_angle, expected_steering_angle",
    [
        (0.6, 0.5),   # exceeds max_steering_angle, should clamp to 0.5
        (-0.6, -0.5)  # exceeds -max_steering_angle, should clamp to -0.5
    ],
)
def test_steering_clamping_produces_same_result_as_max_value(default_model: BicycleModel, 
                                                            excessive_steering_angle: float, 
                                                            expected_steering_angle: float
                                                            ) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action_excessive = VehicleAction(steering_angle=excessive_steering_angle, acceleration=0.0)  
    action_at_limit = VehicleAction(steering_angle=expected_steering_angle, acceleration=0.0)  

    result_excessive = default_model.step(state, action_excessive, dt=1.0)
    result_at_limit = default_model.step(state, action_at_limit, dt=1.0)

    assert result_excessive.theta == pytest.approx(result_at_limit.theta)

# 10. Check that the vehicle's orientation (theta) is wrapped within the range [-pi, pi].
def test_theta_wrapping(default_model: BicycleModel) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=math.pi - 0.1, v=5.0)
    action = VehicleAction(steering_angle=0.3, acceleration=0.0)
    new_state = default_model.step(state, action, dt=1.0)

    assert -math.pi <= new_state.theta <= math.pi

# 11. Check that the vehicle's state is updated correctly after multiple steps.
def test_multiple_steps(default_model: BicycleModel) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=1.0)

    for _ in range(5):
        state = default_model.step(state, action, dt=1.0)

    assert state.v > 5.0  # Speed should have increased
    assert state.x > 0.0  # Should have moved forward
    assert state.y > 0.0  # Should have moved upward

# 12. Check that dt must be positive, otherwise a ValueError is raised.
def test_negative_dt(default_model: BicycleModel) -> None:
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
    action = VehicleAction(steering_angle=0.1, acceleration=1.0)

    with pytest.raises(ValueError):
        default_model.step(state, action, dt=-1.0)