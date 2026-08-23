import pytest

from offroad_autonomous_navigator.envs.mujoco.mujoco_world import MujocoWorld
from offroad_autonomous_navigator.envs.schemas import VehicleAction, VehicleState


@pytest.fixture
def mujoco_world() -> MujocoWorld:
    """Initialize a MujocoWorld instance for testing."""
    return MujocoWorld()


def test_reset_returns_vehicle_at_origin(mujoco_world: MujocoWorld) -> None:
    """Test that after reset, the vehicle is at the origin with zero velocity."""
    state = mujoco_world.reset()

    assert isinstance(state, VehicleState)
    assert state.x == pytest.approx(0.0, abs=1e-2)
    assert state.y == pytest.approx(0.0, abs=1e-2)
    assert state.theta == pytest.approx(0.0, abs=1e-2)
    assert state.v == pytest.approx(0.0, abs=1e-2)


def test_step_null_action_produces_no_significant_movement(mujoco_world: MujocoWorld) -> None:
    """Test that applying a null action does not cause significant movement."""
    mujoco_world.reset()
    null_action = VehicleAction(steering_angle=0.0, acceleration=0.0)

    state = mujoco_world.step(null_action)

    assert state.v == pytest.approx(0.0, abs=1e-2)
    assert state.x == pytest.approx(0.0, abs=1e-2)
    assert state.y == pytest.approx(0.0, abs=1e-2)


def test_step_sustained_positive_acceleration_increases_velocity(mujoco_world: MujocoWorld) -> None:
    """Test that sustained positive acceleration increases velocity."""
    mujoco_world.reset()
    accel_action = VehicleAction(steering_angle=0.0, acceleration=1.0)

    # Advance several steps (e.g., 10 steps = 1 second of simulation with decision_dt=0.1)
    state = None
    for _ in range(100):
        state = mujoco_world.step(accel_action)

    assert state is not None
    assert state.v > 0.1
    assert state.x > 0.0  # With steering=0, it should move primarily along the X-axis