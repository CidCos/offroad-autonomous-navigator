import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from offroad_autonomous_navigator.api.inference import InferenceService
from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleState
from offroad_autonomous_navigator.utils.geometry import euclidean_distance, normalize_angle


# Test that the InferenceService correctly loads the model and predicts actions
def test_predict_calls_model_and_scales_action(default_config: EnvConfig) -> None:
    with patch("offroad_autonomous_navigator.api.inference.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock() 
        mock_model.predict.return_value = np.array([[0.5, -0.3]])  # acción normalizada falsa
        mock_load.return_value = mock_model

        service = InferenceService()
        state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
        action = service.predict(state, default_config)

        assert action.steering_angle == pytest.approx(0.5 * default_config.max_steering_angle)
        assert action.acceleration == pytest.approx(-0.3 * default_config.max_acceleration)

# 2. Check that the model is loaded with the correct URI
def test_model_loaded_with_correct_uri() -> None:
    with patch("offroad_autonomous_navigator.api.inference.mlflow.pyfunc.load_model") as mock_load:
        InferenceService()
        mock_load.assert_called_once_with("models:/offroad-agent@production")

# 3. Check state_to_observation is called with correct parameters with mock_model.predict.call_args
def test_predict_calls_state_to_observation(default_config: EnvConfig) -> None:
    with patch("offroad_autonomous_navigator.api.inference.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock() 
        mock_model.predict.return_value = np.array([[0.5, -0.3]])  
        mock_load.return_value = mock_model

        service = InferenceService()
        state = VehicleState(x=0.0, y=0.0, theta=0.0, v=5.0)
        service.predict(state, default_config)

        # Verify that the observation passed to the model is correct
        expected_observation = np.array([
            state.v,
            state.theta,
            euclidean_distance(state.x, state.y, default_config.goal_x, default_config.goal_y),
            normalize_angle(
                math.atan2(
                    default_config.goal_y - state.y, 
                    default_config.goal_x - state.x
                    )
                - state.theta
            )
        ], dtype=np.float32).reshape(1, -1)

        called_args = mock_model.predict.call_args[0][0] 
        assert np.allclose(called_args, expected_observation)
