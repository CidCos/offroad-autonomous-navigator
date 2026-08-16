import mlflow.pyfunc

from offroad_autonomous_navigator.envs.kinematic.observation import (
    scale_action,
    state_to_observation,
)
from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleAction, VehicleState

MODEL_URI = "models:/offroad-agent@production"


class InferenceService:
    """Loads a registered PPO model and serves predictions for given vehicle states."""

    def __init__(self, model_uri: str = MODEL_URI) -> None:
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, state: VehicleState, config: EnvConfig) -> VehicleAction:
        """Predict the next action given the current vehicle state."""
        observation = state_to_observation(state, config)
        action_array = self.model.predict(observation.reshape(1, -1))
        return scale_action(action_array.flatten(), config) 
