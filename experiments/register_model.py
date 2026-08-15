from pathlib import Path
from typing import Any

import mlflow
import wandb
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from stable_baselines3 import PPO

CHECKPOINTS_PATH = Path(__file__).parent.parent / "checkpoints/xl6e0s0h/model.zip"
WANDB_RUN_ID = "xl6e0s0h"
NAME = "offroad-agent"


class SB3PPOWrapper(PythonModel):
    """A wrapper for Stable Baselines3 PPO models to be used with MLflow."""

    def load_context(self, context: PythonModelContext) -> None:
        """Load the PPO model from the specified path."""
        self.model = PPO.load(context.artifacts["sb3_model"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make predictions using the loaded PPO model."""
        actions, _ = self.model.predict(model_input, deterministic=True)
        return actions

def get_wandb_final_metrics(run_id: str, entity: str, project: str) -> dict[str, float]:
    """Fetch final summary metrics from a completed W&B run."""
    api = wandb.Api()
    # wandb's Api.run is not fully typed
    run = api.run(f"{entity}/{project}/{run_id}") #type: ignore[no-untyped-call]
    return {
        "ep_rew_mean": run.summary.get("rollout/ep_rew_mean"),
        "ep_len_mean": run.summary.get("rollout/ep_len_mean"),
    }

if __name__ == "__main__":
    entity = "ivan-cid-"
    project = "offroad-autonomous-navigator"
    metrics = get_wandb_final_metrics(WANDB_RUN_ID, entity, project)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(project)

    with mlflow.start_run(run_name="register-v1-baseline") as run:
        mlflow.log_param("wandb_run_id", WANDB_RUN_ID)
        mlflow.log_param("algorithm", "PPO")
        mlflow.log_param("policy", "MlpPolicy")
        mlflow.log_metric("final_ep_rew_mean", metrics["ep_rew_mean"])
        mlflow.log_metric("final_ep_len_mean", metrics["ep_len_mean"])
        mlflow.set_tag("description", "Baseline PPO, 200k steps, simple kinematics, no obstacles")

        example_input = [[5.0, 0.0, 20.0, 0.3]]  # [v, theta, distance_to_goal, relative_angle]
        example_output = [[0.5, 0.1]]  # [steering_angle, acceleration] normalizado

        mlflow.pyfunc.log_model(
            name="model",
            python_model=SB3PPOWrapper(),
            artifacts={"sb3_model": str(CHECKPOINTS_PATH)},
            signature=infer_signature(example_input, example_output),
            registered_model_name=NAME,
        )