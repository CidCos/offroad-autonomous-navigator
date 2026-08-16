import mlflow
from mlflow import MlflowClient

MODEL_NAME = "offroad-agent"
VERSION_TO_PROMOTE = "3"
ALIAS = "production"

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=VERSION_TO_PROMOTE,
    )

    print(f"Model '{MODEL_NAME}' version {VERSION_TO_PROMOTE} promoted to alias '@{ALIAS}'.")