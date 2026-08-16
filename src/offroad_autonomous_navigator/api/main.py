from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from loguru import logger

from offroad_autonomous_navigator.api.inference import InferenceService
from offroad_autonomous_navigator.api.schemas import PredictionRequest, PredictionResponse
from offroad_autonomous_navigator.envs.schemas import EnvConfig, VehicleState
from offroad_autonomous_navigator.utils.config_loader import load_settings_from_yaml

CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"


@asynccontextmanager 
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        app.state.inference_service = InferenceService()
        app.state.base_config = load_settings_from_yaml(EnvConfig, CONFIG_DIR / "env_config.yaml")
        app.state.startup_error = None
        logger.info("Inference service loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load inference service: {exc}")
        app.state.startup_error = str(exc)
        app.state.inference_service = None
        app.state.base_config = None
    yield


app = FastAPI(title="Offroad Autonomous Navigator API", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if app.state.inference_service is None:
        raise HTTPException(status_code=503, detail=f"Model not available:"
                            f"{app.state.startup_error}")

    request_config = app.state.base_config.model_copy(
        update={"goal_x": request.goal_x, "goal_y": request.goal_y}
    )
    state = VehicleState(x=request.x, y=request.y, theta=request.theta, v=request.v)

    action = app.state.inference_service.predict(state, request_config)

    return PredictionResponse(steering_angle=action.steering_angle, 
                            acceleration=action.acceleration)


@app.get("/health")
def health() -> dict[str, str]:
    if app.state.inference_service is None:
        return {"status": "degraded", "detail": app.state.startup_error}
    return {"status": "ok"}