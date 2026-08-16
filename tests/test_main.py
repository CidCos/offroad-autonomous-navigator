from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from offroad_autonomous_navigator.api.main import app
from offroad_autonomous_navigator.envs.schemas import VehicleAction


@pytest.fixture
def client() -> Iterator[TestClient]:
    with patch("offroad_autonomous_navigator.api.main.InferenceService") as mock_service_cls:
        mock_service = MagicMock()
        mock_service.predict.return_value = VehicleAction(steering_angle=0.1, acceleration=0.5)
        mock_service_cls.return_value = mock_service

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def failing_client() -> Iterator[TestClient]:
    with patch("offroad_autonomous_navigator.api.main.InferenceService") as mock_service_cls:
        mock_service_cls.side_effect = RuntimeError("Weights failed to load")

        with TestClient(app) as test_client:
            yield test_client


# --- Tests ---


def test_predict_returns_200_and_valid_response(client: TestClient) -> None:
    response = client.post("/predict", json={
        "x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0,
        "goal_x": 20.0, "goal_y": 20.0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["steering_angle"] == pytest.approx(0.1)
    assert body["acceleration"] == pytest.approx(0.5)


def test_health_returns_ok_when_service_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_503_when_service_fails_to_load(failing_client: TestClient) -> None:
    response = failing_client.post("/predict", json={
        "x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0,
        "goal_x": 20.0, "goal_y": 20.0,
    })
    assert response.status_code == 503
    body = response.json()
    assert "Model not available" in body["detail"]
    assert "Weights failed to load" in body["detail"]


def test_health_returns_degraded_when_service_fails_to_load(failing_client: TestClient) -> None:
    response = failing_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["detail"] == "Weights failed to load"


@pytest.mark.parametrize(
    "invalid_payload",
    [
        # Empty Payload
        {},
        # Required fields missing (goal_x and goal_y)
        {"x": 0.0, "y": 0.0, "theta": 0.0, "v": 0.0},
        # Incompatible types (string instead of float)
        {"x": "not_a_number", "y": 0.0, "theta": 0.0, "v": 0.0, "goal_x": 20.0, "goal_y": 20.0},
    ],
)
def test_predict_returns_422_on_invalid_payload(client: TestClient, 
                                                invalid_payload: dict[str, float]) -> None:
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422