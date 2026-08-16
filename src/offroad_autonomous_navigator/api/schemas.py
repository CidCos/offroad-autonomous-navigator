from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    x: float = Field(..., description="Vehicle x position (m).")
    y: float = Field(..., description="Vehicle y position (m).")
    theta: float = Field(..., description="Vehicle heading (rad).")
    v: float = Field(..., description="Vehicle speed (m/s).")
    goal_x: float = Field(..., description="Goal x position (m).")
    goal_y: float = Field(..., description="Goal y position (m).")


class PredictionResponse(BaseModel):
    steering_angle: float = Field(..., description="Predicted steering angle (rad).")
    acceleration: float = Field(..., description="Predicted acceleration (m/s^2).")