from pydantic import BaseModel, ConfigDict, Field


class VehicleState(BaseModel):
    model_config = ConfigDict(frozen=True) 

    x: float = Field(..., description="The x-coordinate of the vehicle's position.")
    y: float = Field(..., description="The y-coordinate of the vehicle's position.")
    theta: float = Field(..., description="The orientation of the vehicle in radians.")
    v: float = Field(..., description="The linear velocity of the vehicle. (m/s)")


class VehicleAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    steering_angle: float = Field(..., description="The steering angle of the vehicle in radians.")
    acceleration: float = Field(..., description="The linear acceleration of the vehicle. (m/s^2)")
