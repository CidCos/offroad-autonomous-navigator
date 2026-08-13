from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Schemas 
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

class RewardBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    reward_progress: float = Field(..., description="Reward for moving closer to the goal.")
    penalty_border: float = Field(..., description="Penalty for proximity to map bounds.")
    penalty_energy: float = Field(..., description="Penalty for control effort (energy proxy).")
    penalty_step: float = Field(..., description="Fixed penalty per step to encourage efficiency.")
    reward_goal: float = Field(..., description="Reward for reaching the goal.")
    penalty_collision: float = Field(..., description="Penalty for leaving map boundaries.")
    total_reward: float = Field(..., description="Total reward for the step, " \
                                "combining all components.")

# BaseSettings to avoid hardcoding parameters. 

class EnvConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OFFROAD_", frozen=True)

    wheelbase: float = Field(default=2.0, gt=0, 
                            description="The wheelbase of the vehicle. (m)")
    max_steering_angle: float = Field(default=0.5, gt=0,
                                    description="The maximum steering angle of the vehicle. (rad)")
    max_acceleration: float = Field(default=3.0, gt=0,
                                    description="The maximum acceleration of the vehicle. (m/s^2)")
    max_speed: float = Field(default=10.0, gt=0,
                            description="The maximum speed of the vehicle. (m/s)")
    dt: float = Field(default=0.1, gt=0,
                    description="The time step for the simulation. (s)")
    max_episode_steps: int = Field(default=500, gt=0, 
                                description="The maximum number of steps in an episode.")
    goal_x: float = Field(default=20.0, 
                        description="The x-coordinate of the goal position. (m)")
    goal_y: float = Field(default=20.0, 
                        description="The y-coordinate of the goal position. (m)")
    goal_tolerance: float = Field(default=1.0, gt=0,
                                description="The tolerance for reaching the goal. (m)")
    map_min_x: float = Field(default=-25.0, 
                            description="The minimum x-coordinate of the map. (m)")
    map_max_x: float = Field(default=25.0, 
                            description="The maximum x-coordinate of the map. (m)")
    map_min_y: float = Field(default=-25.0, 
                            description="The minimum y-coordinate of the map. (m)")
    map_max_y: float = Field(default=25.0, 
                            description="The maximum y-coordinate of the map. (m)")

    @model_validator(mode="after")
    def check_map_bounds(self) -> "EnvConfig":

        if self.map_min_x >= self.map_max_x:
            raise ValueError(f"Map min x-coordinate {self.map_min_x}"
                            f" must be less than max x-coordinate {self.map_max_x}.")
        if self.map_min_y >= self.map_max_y:
            raise ValueError(f"Map min y-coordinate {self.map_min_y}"
                            f" must be less than max y-coordinate {self.map_max_y}.")

        return self

    @model_validator(mode="after")
    def check_goal_within_map(self) -> "EnvConfig":
        if not (self.map_min_x <= self.goal_x <= self.map_max_x):
            raise ValueError(
                f"Goal x-coordinate {self.goal_x} is out of map bounds "
                f"[{self.map_min_x}, {self.map_max_x}]."
            )
        if not (self.map_min_y <= self.goal_y <= self.map_max_y):
            raise ValueError(
                f"Goal y-coordinate {self.goal_y} is out of map bounds "
                f"[{self.map_min_y}, {self.map_max_y}]."
            )
        return self

class RewardConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OFFROAD_REWARD_", frozen=True)

    w_progress: float = Field(default=1.0, gt=0, 
                            description="Weight for progress towards the goal.")
    w_border_penalty: float = Field(default=1.0, gt=0, 
                                    description="Weight for proximity to map bounds.")
    w_energy_penalty: float = Field(default=1.0, gt=0,
                            description="Weight for control effort (energy proxy).")
    w_step_penalty: float = Field(default=0.05, gt=0, 
                                description="Fixed penalty per step to encourage efficiency.")
    goal_reward: float = Field(default=100.0, gt=0,
                            description="Reward for reaching the goal.")
    collision_penalty: float = Field(default=100.0, gt=0,
                                    description="Penalty for leaving map boundaries")
    border_margin: float = Field(default=5.0, gt=0,
                                description="Distance from bounds where border"
                                " penalty starts applying (m).")