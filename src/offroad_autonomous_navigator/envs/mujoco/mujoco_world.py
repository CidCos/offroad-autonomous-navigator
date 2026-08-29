import math
from pathlib import Path

import mujoco
import numpy as np

from offroad_autonomous_navigator.envs.mujoco.sensors import DepthCamera
from offroad_autonomous_navigator.envs.schemas import VehicleAction, VehicleState

VEHICLE_XML_PATH = Path(__file__).resolve().parents[4] / "assets" / "mujoco" / "vehicle.xml"
print(f"Using vehicle XML path: {VEHICLE_XML_PATH}")

class MujocoWorld:
    """Physics simulation of the offroad vehicle using MuJoCo."""
    def __init__(
        self,
        xml_path: Path = VEHICLE_XML_PATH,
        decision_dt: float = 0.1,
        max_speed: float = 10.0,
        camera_name: str = "front_depth_cam",
        depth_max_range: float = 10.0,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.physics_timestep = self.model.opt.timestep
        self.n_substeps = round(decision_dt / self.physics_timestep)
        self.max_speed = max_speed
        self._velocity_target = 0.0
        self.camera = DepthCamera(self.model, camera_name, max_range=depth_max_range)

    def reset(self) -> VehicleState:
        mujoco.mj_resetData(self.model, self.data)
        self._velocity_target = 0.0
        return self._read_state()

    def step(self, action: VehicleAction) -> VehicleState:
        """Apply an action and advance the simulation by one decision interval."""
        self._velocity_target += action.acceleration * (
            self.n_substeps * self.physics_timestep
        )
        self._velocity_target = max(-self.max_speed, min(self.max_speed, self._velocity_target))

        self.data.ctrl[0] = action.steering_angle
        self.data.ctrl[1] = self._velocity_target
        self.data.ctrl[2] = self._velocity_target

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self._read_state()

    def _read_state(self) -> VehicleState:
        x, y = self.data.qpos[0], self.data.qpos[1]
        w, qx, qy, qz = self.data.qpos[3:7]
        theta = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        vx, vy = self.data.qvel[0], self.data.qvel[1]
        speed = math.hypot(vx, vy)

        return VehicleState(x=float(x), y=float(y), theta=float(theta), v=float(speed))

    def render_depth(self) -> np.ndarray:
        """Return the current depth image from the vehicle's front camera."""
        return self.camera.render(self.data)

if __name__ == "__main__":
    world = MujocoWorld()
    state = world.reset()
    print(f"Initial state: {state}")
    for i in range(100):
        action = VehicleAction(steering_angle=0.1, acceleration=1.0)
        state = world.step(action)
        print(f"Step {i + 1}: {state}")
        print(f"Position: ({state.x:.2f}, {state.y:.2f}), "
            f"Orientation: {state.theta:.2f} rad, Speed: {state.v:.2f} m/s")
