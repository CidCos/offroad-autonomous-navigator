from typing import cast

import mujoco
import numpy as np


class DepthCamera:
    """Renders a normalized depth image from a fixed camera in the MuJoCo scene."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str,
        height: int = 64,
        width: int = 64,
        max_range: float = 10.0,
    ) -> None:
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.renderer.enable_depth_rendering()
        self.camera_name = camera_name
        self.max_range = max_range

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Return a depth image normalized to [0, 1], clipped at max_range."""
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = self.renderer.render()
        clipped = np.clip(depth, 0, self.max_range)
        normalized = (clipped / self.max_range).astype(np.float32)
        return cast(np.ndarray, normalized)