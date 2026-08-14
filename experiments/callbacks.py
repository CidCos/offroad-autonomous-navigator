import wandb
from stable_baselines3.common.callbacks import BaseCallback


class RewardBreakdownCallback(BaseCallback):
    """Logs individual reward components from the info dict to W&B."""

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            if "reward_breakdown" in info:
                breakdown = info["reward_breakdown"]
                wandb.log({
                    "reward/progress": breakdown.reward_progress,
                    "reward/border_penalty": breakdown.penalty_border,
                    "reward/energy_penalty": breakdown.penalty_energy,
                    "reward/step_penalty": breakdown.penalty_step,
                    "reward/goal_bonus": breakdown.reward_goal,
                    "reward/collision_penalty": breakdown.penalty_collision,
                    "distance_to_goal": info["distance_to_goal"]
                },
                step=self.num_timesteps)
        return True