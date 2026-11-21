"""Custom MDP terms for Trotter (things mjlab doesn't provide out of the box).

"MDP" (Markov Decision Process) is the RL name for the task definition:
observations, actions, rewards, terminations... mjlab keeps its term functions
in modules called `mdp`, so we do the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


class event_curriculum:
    """Change an event's params in stages as training progresses.

    mjlab has `reward_curriculum` and `termination_curriculum` but nothing for
    events. This is the same idea for events, e.g. pushes that start at zero
    and get stronger over time:

        CurriculumTermCfg(
            func=event_curriculum,
            params={
                "event_name": "push_robot",
                "stages": [
                    {"step": 0, "params": {"velocity_range": {}}},
                    {"step": 24_000, "params": {"velocity_range": {"x": (-0.1, 0.1)}}},
                ],
            },
        )

    `step` counts environment steps (env.common_step_counter). With 24 steps
    per PPO iteration, iteration N is step N * 24.

    Only works for events whose params are read every time they fire
    ("interval" and "reset" events). "startup" events run once at the very
    beginning, so changing their params later has no effect.

    It's a class (not a function) because it needs to remember things between
    calls: mjlab builds it once with __init__, then calls it every time the
    curriculum is updated.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._term_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
        self._stages: list[dict[str, Any]] = cfg.params["stages"]
        steps = [stage["step"] for stage in self._stages]
        assert steps == sorted(steps), "Stages must be in increasing step order."
        self._current = -1  # Index of the stage applied last.

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        event_name: str,
        stages: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del env_ids, event_name, stages  # Already stored in __init__.
        # The latest stage whose step we've reached.
        reached = [i for i, s in enumerate(self._stages) if env.common_step_counter >= s["step"]]
        if reached and reached[-1] != self._current:
            self._current = reached[-1]
            self._term_cfg.params.update(self._stages[self._current]["params"])
        # The returned values get logged (shows up in wandb/tensorboard).
        return {"stage": torch.tensor(float(self._current))}


def feet_air_time_biped(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold: float = 0.4,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward a calm, alternating walk: one foot on the ground, one in the air.

    Only pays in *single stance* (exactly one foot touching the ground). The
    payout is the shorter of the two feet's current phase durations, i.e. how
    long the swing foot has been in the air AND the stance foot on the ground,
    capped at `threshold` seconds. So:

        long, alternating steps    -> pays up to `threshold` per step
        quick little shuffles      -> pays very little (phases are short)
        both feet on the ground    -> pays 0
        both feet in the air (hop) -> pays 0

    This is the biped version used in Isaac Lab's humanoid tasks. mjlab's own
    `feet_air_time` instead pays for every foot in the air, both at once
    included, which Trotter exploited by hopping.

    Only active while moving is commanded (standing still shouldn't step).
    """
    sensor = env.scene[sensor_name]
    air_time = sensor.data.current_air_time  # [num_envs, 2] seconds in the air
    contact_time = sensor.data.current_contact_time  # [num_envs, 2] seconds down
    in_contact = contact_time > 0.0
    # For each foot: how long it has been in its CURRENT phase.
    phase_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.sum(dim=1) == 1
    reward = torch.where(single_stance, phase_time.min(dim=1).values, 0.0)
    reward = reward.clamp(max=threshold)

    command = env.command_manager.get_command(command_name)
    moving = (torch.norm(command[:, :2], dim=1) + command[:, 2].abs()) > command_threshold
    return reward * moving.float()
