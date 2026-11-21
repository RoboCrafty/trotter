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
