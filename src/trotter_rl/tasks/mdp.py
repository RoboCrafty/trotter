"""Custom MDP terms for Trotter (things mjlab doesn't provide out of the box).

"MDP" (Markov Decision Process) is the RL name for the task definition:
observations, actions, rewards, terminations... mjlab keeps its term functions
in modules called `mdp`, so we do the same.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
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


class foot_flat:
    """Penalize a tilted sole while the foot is on the ground.

    Cost per foot in contact = sin^2(tilt), where tilt is the angle between
    the sole's normal and straight up (12 deg -> 0.043). Trotter has no ankle,
    so a foot can only be flat when its leg is vertical: this also stops the
    knock-kneed stance checkpoint 2750 learned (hip roll ~12 deg inward, feet
    4.5 cm from the center instead of 10, standing on their outer edges).

    The sole normal is measured once, in each foot body's own frame, at the
    default pose (all joints 0, where the soles are flat on the ground).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        import mujoco

        from trotter_rl.robot.trotter_constants import get_spec

        model = get_spec().compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        robot = env.scene["robot"]
        self._body_ids, normals = [], []
        for geom in cfg.params["foot_geoms"]:
            body = model.geom_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            self._body_ids.append(robot.body_names.index(name))
            # World "up" expressed in the foot body's frame at the default pose.
            normals.append(data.xmat[body].reshape(3, 3).T @ [0.0, 0.0, 1.0])
        self._normals_b = torch.tensor(np.array(normals), dtype=torch.float32, device=env.device)

    def __call__(
        self, env: ManagerBasedRlEnv, sensor_name: str, foot_geoms: tuple[str, ...]
    ) -> torch.Tensor:
        del foot_geoms  # Used in __init__.
        from mjlab.utils.lab_api.math import quat_apply

        quat = env.scene["robot"].data.body_link_quat_w[:, self._body_ids]  # [N, 2, 4]
        normal_w = quat_apply(quat, self._normals_b.expand(quat.shape[0], -1, -1))
        sin2_tilt = torch.sum(torch.square(normal_w[..., :2]), dim=-1)  # [N, 2]
        in_contact = env.scene[sensor_name].data.current_contact_time > 0.0
        return torch.sum(sin2_tilt * in_contact, dim=1)


def feet_air_time_grounded(env: ManagerBasedRlEnv, sensor_name: str, **kwargs) -> torch.Tensor:
    """mjlab's `feet_air_time`, but paying nothing while both feet are airborne.

    mjlab's version pays per foot in the air, both at once included, which an
    earlier Trotter run exploited by hopping. Walking always keeps one foot
    down, so this changes nothing for a real gait. `kwargs` are passed through
    (threshold_min, threshold_max, command_name, command_threshold).
    """
    from mjlab.tasks.velocity.mdp import feet_air_time

    reward = feet_air_time(env, sensor_name, **kwargs)
    grounded = (env.scene[sensor_name].data.current_contact_time > 0.0).any(dim=1)
    return reward * grounded.float()
