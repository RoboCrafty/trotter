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
    from mjlab.managers.scene_entity_config import SceneEntityCfg
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


class reward_param_curriculum(event_curriculum):
    """Same as event_curriculum, for a reward term's params, e.g. an air-time
    window that moves up as steps get longer:

        params={"reward_name": "air_time", "stages": [
            {"step": 0, "params": {"threshold_min": 0.125, "threshold_max": 0.3}}, ...]}

    (mjlab's reward_curriculum only changes the weight.)
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])
        self._stages = cfg.params["stages"]
        steps = [stage["step"] for stage in self._stages]
        assert steps == sorted(steps), "Stages must be in increasing step order."
        self._current = -1

    def __call__(self, env, env_ids, reward_name, stages):  # type: ignore[override]
        return super().__call__(env, env_ids, reward_name, stages)


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


def knee_bird_bend(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize bending the knees the "bird" way (negative angle).

    Positive knee angles swing the shin backward like a human knee; negative
    ones swing it forward. On the V8 leg the bird way lifted the foot more
    per degree (see README), so policies used it. Cost = sum over knees of how far they bend the bird way [rad], e.g.
    20 deg costs 0.35; any amount the human way costs nothing. Linear, not
    squared: a squared cost made small and medium bird bends nearly free.
    """
    knee = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(-knee, min=0.0), dim=1)


# Gait clock: a phase that cycles with a fixed period. The policy sees it
# (gait_phase) and is paid for moving its feet in time with it (gait_contact,
# gait_swing_height), which gives a steady left-right rhythm. Time comes from
# the episode step counter, so the clock starts at 0 on every reset; on the
# real robot it's a counter in the control loop.


def _foot_phases(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Each foot's position in the cycle, 0..1: [left, right], half a cycle apart."""
    t = env.episode_length_buf.float() * env.step_dt
    phase = torch.remainder(t / period, 1.0)
    return torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=1)


def _moving(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (total > threshold).float()


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Observation: sin and cos of the clock (smooth, no jump at the cycle end)."""
    angle = 2 * torch.pi * _foot_phases(env, period)[:, 0]
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)


def _tracking(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """How well the command is followed, 0..1 (same kernels as the tracking
    rewards). Gait rewards scaled by this pay little for marching in place."""
    from mjlab.tasks.velocity.mdp import track_angular_velocity, track_linear_velocity

    lin = track_linear_velocity(env, std=0.1**0.5, command_name=command_name)
    ang = track_angular_velocity(env, std=0.5**0.5, command_name=command_name)
    return lin * ang


def gait_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period: float,
    stance_fraction: float,
    command_name: str,
    command_threshold: float,
    scale_by_tracking: bool = False,
) -> torch.Tensor:
    """+1 per foot that's on the ground in its stance part of the cycle (the
    first `stance_fraction`) or in the air in its swing part. Only while
    moving is commanded."""
    want_stance = _foot_phases(env, period) < stance_fraction
    contact = env.scene[sensor_name].data.found > 0
    reward = (contact == want_stance).float().sum(dim=1)
    if scale_by_tracking:
        reward = reward * _tracking(env, command_name)
    return reward * _moving(env, command_name, command_threshold)


def gait_swing_height(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    period: float,
    stance_fraction: float,
    target_height: float,
    std: float,
    command_name: str,
    command_threshold: float,
    swing_only: bool = False,
    scale_by_tracking: bool = False,
) -> torch.Tensor:
    """Feet follow a height curve: 0 in stance, a sine arch up to
    `target_height` over the swing. Up to +1 per foot. Only while moving.

    swing_only: score only feet in their swing part. Otherwise a foot resting
    at height 0 in stance scores too, and buzzing the feet a few mm at
    ~15 steps/s collected half this reward for free (clock_w1)."""
    phases = _foot_phases(env, period)
    swing = ((phases - stance_fraction) / (1 - stance_fraction)).clamp(min=0.0)
    target = target_height * torch.sin(torch.pi * swing)
    height = env.scene[height_sensor_name].data.heights
    per_foot = torch.exp(-((height - target) ** 2) / std**2)
    if swing_only:
        per_foot = per_foot * (swing > 0).float()
    reward = per_foot.sum(dim=1)
    if scale_by_tracking:
        reward = reward * _tracking(env, command_name)
    return reward * _moving(env, command_name, command_threshold)


def hip_roll_sway(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, vy_full: float
) -> torch.Tensor:
    """Cost: |hip roll angle| summed over both legs, i.e. swinging the legs
    out to the side. Fades out as a sideways command grows (full cost at
    vy = 0, none from |vy| >= vy_full), since walking sideways needs it."""
    roll = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    vy = env.command_manager.get_command(command_name)[:, 1]
    scale = 1.0 - (torch.abs(vy) / vy_full).clamp(max=1.0)
    return torch.sum(torch.abs(roll), dim=1) * scale
