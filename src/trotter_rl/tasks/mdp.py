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
    from mjlab.managers.reward_manager import RewardTermCfg


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


def track_axis_velocity(env: ManagerBasedRlEnv, command_name: str, axis: str, std: float) -> torch.Tensor:
    """exp(-err^2 / std^2) for one command axis: "y" (sideways) or "yaw" (turn).

    mjlab's tracking terms mix axes (forward + sideways in one error), so a
    small axis that's ignored costs little: missing a 0.2 m/s sideways
    command by 0.16 costs ~23% of the linear term at std sqrt(0.1)."""
    robot = env.scene["robot"].data
    command = env.command_manager.get_command(command_name)
    if axis == "y":
        err = command[:, 1] - robot.root_link_lin_vel_b[:, 1]
    else:
        err = command[:, 2] - robot.root_link_ang_vel_b[:, 2]
    return torch.exp(-(err**2) / std**2)


class feet_stride:
    """Reward for long steps: while a foot is in the air, +distance it has
    moved (horizontally) since lifting off / target, capped at 1, per step.

    So a step pays more the farther it reaches and the longer it lasts,
    without paying for hops (nothing when both feet are in the air). Only
    while moving is commanded."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        n = len(cfg.params["asset_cfg"].site_names)
        self._liftoff = torch.zeros(env.num_envs, n, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        target: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        feet_xy = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, :2]
        contact = env.scene[sensor_name].data.found > 0
        # Feet on the ground keep updating their spot, so in the air it's
        # the lift-off point.
        self._liftoff = torch.where(contact.unsqueeze(-1), feet_xy, self._liftoff)
        dist = torch.norm(feet_xy - self._liftoff, dim=-1)
        reward = ((dist / target).clamp(max=1.0) * (~contact).float()).sum(dim=1)
        grounded = contact.any(dim=1).float()
        command = env.command_manager.get_command(command_name)
        moving = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold
        return reward * grounded * moving.float()


def upright_linear(env: ManagerBasedRlEnv) -> torch.Tensor:
    """0 upside down, 0.5 on its side, 1 upright. Linear, so lying down still
    shows which way is better (an exp(-tilt^2) kernel is flat there)."""
    return (1.0 - env.scene["robot"].data.projected_gravity_b[:, 2]) / 2.0


def base_height_ratio(env: ManagerBasedRlEnv, target: float) -> torch.Tensor:
    """Base height / standing height, capped at 1 (flat ground at z = 0)."""
    return (env.scene["robot"].data.root_link_pos_w[:, 2] / target).clamp(0.0, 1.0)


class height_record:
    """Paid each time the base gets higher than ever before in this episode:
    reward = how much the record went up (m). Over an episode it sums to the
    best height reached, so a throw that gets 2 cm higher than the last try
    pays right away, even if the robot falls back down."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._settle_steps = cfg.params.get("settle_steps", 25)
        self._best = torch.full((env.num_envs,), float("inf"), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._best[slice(None) if env_ids is None else env_ids] = float("inf")

    def __call__(self, env: ManagerBasedRlEnv, settle_steps: int = 25) -> torch.Tensor:
        # Robots are dropped from above standing height at reset, so nothing
        # counts until they've landed (settle_steps); then the record starts
        # at the height they landed at.
        height = env.scene["robot"].data.root_link_pos_w[:, 2]
        settled = env.episode_length_buf >= settle_steps
        self._best = torch.where(settled & torch.isinf(self._best), height, self._best)
        gain = torch.where(settled, (height - self._best).clamp(min=0.0), torch.zeros_like(height))
        self._best = torch.where(settled, torch.maximum(self._best, height), self._best)
        return gain


def feet_under_base(env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Both feet on the ground under the body: exp(-d^2/std^2) of the
    horizontal distance between the base and the midpoint of the feet, only
    when both feet touch the ground."""
    robot = env.scene[asset_cfg.name].data
    feet_mid = robot.site_pos_w[:, asset_cfg.site_ids, :2].mean(dim=1)
    dist = torch.norm(feet_mid - robot.root_link_pos_w[:, :2], dim=1)
    both_down = (env.scene[sensor_name].data.found > 0).all(dim=1).float()
    return torch.exp(-(dist**2) / std**2) * both_down


def reset_split_or_fallen(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    split_fraction: float,
    split_angle: tuple[float, float],
    split_height: float,
    fallen_pose_range: dict,
) -> None:
    """Get-up starts: `split_fraction` of robots begin in the splits (one
    thigh forward, one back, body upright on the ground, random heading and
    random leg forward), the rest dropped in a random orientation."""
    from mjlab.envs.mdp import reset_joints_by_offset, reset_root_state_uniform

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    split = torch.rand(len(env_ids), device=env.device) < split_fraction
    fallen_ids, split_ids = env_ids[~split], env_ids[split]
    if len(fallen_ids):
        reset_root_state_uniform(env, fallen_ids, pose_range=fallen_pose_range)
        reset_joints_by_offset(env, fallen_ids, position_range=(-0.5, 0.5), velocity_range=(0.0, 0.0))
    if len(split_ids):
        robot = env.scene["robot"]
        n = len(split_ids)
        root = torch.zeros(n, 13, device=env.device)
        root[:, :3] = env.scene.env_origins[split_ids]
        root[:, 2] += split_height
        yaw = (torch.rand(n, device=env.device) * 2 - 1) * torch.pi
        root[:, 3] = torch.cos(yaw / 2)
        root[:, 6] = torch.sin(yaw / 2)
        robot.write_root_state_to_sim(root, env_ids=split_ids)
        angle = split_angle[0] + (split_angle[1] - split_angle[0]) * torch.rand(n, device=env.device)
        sign = torch.where(torch.rand(n, device=env.device) < 0.5, 1.0, -1.0)
        pos = torch.zeros(n, len(robot.joint_names), device=env.device)
        for i, name in enumerate(robot.joint_names):
            if name == "joint_thigh_left":
                pos[:, i] = sign * angle
            elif name == "joint_thigh_right":
                pos[:, i] = -sign * angle
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=split_ids)


def split_pose(env: ManagerBasedRlEnv, angle: float, std: float, height_max: float) -> torch.Tensor:
    """Stage 1 of getting up: legs in the splits (thighs +-angle, opposite;
    either leg forward), paid while the body is low and fading out as it
    rises (zero from `height_max` up), so it leads into the rise."""
    robot = env.scene["robot"]
    q = robot.data.joint_pos
    names = list(robot.joint_names)
    left, right = q[:, names.index("joint_thigh_left")], q[:, names.index("joint_thigh_right")]
    err = torch.minimum((left - angle) ** 2 + (right + angle) ** 2, (left + angle) ** 2 + (right - angle) ** 2)
    low = (1.0 - robot.data.root_link_pos_w[:, 2] / height_max).clamp(0.0, 1.0)
    return torch.exp(-err / std**2) * low
