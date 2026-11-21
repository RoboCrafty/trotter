"""Registers Trotter's tasks with mjlab.

`register_mjlab_task` gives a task a name. mjlab's `train` and `play` commands
then find it by that name:

    uv run train Trotter-Velocity-Flat
    uv run play Trotter-Velocity-Flat --checkpoint-file <model.pt>

How does mjlab know to import this file? Through the entry point in
pyproject.toml ([project.entry-points."mjlab.tasks"]): when mjlab starts, it
imports every package listed there, which runs the code below.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from trotter_rl.tasks.rl_cfg import trotter_ppo_runner_cfg
from trotter_rl.tasks.velocity_env_cfg import trotter_flat_env_cfg

register_mjlab_task(
    task_id="Trotter-Velocity-Flat",
    env_cfg=trotter_flat_env_cfg(),  # used by `train`
    play_env_cfg=trotter_flat_env_cfg(play=True),  # used by `play`
    rl_cfg=trotter_ppo_runner_cfg(),
    # mjlab's runner for velocity tasks: standard PPO training loop, plus it
    # knows how to export the policy to ONNX for the velocity task.
    runner_cls=VelocityOnPolicyRunner,
)
