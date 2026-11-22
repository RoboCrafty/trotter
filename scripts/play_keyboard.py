"""Play a trained policy in the native MuJoCo viewer, driven from the keyboard.

mjlab's `play` command has no keyboard control in the native viewer (only
sliders in viser). This script loads a checkpoint the same way `play` does,
then lets you set the velocity command with keys typed in the TERMINAL:

    Up / Down     forward / backward     (lin_vel_x)
    a / d         sideways left / right  (lin_vel_y)
    Left / Right  turn left / right      (ang_vel_z)
    Space         stop (all commands to 0)

Keys are read from the terminal, not the viewer window, because MuJoCo's
viewer already uses almost every letter key for display toggles (contacts,
joints, transparency...). Keep the terminal focused while driving. The
viewer's own keys still work in its window (Enter = reset, P = plots, ...).

Run (macOS needs mjpython for the native viewer):

    uv run mjpython scripts/play_keyboard.py model_2400.pt

Add --foxglove to also stream to Foxglove (connect to ws://localhost:8765):
joint states, policy actions, the command, base state, foot contacts, and
two cameras (the robot's head camera and one following the robot).
"""

import argparse
import io
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

import trotter_rl.tasks  # noqa: F401  (registers Trotter-Velocity-Flat)
from trotter_rl.tasks.velocity_env_cfg import OBSERVATION_VARIANTS, trotter_flat_env_cfg

# Change per keypress, and the largest command allowed. The limits are well
# past the training ranges (0.4 / 0.3 / 1.0)
# on purpose: command anything and see what the policy does. Beyond what it
# was trained on it can't follow, and may stumble or fall.
STEP = {"x": 0.1, "y": 0.05, "yaw": 0.1}
LIMIT = {"x": 2.0, "y": 1.0, "yaw": 3.0}


class KeyboardCommand:
    """Reads single keypresses from the terminal and keeps the current command.

    The terminal is put in "cbreak" mode: keys arrive one at a time without
    pressing Enter, and Ctrl+C still works. A background thread does the
    reading so the simulation never waits on the keyboard.
    """

    # Arrow keys arrive as three characters: ESC, "[", then one of these.
    ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self):
        self.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self._lock = threading.Lock()  # cmd is written here, read by the sim.
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._read_loop, daemon=True).start()

    def restore_terminal(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def get(self) -> list[float]:
        with self._lock:
            return [self.cmd["x"], self.cmd["y"], self.cmd["yaw"]]

    def _read_char(self, timeout: float) -> str | None:
        ready, _, _ = select.select([self._fd], [], [], timeout)
        return os.read(self._fd, 1).decode(errors="ignore") if ready else None

    def _read_loop(self):
        while True:
            ch = self._read_char(0.1)
            if ch == "\x1b" and self._read_char(0.05) == "[":
                ch = self.ARROWS.get(self._read_char(0.05) or "")
            if ch:
                self._handle(ch.lower())

    def _handle(self, key: str):
        # key -> (which command, direction)
        bindings = {
            "up": ("x", +1), "down": ("x", -1),
            "a": ("y", +1), "d": ("y", -1),
            "left": ("yaw", +1), "right": ("yaw", -1),
        }
        with self._lock:
            if key == " ":
                self.cmd = {"x": 0.0, "y": 0.0, "yaw": 0.0}
            elif key in bindings:
                name, sign = bindings[key]
                value = self.cmd[name] + sign * STEP[name]
                self.cmd[name] = round(max(-LIMIT[name], min(LIMIT[name], value)), 2)
            else:
                return
            c = self.cmd
        # "\r" + padding: overwrite one status line instead of scrolling.
        print(f"\rcmd  x {c['x']:+.2f} m/s   y {c['y']:+.2f} m/s   "
              f"yaw {c['yaw']:+.2f} rad/s     ", end="", flush=True)


class FoxgloveStreamer:
    """Publishes the robot's state and two camera images to Foxglove.

    JSON topics (every policy step, 50 Hz):
        /joint_states   name, position, velocity, effort (torque), target
        /policy/action  the raw policy output (before scaling)
        /command        the velocity command
        /base           position, orientation, velocities, projected gravity
        /feet           ground contact and height per foot
    Image topics (every `camera_every` steps, ~17 Hz), JPEG:
        /camera/head    the robot's own head camera (on the base front)
        /camera/follow  a camera that follows the robot
    """

    def __init__(self, env, camera_every: int = 3, width: int = 320, height: int = 240):
        import foxglove

        self._foxglove = foxglove
        self._env = env.unwrapped
        self._robot = self._env.scene["robot"]
        self._contact = self._env.scene["feet_ground_contact"]
        self._feet = [self._robot.site_names.index(s) for s in ("left_foot", "right_foot")]
        self._camera_every, self._size = camera_every, (height, width)
        self._step = 0
        # Our own MjData copy for rendering: the simulation's data lives in
        # batched (Warp) arrays, so each frame copies robot 0's state over.
        self._model = self._env.sim.mj_model
        self._data = mujoco.MjData(self._model)
        self._renderer = None  # Created on first use, in the simulation thread.
        self._head_cam = self._find(mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        self._follow_cam = mujoco.MjvCamera()
        self._follow_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._follow_cam.trackbodyid = self._find(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._follow_cam.distance, self._follow_cam.elevation = 0.9, -20.0
        self._channels = {}  # topic -> foxglove.Channel, created on first log
        foxglove.start_server(name="trotter")
        print("Foxglove: connect to ws://localhost:8765")

    def _find(self, obj_type, name: str) -> int:
        # mjlab prefixes entity element names (e.g. "robot/head_camera").
        for i in range(self._model.nbody if obj_type == mujoco.mjtObj.mjOBJ_BODY else self._model.ncam):
            full = mujoco.mj_id2name(self._model, obj_type, i) or ""
            if full == name or full.endswith("/" + name):
                return i
        raise KeyError(name)

    @staticmethod
    def _schema(value):
        """JSON schema for a value, so Foxglove can show the message's fields.

        Logging a plain dict creates a channel without a schema, which
        Foxglove can't expand. So each channel gets one, built from its first
        message.
        """
        if isinstance(value, dict):
            props = {k: FoxgloveStreamer._schema(v) for k, v in value.items()}
            return {"type": "object", "properties": props}
        if isinstance(value, list):
            return {"type": "array", "items": FoxgloveStreamer._schema(value[0]) if value else {}}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, (int, float)):
            return {"type": "number"}
        return {"type": "string"}

    def _log(self, topic: str, message: dict) -> None:
        if topic not in self._channels:
            self._channels[topic] = self._foxglove.Channel(
                topic, schema=self._schema(message), message_encoding="json"
            )
        self._channels[topic].log(message)

    def publish(self, action: torch.Tensor, command: list[float]) -> None:
        log, r = self._log, self._robot.data
        log("/joint_states", {
            "name": list(self._robot.joint_names),
            "position": r.joint_pos[0].tolist(),
            "velocity": r.joint_vel[0].tolist(),
            "effort": r.actuator_force[0].tolist(),
            "target": r.joint_pos_target[0].tolist(),
        })
        log("/policy/action", {"action": action[0].tolist()})
        log("/command", dict(zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command)))
        log("/base", {
            "position": r.root_link_pos_w[0].tolist(),
            "quaternion_wxyz": r.root_link_quat_w[0].tolist(),
            "lin_vel_body": r.root_link_lin_vel_b[0].tolist(),
            "ang_vel_body": r.root_link_ang_vel_b[0].tolist(),
            "projected_gravity": r.projected_gravity_b[0].tolist(),
        })
        log("/feet", {
            "contact": (self._contact.data.found[0, :2] > 0).tolist(),
            "height": r.site_pos_w[0, self._feet, 2].tolist(),
        })
        if self._step % self._camera_every == 0:
            self._publish_cameras()
        self._step += 1

    def _publish_cameras(self) -> None:
        from foxglove.messages import CompressedImage, Timestamp
        from PIL import Image

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, *self._size)
        sim = self._env.sim.data
        self._data.qpos[:] = sim.qpos[0].cpu().numpy()
        self._data.qvel[:] = sim.qvel[0].cpu().numpy()
        mujoco.mj_forward(self._model, self._data)
        now = time.time_ns()
        stamp = Timestamp(now // 1_000_000_000, now % 1_000_000_000)
        for topic, camera in (("/camera/head", self._head_cam), ("/camera/follow", self._follow_cam)):
            self._renderer.update_scene(self._data, camera=camera)
            buffer = io.BytesIO()
            Image.fromarray(self._renderer.render()).save(buffer, format="JPEG", quality=80)
            self._foxglove.log(topic, CompressedImage(
                timestamp=stamp, frame_id=topic.split("/")[-1], data=buffer.getvalue(), format="jpeg"
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="path to a model_<N>.pt file")
    parser.add_argument("--task", default="Trotter-Velocity-Flat")
    parser.add_argument("--foxglove", action="store_true", help="stream to Foxglove")
    parser.add_argument("--variant", help="sweep variant the checkpoint was trained with "
                        "(default: from the run folder name, e.g. ..._clock_w2)")
    args = parser.parse_args()

    # Variants like the gait clock add observations, so the env must match.
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    variant = args.variant or next(
        (v for v in OBSERVATION_VARIANTS if run_name.endswith("_" + v)), None)
    if variant:
        os.environ["TROTTER_VARIANT"] = variant
        print(f"variant: {variant}")

    # Same env as `play` (infinite episodes, no pushes), with one robot whose
    # command is never resampled: the keyboard is the only source.
    # Built here, not taken from the registry: the registry's config was made
    # at import time, before TROTTER_VARIANT was set above.
    env_cfg = trotter_flat_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    twist_cfg = env_cfg.commands["twist"]
    twist_cfg.resampling_time_range = (1e9, 1e9)
    twist_cfg.rel_standing_envs = 0.0

    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load only the actor (the critic isn't needed to run the policy).
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cpu")
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    streamer = FoxgloveStreamer(env) if args.foxglove else None
    keyboard = KeyboardCommand()
    twist = env.unwrapped.command_manager.get_term("twist")

    def keyboard_policy(obs):
        # Write the keyboard command into the env before every step. The
        # policy sees it in its next observation (one 20 ms step later).
        command = keyboard.get()
        twist.vel_command_b[:] = torch.tensor(command)
        action = policy(obs)
        if streamer is not None:
            streamer.publish(action, command)
        return action

    print(__doc__.split("\n\n")[2])  # The key list.
    try:
        NativeMujocoViewer(env, keyboard_policy).run()
    finally:
        keyboard.restore_terminal()
        env.close()


if __name__ == "__main__":
    main()
