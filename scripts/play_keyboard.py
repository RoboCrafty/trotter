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
"""

import argparse
import os
import select
import sys
import termios
import threading
import tty
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

import trotter_rl.tasks  # noqa: F401  (registers Trotter-Velocity-Flat)

# Change per keypress, and the largest command allowed. The limits go past
# the training ranges (0.3 / 0.2 / 0.5) on purpose, to see where it breaks.
STEP = {"x": 0.05, "y": 0.05, "yaw": 0.1}
LIMIT = {"x": 0.6, "y": 0.5, "yaw": 1.2}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="path to a model_<N>.pt file")
    parser.add_argument("--task", default="Trotter-Velocity-Flat")
    args = parser.parse_args()

    # Same env as `play` (infinite episodes, no pushes), with one robot whose
    # command is never resampled: the keyboard is the only source.
    env_cfg = load_env_cfg(args.task, play=True)
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

    keyboard = KeyboardCommand()
    twist = env.unwrapped.command_manager.get_term("twist")

    def keyboard_policy(obs):
        # Write the keyboard command into the env before every step. The
        # policy sees it in its next observation (one 20 ms step later).
        twist.vel_command_b[:] = torch.tensor(keyboard.get())
        return policy(obs)

    print(__doc__.split("\n\n")[2])  # The key list.
    try:
        NativeMujocoViewer(env, keyboard_policy).run()
    finally:
        keyboard.restore_terminal()
        env.close()


if __name__ == "__main__":
    main()
