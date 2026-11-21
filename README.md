# Trotter

A small 6-DOF biped (3 STS3215 servos per leg: hip pitch, hip roll, knee),
designed in Onshape and trained to walk with reinforcement learning in
[mjlab](https://github.com/mujocolab/mjlab).

- `trotter_description/`: the robot exported from Onshape (onshape-to-robot).
- `src/trotter_rl/robot/`: the robot as an mjlab entity (motors, collisions, IMU, cameras).
- `src/trotter_rl/tasks/`: the velocity-tracking task (rewards, curriculum, PPO settings).
- `scripts/play_keyboard.py`: drive a trained policy from the keyboard in the native
  MuJoCo viewer, optionally streaming to Foxglove (`--foxglove`).
- `train.slurm`: train on the cluster as a batch job.

```bash
uv run train Trotter-Velocity-Flat --env.scene.num-envs 8192   # train
sbatch train.slurm                                             # or on Slurm
uv run mjpython scripts/play_keyboard.py model_<N>.pt          # play (macOS: mjpython)
```

## Design notes for the next hardware revision

The robot isn't built yet. Findings from training that should shape the redesign:

**The knee bends "backwards" (bird-like) when walking, because the geometry
favors it.** With no ankle and the shin tilted forward at the zero pose, the
knee is the only way to shorten the leg, and bending it the bird way (negative)
lifts the foot much more per degree than the human way (positive). Measured
with the foot kept under the hip:

| Knee bend | Foot lift, human direction (+) | Foot lift, bird direction (-) |
|---|---|---|
| 20 deg | 0.5 cm | 1.1 cm |
| 40 deg | 1.5 cm | 2.8 cm |
| 60 deg (current human-side limit) | 3.4 cm | 5.3 cm |

So human-style steps can't get past ~3.4 cm, and policies pick the bird knee.
Options, best first:

1. **Add an ankle pitch servo per leg.** Lets the knee bend the human way while
   the foot stays flat in stance, lifts the toes for clearance, and helps push
   recovery. Costs 2 servos and some weight low on the legs (microduck and
   Disney's BD-X have ankles).
2. **More human-side knee range** (~90-100 deg instead of 60; check the shin
   doesn't hit the thigh), optionally with less forward shin tilt at zero. Then
   block or penalize the bird direction in simulation.
3. Keep the design: bird knee with 3-5 cm steps, or force the human knee in
   simulation and accept ~3 cm steps.

**Speed.** Expect a brisk walk of ~0.6-0.8 m/s at most: the stride is short
(~30 cm at +-30 deg thigh swing) and the STS3215 is slow (~4.7 rad/s no-load at
7.4 V, modeled in simulation with a DC motor torque-speed curve). Running needs a
flight phase the 1.9 N*m servos likely can't push off for.

**Mass.** The simulated robot is 1.31 kg without the battery, Raspberry Pi, IMU
and converters (~+200-250 g). Add their real masses in Onshape (battery low and
centered) and retrain before trying the real robot.
