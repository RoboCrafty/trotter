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

## Design notes

### Knee geometry (V8 -> V9)

**V8:** thigh vertical, shin's lower end tilted 7 deg forward at the zero pose, so
the knee sat 0.9 cm *behind* the hip-foot line (pre-bent the bird way). With no
ankle, bending the knee the bird way (negative) then lifted the foot far more
per degree than the human way, and policies walked bird-kneed. Foot lift, foot
kept under the hip:

| Knee bend | V8 human (+) | V8 bird (-) | **V9 human (+)** | V9 bird (-) |
|---|---|---|---|---|
| 20 deg | 0.5 cm | 1.1 cm | **1.3 cm** | 0.4 cm |
| 40 deg | 1.5 cm | 2.8 cm | **3.1 cm** | 1.4 cm |
| 60 deg | 3.4 cm (limit) | 5.3 cm | **5.7 cm** | 3.2 cm |

**V9:** crouched home pose (thigh +10 deg, knee +22.5 deg, foot mounted at 12.5 deg
so the sole is flat), exported as the zero pose. The knee now sits 1.2 cm *in
front of* the hip-foot line, the human way lifts ~2x more per degree, and the
human-side knee range from home is +67.5 deg. The CoM at home is 2.5 cm in front
of the heel edge: stable, but toward the back.

Still open:

- The left knee's limits (-87.5..+42.5 deg from home) don't match the right
  (-112.5..+67.5); the left mate likely still has old limits.
- An ankle pitch servo per leg would still help most for natural walking (flat
  foot with a bent knee, toe clearance, push-off) at the cost of 2 servos.
- A slightly curved (rocker) sole works like a passive ankle without a servo.

**Speed.** Expect a brisk walk of ~0.6-0.8 m/s at most: the stride is short
(~30 cm at +-30 deg thigh swing) and the STS3215 is slow (~4.7 rad/s no-load at
7.4 V, modeled in simulation with a DC motor torque-speed curve). Running needs a
flight phase the 1.9 N*m servos likely can't push off for.

**Mass.** The simulated robot is 1.29 kg (V9) without the battery, Raspberry Pi, IMU
and converters (~+200-250 g). Add their real masses in Onshape (battery low and
centered) and retrain before trying the real robot.
