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

## Training results

**Current base (tag `base-max04`):** the config here. Its 10k-iteration run
(`2026-11-22_10-50-56_max04`, `model_9900.pt`) walks with a steady rhythm: ~2.5
steps/s, each foot ~0.37 s in the air (+-0.01-0.03 s), 4-5 cm lift, accurate at
slow forward speeds. Weak at sideways (0.04 of 0.2 m/s) and turning (0.37 of
1.0 rad/s). Its gait only became steady late (irregular at 4300, steady by
9900), so judge a promising config at the end, not at 1500.

Lessons from the sweeps (details in the git history):

- Keep the command ranges fixed and reachable. Widening them during training,
  or ranges past what the servos allow (x +-0.7, yaw +-3.0), made tracking
  worse every time.
- Air-time window: raise only the maximum. A minimum above the steps a young
  policy takes (0.2 or 0.3 s, or a ramp that raises it later) pays nothing,
  and the policy settles into fast hopping or standing.
- A gait clock (phase observation + contact-timing reward) was tried three
  ways; each time the policy found a way to buzz its feet a few mm at ~20
  steps/s that satisfied it. It would need a cost per lift-off or a minimum
  swing height before a contact counts.

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
