"""Trotter velocity task: walk at whatever velocity the command asks for.

How mjlab tasks are built
=========================
An mjlab task ("environment") is described by one big config object,
`ManagerBasedRlEnvCfg`. It's split into sections, each handled by a *manager*
(the part of mjlab that runs that section every step):

    scene          what's in the world: terrain, robot, sensors
    actions        how the policy's output becomes motor targets
    commands       the goal given to the robot (here: a target velocity)
    observations   what the policy gets to see each step
    rewards        what earns/costs points (this is what defines "good walking")
    terminations   when an episode ends early (e.g. the robot fell)
    events         random things that happen: resets, pushes, randomized friction
    curriculum     rules that make the task harder as training improves

Each section is a dict of *terms*: {"name": SomeTermCfg(func=..., params=...)}.
Every term points at a function (`func`) that mjlab calls with `params`. That's
the whole pattern: to change behaviour you add, remove or tweak terms.

We don't write all of this from scratch. mjlab's built-in velocity task
(`make_velocity_env_cfg()`, used by G1, Go1, ANYmal...) already contains a good
set of terms. It leaves robot-specific blanks marked "# Set per-robot". This
file starts from that recipe, fills in the blanks for Trotter, and adjusts the
parts that don't fit a small robot.

Read mjlab's recipe next to this file:
    .venv/lib/python3.12/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from trotter_rl.tasks import mdp as trotter_mdp
from trotter_rl.robot.trotter_constants import (
    TROTTER_ACTION_SCALE,
    get_trotter_robot_cfg,
)

# Names defined in robot/trotter_constants.py.
BASE_BODY = "base_link"
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def trotter_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Trotter walking on flat ground.

    Args:
        play: False = config for training. True = config for watching a trained
            policy (no time limit, no random pushes, no sensor noise).
    """
    cfg = make_velocity_env_cfg()

    ##
    # Scene: flat ground + our robot + sensors.
    ##

    # The recipe defaults to rough terrain. Start with flat ground: much easier
    # to learn, and rough terrain can come later.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # "robot" is the name every term uses to refer to it (entity_name="robot").
    cfg.scene.entities = {"robot": get_trotter_robot_cfg()}

    # The recipe's "terrain_scan" sensor measures the terrain shape around the
    # robot. Useless on flat ground, so remove it (and its observation below).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    # "foot_height_scan" measures how high each foot is above the ground, by
    # shooting a few rays down from points around each foot site.
    for sensor in cfg.scene.sensors:
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES
            )
            # 6 rays in a 2 cm ring around each foot site (the soles are ~5x8 cm).
            sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=6)

    # Contact sensors report which geoms are touching what, and how hard.
    # A ContactMatch picks geoms: `primary` = the ones we care about,
    # `secondary` = what they touch.
    feet_ground = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",  # One total force per foot.
        num_slots=1,
        track_air_time=True,  # Also tracks how long each foot is in the air.
    )
    # Anything above the knees touching the ground = the robot has fallen.
    # The lower legs are excluded on purpose: their bottom ends are only ~6 mm
    # above the soles, and with no ankle the feet tilt during normal walking,
    # so a shin brushing the floor is NOT a fall.
    body_ground = ContactSensorCfg(
        name="body_ground_touch",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*_collision$",
            # Regex: the feet and every geom on the lower leg bodies.
            exclude=(*FOOT_GEOMS, r"^lower_leg_link.*"),
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    # The robot touching itself (in practice: one leg hitting the other).
    # "subtree" = a body and everything attached below it, so base_link's
    # subtree is the whole robot.
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors += (feet_ground, body_ground, self_collision)

    # Simulation buffer sizes. Copied from the recipe's flat-ground robots.
    cfg.sim.njmax = 300
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64

    ##
    # Actions: policy output -> servo targets.
    ##
    # target = default joint angle + scale * policy output (see trotter_constants).
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = TROTTER_ACTION_SCALE

    ##
    # Commands: the target velocity.
    ##
    # Every few seconds (resampling_time_range) each robot gets a new random
    # target, expressed in the robot's own frame:
    #   lin_vel_x = forward(+)/backward(-) [m/s]
    #   lin_vel_y = left(+)/right(-) [m/s]
    #   ang_vel_z = turn left(+)/right(-) [rad/s]
    # A fraction of robots get "stand still" (rel_standing_envs), so the same
    # policy also learns to stand.
    #
    # The recipe's speeds are for robots like G1 (legs ~0.8 m, up to 1 m/s).
    # Scaled down to what microduck (a similar-sized biped on mjlab 1.3.0)
    # trains with. Checkpoint 2400 already went 0.35-0.41 m/s forward when
    # commanded past its 0.3 training range. Turning: with +-0.5, ignoring a
    # turn command cost little, so turning wasn't learned (microduck: "ang
    # +-1.0 makes turning learnable"); policies have turned near 3 rad/s.
    # Fixed ranges, no widening during training (see the curriculum section).
    # +-0.7 / +-0.4 / +-3.0 (sweep of 2026-11-22) was mostly out of reach:
    # turning was ignored and the robot fell more.
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-0.4, 0.4)
    twist.ranges.lin_vel_y = (-0.3, 0.3)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    twist.rel_standing_envs = 0.1
    # The recipe also has "heading" mode (the command is a direction to face,
    # converted into a turn rate) and "forward only" robots (meant for stairs).
    # We want plain velocity commands, exactly like a joystick, so turn those off.
    twist.heading_command = False
    twist.ranges.heading = None  # mjlab requires this when heading mode is off.
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.viz.z_offset = 0.25  # Height of the command arrow above the base.

    ##
    # Observations: what the policy sees.
    ##
    # Two groups: "actor" = the policy (runs on the real robot), "critic" = a
    # second network used ONLY during training to judge how good a state is.
    # The critic may see things the real robot can't measure ("privileged"
    # information), because it never runs on the robot.
    #
    # The recipe's actor gets base_lin_vel (the robot's linear velocity). A
    # real IMU can't measure that, only estimate it badly, so we take it away
    # from the actor. The critic keeps it.
    del cfg.observations["actor"].terms["base_lin_vel"]
    # No terrain scan on flat ground (we removed the sensor above).
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    ##
    # Events: resets and randomization.
    ##
    # "Domain randomization": vary things that differ between simulation and
    # reality, so the policy learns to cope with a range rather than one exact
    # value.
    # Foot friction: random per robot, between 0.3 (slippery) and 1.2 (grippy).
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    # Base center of mass: shifted a little at random (real CoM differs from CAD).
    cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.015, 0.015),  # x [m]
        1: (-0.015, 0.015),  # y [m]
        2: (-0.015, 0.015),  # z [m]
    }
    # Note: these three are "startup" events: each robot gets its random value
    # once, when training starts. That's why they can't be ramped up by a
    # curriculum (they never run again). They're mild, so it's fine.

    # Random pushes every 1-3 s, by suddenly changing the base velocity. They
    # start at ZERO and are ramped up by the curriculum (see below), so the
    # policy first learns to walk and only then to handle pushes.
    cfg.events["push_robot"].params["velocity_range"] = {}

    ##
    # Rewards.
    ##
    # Each reward term returns a number per robot per step; it's multiplied by
    # `weight` and summed. Positive weight = encourage, negative = discourage.
    # The recipe's main terms:
    #   track_linear_velocity / track_angular_velocity (+2): match the command.
    #   upright (+1): keep the body level.
    #   pose (+1): stay close to the default pose (tight when standing, loose
    #       when walking; how loose is set per joint below).
    #   dof_pos_limits (-1): don't push joints past the soft limits.
    #   action_rate_l2 (-0.1): don't change targets abruptly (smooth motion).
    #   foot_clearance / foot_swing_height: lift feet to a target height.
    #   foot_slip: don't drag feet on the ground.
    #   soft_landing: don't stomp.

    # Velocity tracking. Each pays weight * exp(-error^2 / std^2): full reward
    # for perfect tracking, falling off as the error grows. `std` sets how
    # strict that is, and it MUST match the robot's speeds. The recipe's stds
    # (0.5 m/s, 0.71 rad/s) are sized for big robots: with them, Trotter
    # standing still while commanded 0.2 m/s still earned 85% of the reward
    # (61% for a 0.5 rad/s turn), so checkpoint 650 learned to just stand.
    # But 0.1 m/s is too strict: a 0.3 m/s error pays exp(-9) ~ 0, so the
    # reward is flat unless the robot is already nearly on target, and PPO
    # gets no hint which way to go. With the air_time reward paying for steps,
    # checkpoint 850 (commit d6d0f77) settled on marching in place (vx ~0 for
    # every command). Microduck's sqrt(0.1) = 0.32 m/s: marching in place at
    # a 0.3 m/s command still earns 41%, tracking it earns 100%. Standing
    # still no longer wins either, because air_time pays for stepping.
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    # ...but loose alone makes small commands not worth walking for: standing
    # still at 0.1 m/s keeps 90% of it (loses only 0.2), less than walking
    # costs in action_rate/foot/knee/pose penalties. V9 checkpoint 1000 still
    # walked at 0.1-0.3 m/s and backward; by 3000, with the action_rate ramp
    # at full weight, it froze below ~0.3 m/s, right where the break-even
    # lands. A second, strict kernel (std 0.1) makes small errors matter:
    # standing at 0.1 m/s now loses 0.2 + 0.63. The loose term still gives
    # the direction from far away, so this doesn't bring back marching in
    # place (checkpoint 850).
    cfg.rewards["track_linear_velocity_fine"] = RewardTermCfg(
        func=mdp.track_linear_velocity,
        weight=1.0,
        params={**cfg.rewards["track_linear_velocity"].params, "std": 0.1},
    )
    # The angular term is exp(-(yaw error^2 + roll/pitch rates^2) / std^2):
    # turning only pays while the base isn't wobbling, which also discourages
    # jitter. At std 0.25 (checkpoint 2400) normal step wobble already killed
    # it, so turning was barely learned; splitting it into yaw-only + a calm
    # reward (commits 38afd75, 31be9b2) let the policy jitter instead. The
    # recipe's 0.71 (also microduck's) with commands up to 1.0 rad/s: ignoring
    # a 1.0 rad/s turn earns 13%, and 0.5 rad/s of wobble still keeps 61%.
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # Upright: stronger and stricter than the recipe (1.0, std^2 0.2), as in
    # microduck: a 4 deg lean then costs ~0.19 per step instead of ~0.05, so
    # the body stays level instead of leaning into a shuffle.
    cfg.rewards["upright"].params["asset_cfg"].body_names = (BASE_BODY,)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Pose reward: allowed deviation (std, in rad) from the default pose.
    # Bigger = more freedom for that joint.
    # Weight halved (recipe: 1.0): at full weight, the pose reward pays almost
    # as much as tracking for simply not moving, which helped the policy settle
    # on standing still.
    cfg.rewards["pose"].weight = 0.5
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    # The pose reward is exp(-mean(error^2 / std^2)) over all joints, so the
    # stds decide which way of stepping is "cheap". With the old values
    # (thigh 0.3, hip roll 0.15, knee 0.35) a proper 5 cm step (thigh -30,
    # knee -60 deg) cost 0.43 per step, while swinging a straight leg 9 deg
    # outward cost only 0.08. Checkpoint 3350 learned exactly that: the hip
    # swung out 5-9 deg (up to 14) in every swing, knees nearly straight, a
    # stiff "broken leg" walk with ~2 cm steps. Now the knee and thigh are
    # free to bend (the high step costs 0.13) and hip roll is tight (the
    # outward swing costs 0.24), so lifting with the knee is the cheap way.
    cfg.rewards["pose"].params["std_walking"] = {
        r"joint_thigh_.*": 0.6,  # hip pitch: legs swing a lot when walking
        r"joint_hip_.*": 0.08,  # hip roll: small sway only, no swinging out
        r"joint_knee_.*": 1.0,  # knee: bends a lot to lift the foot
    }
    # "Running" kicks in above 1.5 m/s, which Trotter won't reach. It still has
    # to be set; use slightly looser values.
    cfg.rewards["pose"].params["std_running"] = {
        r"joint_thigh_.*": 0.8,
        r"joint_hip_.*": 0.12,
        r"joint_knee_.*": 1.2,
    }

    # Penalize the base spinning/wobbling (helps a biped stay calm).
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
    cfg.rewards["body_ang_vel"].weight = -0.05

    # Feet: how high to lift each foot during a step, measured at the foot
    # site (sole center). The recipe's 10 cm is for G1-sized legs. With the
    # crouched V9 leg (home: knee 1.2 cm ahead of the hip-foot line) the
    # human-way knee lifts the sole 3.1 cm at 40 deg and 5.7 cm at 60 deg of
    # bend (the bird way only 1.4 / 3.2 cm), foot kept under the hip. 5 cm is
    # a clear, natural step at ~55 deg, below the +67.5 deg knee limit.
    for name in ("foot_clearance", "foot_swing_height"):
        cfg.rewards[name].params["target_height"] = 0.05
    # The two terms work differently:
    #   foot_clearance: while a foot moves, cost = |height - target| * foot
    #       speed. Trotter's feet move slowly, so this stays small by nature.
    #   foot_swing_height: at each landing, cost = (peak height / target - 1)^2,
    #       e.g. a 2.5 cm step costs 0.25, a shuffle (no lift) costs 1.0.
    # The recipe's swing weight (-0.25) is tuned for big robots; at that weight
    # a shuffle costs Trotter almost nothing next to the tracking rewards.
    # -1.0 makes real steps clearly worth it (-3.0 and a 7 cm target made no
    # difference in the 2026-11-22 sweep). Both terms are only active when
    # moving is commanded, and the tracking rewards mean standing still to
    # avoid them costs far more.
    cfg.rewards["foot_swing_height"].weight = -1.0
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = FOOT_SITES

    # Legs hitting each other.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        # Counts contacts above 2 N (the recipe's 10 N is for 30 kg robots).
        params={"sensor_name": self_collision.name, "force_threshold": 2.0},
    )

    # Air time: +1 per step for each foot that's been in the air between
    # 0.125 and 0.4 s, i.e. a reward for real steps. This is the main gait
    # shaper in microduck (weight 3.0, window 0.125-0.3 s). It pays every
    # step the foot is up inside the window, so longer steps earn more, up to
    # the maximum. With microduck's 0.3 s max our feet stayed up ~0.27 s:
    # right at the cap. Max 0.4 s (sweep of 2026-11-22: 0.4 / 0.5 / 0.6) gave
    # the steadiest gait by 10k iterations: ~0.37 s steps, 2.5 steps/s, 4-5 cm
    # lift. The 0.125 s minimum must stay reachable: at 0.2 or
    # 0.3 s, young policies (~0.1 s steps) never earned anything from it and
    # settled into fast hopping. Only active when moving is
    # commanded. Uses the grounded variant: nothing while both feet are in
    # the air, so hopping doesn't pay double.
    cfg.rewards["air_time"].func = trotter_mdp.feet_air_time_grounded
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.4
    cfg.rewards["air_time"].params["command_threshold"] = 0.01

    # Recipe term we don't use: angular_momentum needs a sensor we don't have.
    cfg.rewards.pop("angular_momentum")

    ##
    # Terminations: when an episode ends early.
    ##
    # From the recipe: "time_out" (20 s episode) and "fell_over" (body tilted
    # more than 70 deg). We add: anything above the knees touching the ground.
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": body_ground.name, "force_threshold": 1.0},
    )
    # No terrain edges on an infinite plane.
    cfg.terminations.pop("out_of_terrain_bounds")

    ##
    # Curriculum.
    ##
    # Human-like knees: penalize bending them the bird way (negative angle;
    # see trotter_mdp.knee_bird_bend). On the V8 leg, policies swung the shin
    # forward by 7-18 deg in every swing, because that geometry lifted the
    # foot more per degree the bird way. The crouched V9 leg favors the human
    # way by itself; this keeps it that way. At -4, checkpoint 1250's gait pays
    # ~0.36 per step (about a quarter of what air_time pays), so stepping with
    # the human knee clearly wins; the human way costs nothing.
    cfg.rewards["knee_bird_bend"] = RewardTermCfg(
        func=trotter_mdp.knee_bird_bend,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"joint_knee_.*",))},
    )

    # Keep the soles flat on the ground in stance (see trotter_mdp.foot_flat).
    # Checkpoint 2750 stood on the outer edges at ~12 deg: 2 x 0.043 x 5 =
    # 0.43 per step, about a fifth of what tracking pays.
    cfg.rewards["foot_flat"] = RewardTermCfg(
        func=trotter_mdp.foot_flat,
        weight=-5.0,
        params={"sensor_name": feet_ground.name, "foot_geoms": FOOT_GEOMS},
    )

    # A curriculum changes the task during training, from easy to hard.
    # The recipe's two curricula don't fit: one raises the command speed up to
    # 3 m/s, the other makes rough terrain harder (we're on flat ground).
    # No speed curriculum: the command ranges set above stay fixed. Widening
    # them mid-run made tracking worse at every stage, and the policy never
    # recovered (best run so far: tracking peaked at 1500, dropped after the
    # widening at 2000, falls jumped after the one at 4000).
    cfg.curriculum.pop("command_vel")
    cfg.curriculum.pop("terrain_levels")

    # Ours: ramp up the pushes. One PPO iteration = 24 env steps, so
    # "step": 500 * 24 means "from iteration 500 on".
    full_push = {
        "x": (-0.3, 0.3),  # [m/s] Trotter walks at up to ~0.3 m/s
        "y": (-0.3, 0.3),
        "z": (-0.2, 0.2),
        "roll": (-0.3, 0.3),  # [rad/s]
        "pitch": (-0.3, 0.3),
        "yaw": (-0.5, 0.5),
    }

    def scaled(factor: float) -> dict[str, tuple[float, float]]:
        return {k: (lo * factor, hi * factor) for k, (lo, hi) in full_push.items()}

    # Smoothness: raise the action_rate_l2 penalty step by step (recipe: -0.1).
    # Policies so far chattered (actions flipping every step, ~20% of the
    # time at the torque limit), which would cook the real servos. Starting
    # strong instead makes a fresh policy afraid to move at all, so it ramps:
    # walking is learned first, smoothness is squeezed in after. Same ramp as
    # microduck (-1.0 by iteration 1500). Counted from the start of the run
    # (on --agent.resume the counter restarts at 0, so the ramp restarts too).
    cfg.curriculum["action_rate_ramp"] = CurriculumTermCfg(
        func=mdp.reward_curriculum,
        params={
            "reward_name": "action_rate_l2",
            "stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.3},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    cfg.curriculum["push_strength"] = CurriculumTermCfg(
        func=trotter_mdp.event_curriculum,
        params={
            "event_name": "push_robot",
            "stages": [
                {"step": 0, "params": {"velocity_range": {}}},  # learn to walk first
                {"step": 500 * 24, "params": {"velocity_range": scaled(1 / 3)}},
                {"step": 1500 * 24, "params": {"velocity_range": scaled(2 / 3)}},
                {"step": 3000 * 24, "params": {"velocity_range": full_push}},
            ],
        },
    )

    ##
    # Viewer.
    ##
    cfg.viewer.body_name = BASE_BODY  # The camera follows this body.
    cfg.viewer.distance = 1.0

    ##
    # Sweep variants: TROTTER_VARIANT=<name> applies one of these on top of
    # the config above (see train.slurm). Unset = the base config.
    ##
    variant = os.environ.get("TROTTER_VARIANT", "")
    if variant:
        _apply_variant(cfg, variant)

    ##
    # Play mode (watching a trained policy).
    ##
    if play:
        cfg.episode_length_s = int(1e9)  # Effectively no time limit.
        cfg.observations["actor"].enable_corruption = False  # No sensor noise.
        cfg.events.pop("push_robot")  # You can push it yourself in the viewer.
        cfg.curriculum = {}  # Nothing to ramp up (and push_robot is gone).

    return cfg





def _apply_variant(cfg: ManagerBasedRlEnvCfg, name: str) -> None:
    """Sweep variants, each a change from the base config. Past sweeps
    (air-time windows and ramps, a gait clock) are in the git history."""
    raise ValueError(f"unknown TROTTER_VARIANT {name!r}")
