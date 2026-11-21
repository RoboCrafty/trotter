"""Trotter robot definition for mjlab.

This file answers one question: "what is the robot?" It does NOT say anything
about walking, rewards or training. mjlab splits those concerns:

    robot/trotter_constants.py   -> the robot (this file): model, motors, start pose
    tasks/...                    -> the task: what the robot should do, rewards, etc.

In mjlab, anything that lives in the simulated world (a robot, a box, the
terrain) is called an *Entity*. You describe an entity with an `EntityCfg`
("Cfg" = config: a plain dataclass holding settings, no logic). mjlab later
builds the real `Entity` object from it, once per simulated world, and runs
thousands of copies in parallel on the GPU.

Run this file directly to look at the robot as mjlab sees it:

    uv run python -m trotter_rl.robot.trotter_constants
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF (the MuJoCo XML exported by onshape-to-robot).
##

# parents[3] walks up from src/trotter_rl/robot/ to the repo root.
TROTTER_XML: Path = (
    Path(__file__).resolve().parents[3] / "trotter_description" / "robot.xml"
)
assert TROTTER_XML.exists(), f"Robot XML not found: {TROTTER_XML}"


def get_spec() -> mujoco.MjSpec:
    """Load the robot XML as an MjSpec.

    MjSpec is MuJoCo's *editable* model: like the XML file, but as Python objects
    you can modify before compiling. mjlab takes this spec, adds motors, the
    terrain, sensors etc. to it, and only then compiles it into a model.
    """
    spec = mujoco.MjSpec.from_file(str(TROTTER_XML))

    # onshape-to-robot writes its own <position> actuators (kp=50) into the XML.
    # We want mjlab to own the motors (defined below, in Python), so we delete
    # the XML ones. Otherwise each joint would end up with two motors.
    for actuator in list(spec.actuators):
        spec.delete(actuator)

    _reframe_base(spec)
    _add_imu(spec)
    _name_collision_geoms(spec)
    _add_foot_sites(spec)
    return spec


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, vec, quat)
    return out


# Rotation from the CAD base frame to the frame mjlab expects: -90 deg about Z.
# (MuJoCo quaternions are [w, x, y, z].)
_CAD_TO_MJLAB = np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)])


def _reframe_base(spec: mujoco.MjSpec) -> None:
    """Move base_link's coordinate frame so +X points forward, with its origin at
    the base's center of mass. The robot itself doesn't move at all.

    Why: mjlab's velocity task measures the robot's speed in the base body's own
    frame and treats +X as "forward", +Y as "left", +Z as "up" (the usual robot
    convention). The exported frame instead comes from the Onshape assembly
    origin: it sits at a corner of the CAD and has the robot facing -Y (the eyes
    are on the -Y side, the left leg on +X). Without this fix, a "walk forward"
    command would reward walking sideways.

    How: a body's frame is just the coordinate system its children (geoms, child
    bodies, inertia) are described in. We pick a new frame and re-express every
    child in it, so everything stays exactly where it was in the world.

    TODO: once base_link is placed in Onshape facing +X with its origin in the
    torso, delete this function (and its call in get_spec).
    """
    base = spec.body("base_link")
    # New frame, described in the old frame: origin at the base's center of mass
    # (read from the XML, so it follows CAD changes), rotated -90 deg about Z.
    origin = np.array(base.ipos)
    rot = _CAD_TO_MJLAB
    rot_inv = np.array([rot[0], -rot[1], -rot[2], -rot[3]])  # inverse rotation

    # Re-express every direct child (geoms, sites, child bodies) in the new frame:
    # subtract the new origin, then undo the rotation.
    for child in [*base.geoms, *base.sites, *base.bodies]:
        child.pos = _rotate(rot_inv, np.array(child.pos) - origin)
        child.quat = _quat_mul(rot_inv, np.array(child.quat))

    # The inertia tensor also has to be re-expressed in the new axes. The CoM
    # is now exactly at the new origin. fullinertia order: xx, yy, zz, xy, xz, yz.
    ixx, iyy, izz, ixy, ixz, iyz = base.fullinertia
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, rot)
    rot_mat = rot_mat.reshape(3, 3)
    inertia = rot_mat.T @ inertia @ rot_mat
    base.fullinertia = [
        inertia[0, 0], inertia[1, 1], inertia[2, 2],
        inertia[0, 1], inertia[0, 2], inertia[1, 2],
    ]
    base.ipos = [0.0, 0.0, 0.0]

    # Finally, place the body itself at the new frame (in world coordinates).
    base_quat = np.array(base.quat)
    base.pos = np.array(base.pos) + _rotate(base_quat, origin)
    base.quat = _quat_mul(base_quat, rot)


def _add_imu(spec: mujoco.MjSpec) -> None:
    """Add an IMU (a site plus three sensors) to the base.

    A *site* is an invisible, massless point attached to a body: a named spot
    you can put sensors on. The IMU sits at the base's center of mass (the base
    frame's origin after _reframe_base), with its axes aligned to the base
    frame: +X forward, +Y left, +Z up.

    mjlab's velocity task reads these sensors by name ("imu_ang_vel",
    "imu_lin_vel"), so the names must match exactly.

    TODO: when the IMU's real mounting point is known, add a mate connector in
    Onshape named "frame_imu" (onshape-to-robot then exports a site called
    "imu") and this function will use that one instead of making its own.
    Mind its axes: they must match the base frame described above.
    """
    if not any(site.name == "imu" for site in spec.sites):
        spec.body("base_link").add_site(name="imu", pos=[0, 0, 0], group=5)

    # What a real IMU measures (in the IMU's own axes):
    spec.add_sensor(  # angular velocity [rad/s], i.e. the gyroscope
        name="imu_ang_vel",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu",
    )
    spec.add_sensor(  # linear acceleration incl. gravity [m/s^2]
        name="imu_lin_acc",
        type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu",
    )
    # Linear velocity [m/s]. A real IMU can NOT measure this directly (it would
    # have to be estimated). mjlab's default observations still use it, so we
    # provide it; whether the real robot can get it is a decision for step 3.
    spec.add_sensor(
        name="imu_lin_vel",
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="imu",
    )


def _name_collision_geoms(spec: mujoco.MjSpec) -> None:
    """Give every collision geom a name, so the collision config can find it.

    In the XML, each CAD part appears twice: a *visual* geom (what you see,
    class="visual", group 2) and a *collision* geom (what physics uses,
    class="collision", group 3), both using the same mesh. onshape-to-robot
    leaves them unnamed, but mjlab's CollisionCfg selects geoms BY NAME (regex),
    so we name the collision ones here:

        left_foot_collision, right_foot_collision    <- the two feet
        <body>_<mesh>_<n>_collision                  <- everything else,
                                                        e.g. base_link_main_0_collision
    """
    counts: dict[str, int] = {}
    for geom in spec.geoms:
        if geom.classname.name != "collision":
            continue
        body = geom.parent
        if geom.meshname == "foot_link":
            # Which foot? The lower leg body carries joint_knee_left/right.
            side = "left" if any("left" in j.name for j in body.joints) else "right"
            geom.name = f"{side}_foot_collision"
        else:
            prefix = f"{body.name}_{geom.meshname}"
            # A body can contain the same mesh twice (e.g. two servos in the
            # base), so number them to keep names unique.
            n = counts.get(prefix, 0)
            counts[prefix] = n + 1
            geom.name = f"{prefix}_{n}_collision"


def _add_foot_sites(spec: mujoco.MjSpec) -> None:
    """Add a site at the center of each sole: "left_foot" and "right_foot".

    The velocity task uses these points to measure foot height (for the "lift
    your feet" rewards) and foot sliding speed (the "don't slip" reward).

    We don't hardcode positions; we compute them from the foot mesh, so they
    follow CAD changes. Mesh vertices only exist in the *compiled* model, so we
    compile once here, find the sole, then add the sites to the spec.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    # Kinematics only (no physics): computes where everything is in the XML's
    # default pose, where all joints are 0 and the soles are flat on the ground.
    mujoco.mj_kinematics(model, data)

    for side in ("left", "right"):
        if any(site.name == f"{side}_foot" for site in spec.sites):
            continue  # Already defined (e.g. by a frame_ mate in Onshape).
        geom = model.geom(f"{side}_foot_collision")
        mesh = geom.dataid[0]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        # Foot mesh vertices in world coordinates.
        rot = data.geom_xmat[geom.id].reshape(3, 3)
        verts = model.mesh_vert[start : start + count] @ rot.T + data.geom_xpos[geom.id]
        # The sole = the vertices within 2 mm of the lowest point. Site goes at
        # their horizontal center, at the lowest height.
        z_min = verts[:, 2].min()
        sole = verts[verts[:, 2] < z_min + 0.002]
        point_world = np.array([*sole[:, :2].mean(axis=0), z_min])
        # Sites are defined relative to their body, so convert world -> body.
        body = geom.bodyid[0]
        body_rot = data.xmat[body].reshape(3, 3)
        point_body = body_rot.T @ (point_world - data.xpos[body])
        spec.body(model.body(body).name).add_site(
            name=f"{side}_foot", pos=point_body, group=5
        )


##
# Joint names.
##
# The names come from the Onshape mate names (dof_<name>). Note they don't match
# the usual anatomy names, so here is what each one actually does:
#
#   joint_thigh_{left,right}  axis = left-right (Y)   -> hip PITCH (swing leg fwd/back)
#   joint_hip_{left,right}    axis = forward (X)      -> hip ROLL  (move leg sideways)
#   joint_knee_{left,right}   axis = left-right (Y)   -> knee pitch
#
# mjlab selects joints with *regular expressions* (regex), so ".*" means
# "anything": "joint_knee_.*" matches both joint_knee_left and joint_knee_right.

##
# Actuators (motors).
##
# Each joint is driven by an STS3215 servo. A hobby servo is a position
# controller: you send a target angle and its internal loop pushes toward it.
# The closest MuJoCo equivalent is a PD controller:
#
#     torque = stiffness * (target_angle - angle) - damping * velocity
#
# clipped to +/- effort_limit. `BuiltinPositionActuatorCfg` creates exactly that
# using MuJoCo's own <position> actuator.
#
# The numbers below are reasonable first guesses, NOT measured values. Assumes
# the 7.4V STS3215 (stall torque ~1.9 N*m). Getting these closer to the real
# servo (system identification) matters a lot for sim-to-real later; for
# learning to walk in simulation they're good enough.

# Max torque [N*m] = stall torque of the 7.4V STS3215 (19.5 kg*cm). If yours
# is the 12V version, use ~2.9.
# For scale: standing on ONE leg, the hip roll joint needs ~1 N*m just to hold
# the body up (~1 kg of robot, ~10 cm sideways from the hip axis).
STS3215_EFFORT_LIMIT = 1.9
# Proportional gain [N*m/rad]. Real servos hold position stiffly; at 30, holding
# that 1 N*m one-leg load costs ~2 deg of sag. The servo hits full torque once
# it's ~0.06 rad (~3.6 deg) away from its target.
STS3215_STIFFNESS = 30.0
# Derivative gain [N*m*s/rad]. Damps oscillation around the target.
STS3215_DAMPING = 1.0
# "Armature" [kg*m^2] is extra rotational inertia added to the joint. It models
# the motor rotor spinning fast behind the gearbox, which the leg "feels" as
# extra inertia. It also makes the simulation more numerically stable.
STS3215_ARMATURE = 0.005
# Constant friction torque [N*m] opposing motion (gearbox friction).
STS3215_FRICTIONLOSS = 0.05

TROTTER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    # All six joints use the same servo, so one config covers all of them.
    target_names_expr=("joint_.*",),
    stiffness=STS3215_STIFFNESS,
    damping=STS3215_DAMPING,
    effort_limit=STS3215_EFFORT_LIMIT,
    armature=STS3215_ARMATURE,
    frictionloss=STS3215_FRICTIONLOSS,
)

##
# Initial state (where each episode starts).
##
# The all-zero pose is Trotter's natural standing pose: the knee is already bent
# ~18 deg in the CAD, and without an ankle, any other flat-footed pose shifts the
# feet forward/back away from under the hips.
#
# `pos` is where the base frame's origin starts, i.e. the base's center of mass
# (see _reframe_base). In the zero pose it sits 0.343 m above the lowest foot
# point, so 0.35 m starts the robot with its feet ~6 mm above the floor.

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.35),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

##
# Collisions.
##
# Same approach as ANYmal C: every collision geom collides with everything
# (the floor, and the robot's other parts), all with the same contact settings.
# Unlike ANYmal (whose XML was hand-made with named boxes and capsules), our
# collision shapes are the real CAD meshes, named by _name_collision_geoms().
#
# One thing to know: MuJoCo collides each mesh as its CONVEX HULL (the part
# "shrink-wrapped"; holes and notches are filled in). Trotter's parts are close
# to convex, and with this setup the standing pose has no fake contacts, and
# sweeping each joint through its full range only produces real leg-on-leg
# contacts. MuJoCo automatically skips directly connected bodies (parent/child,
# e.g. base <-> hip), whose meshes touch at the joint.

TROTTER_COLLISION = CollisionCfg(
    # Which geoms this config applies to. Geoms NOT matched get collisions
    # switched off (disable_other_geoms=True by default), which keeps the
    # visual copies of each part out of the physics.
    geom_names_expr=(r".*_collision$",),
    # condim = how many directions a contact can push in. 3 = push apart
    # (normal force) + sliding friction, which the feet need to grip.
    condim=3,
    # When two geoms touch, MuJoCo takes the contact settings (like friction)
    # from the one with higher priority. Priority 1 means the robot's settings
    # win over the floor's.
    priority=1,
    # Sliding friction coefficient. 1.0 is grippy, like rubber on a floor. The
    # task will randomize foot friction so the policy doesn't rely on one value.
    friction=(1.0,),
)

##
# Action scale.
##
# The policy (neural network) outputs one number per joint, roughly in the range
# [-1, 1]. mjlab turns that into a servo target like this:
#
#     target_angle = default_angle + action_scale * policy_output
#
# so 0.5 means "a policy output of 1.0 moves the joint 0.5 rad (~29 deg) away
# from the standing pose". Too small and the robot can't take real steps; too
# big and early random actions are violent. Used by the task config (next step).
TROTTER_ACTION_SCALE = 0.5

##
# Final config.
##

TROTTER_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(TROTTER_ACTUATOR_CFG,),
    # Treat 90% of each joint's range as the "soft" limit. Rewards can penalise
    # the policy for going past it, so it learns to stay away from the hard
    # stops (where real servos/parts get damaged).
    soft_joint_pos_limit_factor=0.9,
)


def get_trotter_robot_cfg() -> EntityCfg:
    """Return a fresh Trotter config.

    A function instead of a global variable: configs are mutable dataclasses,
    and tasks tweak them (e.g. change the start pose). Returning a new object
    each time means one task's tweaks can't leak into another.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        spec_fn=get_spec,
        articulation=TROTTER_ARTICULATION,
        collisions=(TROTTER_COLLISION,),
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Build the entity exactly as mjlab would, then open it in MuJoCo's viewer.
    # The motors in the viewer's Control panel are the ones defined above.
    robot = Entity(get_trotter_robot_cfg())
    spec = robot.spec

    # The entity is ONLY the robot: no floor, no lights (in training, mjlab's
    # scene adds those). For this preview we add a floor and a light ourselves.
    spec.worldbody.add_light(
        pos=[0, 0, 2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.3, 0.4, 0.5, 1]
    )

    model = spec.compile()

    # mjlab's actuators have no control range (in training, mjlab computes the
    # targets itself). The viewer then shows a tiny default slider range, so for
    # this preview we give each slider its joint's range (+/-pi if unlimited).
    for i in range(model.nu):
        joint = model.actuator_trnid[i, 0]
        limited = model.jnt_limited[joint]
        model.actuator_ctrlrange[i] = model.jnt_range[joint] if limited else (-3.14, 3.14)
        model.actuator_ctrllimited[i] = 1

    data = mujoco.MjData(model)
    # Start from INIT_STATE (the same start pose training uses).
    data.qpos[:3] = INIT_STATE.pos
    data.qpos[3:7] = INIT_STATE.rot

    print(f"bodies: {model.nbody}  joints: {model.njnt}  actuators: {model.nu}")
    for i in range(model.nu):
        print(f"  actuator {i}: {model.actuator(i).name}")
    # `launch` (blocking) works with plain python on macOS; only
    # `launch_passive` (non-blocking, used by scripts that step the sim
    # themselves) needs mjpython.
    viewer.launch(model, data)
