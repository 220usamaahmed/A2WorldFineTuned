"""Pick-and-place episodes: one object from a drawer crate or a shelf into the container on the table.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python pick_place.py --headless [--episodes 6] [--seed 0]

Scene (layout.py): table with the arm at its centre; three drawer housings with sliding crates on the -y side; a
two-level wooden shelf on the +y side; Isaac's shallow grey tray (container_h20) on top of the drawer housings, which
form a shop counter (the customer would stand beyond them; layout.CONTAINER_POS). Short YCB
objects stand in the crates (one per crate), taller ones on the shelf (two per level, upright). Gripper: the simplified
physics (ur5e_env.use_simple_gripper).

Each episode: reset (crates closed, every object in a fresh random pose: crates -> a random graspable resting pose,
shelf -> upright, random spot and yaw), choose the source (alternating drawer / shelf) and a target object there, then
  1. drawer only: hook the crate's rim bar and pull the crate out;
  2. grasp: sampled antipodal grasps at the object's pose, score-weighted random order; a grasp is accepted only if the
     object can also be got out from it (planning/tasks.plan_extraction: lift 3 cm, attach, then up out of the crate or
     straight out of the shelf towards the arm);
  3. place: drop poses in the reachable half of the container, object in the orientation it rested in at its source
     (several positions x yaws), reached like a grasp with a straight vertical descent; release; back off; home.
If no grasp of the chosen object passes, the episode tries the other objects of that source. Outputs per episode: an
mp4 (two overview cameras + wrist) and a record in episodes.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

# ======================================================================================================================
# 1. COMMAND-LINE OPTIONS AND ISAAC SIM START-UP
#   Isaac Sim has to be running before anything from isaaclab (other than the AppLauncher) can be imported.
# ======================================================================================================================
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--speed", type=float, default=0.5, help="cuRobo time dilation (1 = time-optimal at the joint limits)")
ap.add_argument("--grip-force", type=float, default=20.0)
ap.add_argument("--friction", type=float, default=1.0)
ap.add_argument("--pull", type=float, default=0.15)
ap.add_argument("--hook-gap", type=float, default=0.040)
ap.add_argument("--latch-stiffness", type=float, default=3000.0)
ap.add_argument("--slide-damping", type=float, default=15.0)
ap.add_argument("--candidates", type=int, default=40, help="sampled grasps handed to the planner per pick")
ap.add_argument("--drawer-objects", default="", help="3 YCB keys, one per crate (default: random from the short set)")
ap.add_argument("--shelf-objects", default="", help="4 YCB keys, 2 per level (default: random from the tall set)")
ap.add_argument("--out", type=Path, default=paths.EPISODES)
ap.add_argument("--tag", default="")
ap.add_argument("--stills", action="store_true",
                help="only render high-resolution stills of the scene (crates closed, then pulled out) to --out, then exit")
ap.add_argument("--stills-size", default="1920x1080", help="WxH of the stills")
ap.add_argument("--stills-open", default="0,1,2", help="crates (0 = -x .. 2 = +x) pulled out for the 'open' stills")
ap.add_argument("--no-video", action="store_true",
                help="no episode cameras or videos (saves GPU memory, e.g. when watching via --livestream 2)")
ap.add_argument("--stills-cams", default="", help="comma-separated subset of the STILLS cameras (default: all; each costs GPU memory)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


# on a crash, exit at once (otherwise Isaac can hang in shutdown and keep its GPU memory)
def _die(exc_type, exc, tb):
    """An uncaught exception must not leave Isaac hanging in shutdown with its GPU memory (a crashed run sat on 12 GB)."""
    import os
    import traceback

    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = _die

# ======================================================================================================================
# 2. IMPORTS THAT NEED THE RUNNING APP, AND THIS PROJECT'S MODULES
#   layout = scene geometry (numbers), looks = textures/lights, ur5e_env = base env (robot, wrist camera, actions),
#   grasps / planner / tasks = grasp sampling and cuRobo motion planning (planning/).
# ======================================================================================================================
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "planning"))
import layout as L  # noqa: E402
import looks as LK  # noqa: E402
import robot_defs as n  # noqa: E402
import ur5e_env as E  # noqa: E402
from grasps import cached_grasps, collider_mesh, graspable_stable_poses, world_filter  # noqa: E402
from planner import ArmPlanner, Plan, pregrasp  # noqa: E402
from tasks import T_of, clear_container_walls, drop_poses, plan_extraction, pose7  # noqa: E402

# ======================================================================================================================
# 3. CONSTANTS: asset locations, which YCB objects may go where, gripper orientation for hooking a crate
# ======================================================================================================================
ISAAC, PHYS, MESHES = paths.ISAAC, paths.YCB_PHYSICS, paths.YCB_MESHES
BIN_USD = paths.USD / "pullbin.usd"
SHORT = ["005_tomato_soup_can", "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "061_foam_brick", "011_banana"]
TALL = ["006_mustard_bottle", "004_sugar_box", "003_cracker_box", "021_bleach_cleanser"]
Q_HOOK = np.array([*R.from_matrix(np.stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]], 1)).as_quat()[[3, 0, 1, 2]]])  # fingers along y
to_base = np.array([0.0, 0.0, L.TABLE_Z])
rng = np.random.default_rng(args.seed)

# ======================================================================================================================
# 4. OBJECTS: 3 in the crates (one each), 4 on the shelf
#   Pick the YCB objects (by --seed), load their meshes and pre-computed grasps (cached on disk),
#   and define random_layout(): a random start pose for every object (also used at each episode reset).
# ======================================================================================================================
drawer_keys = args.drawer_objects.split(",") if args.drawer_objects else list(rng.choice(SHORT, 3, replace=False))
shelf_keys = args.shelf_objects.split(",") if args.shelf_objects else list(rng.choice(TALL, 4, replace=len(TALL) < 4))
KEYS = drawer_keys + shelf_keys
SRC = ["drawer"] * 3 + ["shelf"] * 4
summ = json.loads((MESHES / "summary.json").read_text())
INFO = {}
for k in set(KEYS):
    m = collider_mesh(k, summ[k]["obj"])
    g = cached_grasps(k, summ[k]["obj"], n_pairs=500, n_roll=16)
    poses = graspable_stable_poses(m, g)
    assert poses, f"{k}: not graspable by the 2F-85 in any resting pose"
    heights = [float((T[:3, :3] @ m.vertices.T + T[:3, 3:4])[2].max()) for T, _, _ in poses]
    G = np.array([x.T for x in g])
    # stronger preference for grasp axes through the centre of mass than the sampler's score has (a mustard bottle
    # grasped by its 20 mm cap was lost on the way to the container)
    c = m.centroid if m.is_watertight else m.bounding_box.centroid
    off = np.linalg.norm(np.cross(c[None] - G[:, :3, 3], G[:, :3, 0]), axis=1)
    INFO[k] = dict(mesh=m, grasps=g, G=G, widths=np.array([x.width for x in g]),
                   scores=np.array([x.score for x in g]) - 4.0 * off, poses=poses, upright=poses[int(np.argmax(heights))][0])
print("drawer objects:", drawer_keys, "| shelf objects:", shelf_keys, flush=True)


def footprint(k, T):
    """Half the diagonal of the object's horizontal extent in rest pose T."""
    v = INFO[k]["mesh"].vertices @ T[:3, :3].T
    return 0.5 * float(np.linalg.norm(np.ptp(v[:, :2], axis=0)))


def random_layout():
    """World poses (pose7) for all 7 objects: crates (closed) and shelf."""
    out = []
    for i in range(3):  # one per crate, under the roof opening once the crate is pulled out ~0.12-0.15 m (layout.CRATE_OBJ_XC)
        k = KEYS[i]
        Ts, ps, _ = zip(*INFO[k]["poses"])
        T = np.array(Ts[rng.choice(len(Ts), p=np.array(ps) / sum(ps))])
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
        T = Rz @ T
        T[:3, 3] += L.crate_to_world(i, rng.uniform(*L.CRATE_OBJ_XC), rng.uniform(*L.CRATE_OBJ_YC), L.FLOOR_TOP + 0.003)
        out.append(pose7(T))
    for lvl in (0, 1):  # two per level, upright, on non-adjacent spots
        spots = [L.SHELF_SPOTS_X[j] for j in [(0, 2), (1, 3), (0, 3)][rng.integers(3)]]
        for j, sx in enumerate(spots):
            k = KEYS[3 + 2 * lvl + j]
            Rz = np.eye(4)
            Rz[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
            T = Rz @ INFO[k]["upright"]
            r = footprint(k, T)
            x = np.clip(sx + rng.uniform(-0.02, 0.02), L.SHELF_X[0] + 0.02 + r + 0.01, L.SHELF_X[1] - 0.02 - r - 0.01)
            y = np.clip(rng.uniform(*L.SHELF_SPOTS_Y), L.SHELF_FRONT_Y + r + 0.01, L.SHELF_FRONT_Y + L.SHELF_DEPTH - r - 0.01)
            T[:3, 3] += [x, y, L.SHELF_LEVELS[lvl] + 0.003]
            out.append(pose7(T))
    return out


layout0 = random_layout()  # start poses for the first episode (the scene is built with these)


# ======================================================================================================================
# 5. SCENE: what Isaac builds
#   Factory functions for the parts, the two overview cameras, then SceneCfg = the base scene of ur5e_env
#   (ground, lights, robot, wrist camera) plus desk, drawers, shelf, counter, container and objects.
#   All sizes and positions come from layout.py; the look (wood, paint, room) is added later by looks.py.
# ======================================================================================================================
# a drawer: housing + sliding crate (setup/build_pullbin.py), an articulation with one prismatic joint 'crate_slide'
def bin_cfg(i):
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/bin{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(BIN_USD), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(L.BIN_X[i], L.ORIGIN_Y, 0.0), rot=L.HOUSING_QUAT, joint_pos={"crate_slide": 0.0}),
        actuators={"slide": ImplicitActuatorCfg(joint_names_expr=["crate_slide"], stiffness=0.0, damping=args.slide_damping, effort_limit_sim=500.0)},
    )


# a YCB object as a free rigid body (physics version from setup/make_ycb_physics.py)
def obj_cfg(i):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(PHYS / f"{KEYS[i]}.usd"), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(map(float, layout0[i][:3])), rot=tuple(map(float, layout0[i][3:]))),
    )


# a static box with a collider (shelf boards and walls, desk, counter ends)
def board_cfg(name, lo, hi, prefix="shelf"):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prefix}_{name}",
        spawn=sim_utils.CuboidCfg(size=tuple(map(float, hi - lo)), collision_props=sim_utils.CollisionPropertiesCfg(),  # look: looks.py
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(map(float, (lo + hi) / 2))),
    )


# the shelf opens towards the arm (-y) and its walls block views from +-x and from behind: one camera high beyond the
# drawers looking at the shelf front, container and arm; one high at the +x end looking back over the container at the
# arm and the drawers
EYE_A, LOOK_A = (0.60, -1.30, 2.10), (0.05, 0.30, 0.95)
EYE_B, LOOK_B = (1.65, 0.05, 1.95), (0.0, -0.15, 0.80)
_SB = L.SHELF_BOXES
_FB = L.FILLER_BOXES
_SKY, _SUN = LK.light_cfgs()
# the desk: the table plus the filler strips on both sides of it (same union as the table + layout's strips), drawn as a
# painted body under one wooden top
_DESK = ((-L.TABLE_SIZE[0] / 2, _FB["table_strip_front"][0][1], 0.0), (L.TABLE_SIZE[0] / 2, _FB["table_strip"][1][1], L.TABLE_Z))


def _split_top(lo, hi, t=L.T):
    """(body lo, hi), (top lo, hi): a box cut t below its top, so the top can take the wood (looks.py)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (lo, np.r_[hi[:2], hi[2] - t]), (np.r_[lo[:2], hi[2] - t], hi)


# the full scene; attribute names become prim names (/World/envs/env_0/<name>) and keys of env.scene[...]
@configclass
class SceneCfg(E.SceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(visible=False))  # backdrop floor: looks.py
    sky = _SKY
    sun = _SUN
    table = board_cfg("body", *_split_top(*_DESK)[0], prefix="desk")
    table_top = board_cfg("top", *_split_top(*_DESK)[1], prefix="desk")
    bin0: ArticulationCfg = bin_cfg(0)
    bin1: ArticulationCfg = bin_cfg(1)
    bin2: ArticulationCfg = bin_cfg(2)
    shelf_plinth = board_cfg("plinth", *_SB["plinth"])
    shelf_board0 = board_cfg("board0", *_SB["board0"])
    shelf_board1 = board_cfg("board1", *_SB["board1"])
    shelf_side_l = board_cfg("side_l", *_SB["side_l"])
    shelf_side_r = board_cfg("side_r", *_SB["side_r"])
    shelf_back = board_cfg("back", *_SB["back"])
    counter_end_l = board_cfg("end_l", *_split_top(*_FB["end_l"])[0], prefix="filler")
    counter_end_r = board_cfg("end_r", *_split_top(*_FB["end_r"])[0], prefix="filler")
    counter_top_l = board_cfg("top_l", *_split_top(*_FB["end_l"])[1], prefix="filler")
    counter_top_r = board_cfg("top_r", *_split_top(*_FB["end_r"])[1], prefix="filler")
    container = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/container", spawn=sim_utils.UsdFileCfg(usd_path=str(ISAAC / L.CONTAINER_USD)),
                             init_state=AssetBaseCfg.InitialStateCfg(pos=L.CONTAINER_POS))
    obj0: RigidObjectCfg = obj_cfg(0)
    obj1: RigidObjectCfg = obj_cfg(1)
    obj2: RigidObjectCfg = obj_cfg(2)
    obj3: RigidObjectCfg = obj_cfg(3)
    obj4: RigidObjectCfg = obj_cfg(4)
    obj5: RigidObjectCfg = obj_cfg(5)
    obj6: RigidObjectCfg = obj_cfg(6)
    cam_a = E._camera("cam_a", EYE_A, 14.0, rot=E.look_at_quat(EYE_A, LOOK_A))
    cam_b = E._camera("cam_b", EYE_B, 14.0, rot=E.look_at_quat(EYE_B, LOOK_B))


# the environment = this scene + ur5e_env's actions (arm joints + gripper), 10 Hz control, 120 Hz physics
@configclass
class EnvCfg(E.Ur5eEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=8.0)


# ======================================================================================================================
# 6. MOTION PLANNER (cuRobo)
#   It has its OWN collision model of the scene, built from layout.py (not read from Isaac). sync_world() below
#   copies the simulated object poses and crate positions into it before each motion.
# ======================================================================================================================
objects0 = [(f"obj{i}", MESHES / f"{KEYS[i]}.obj", np.r_[layout0[i][:3] - to_base, layout0[i][3:]]) for i in range(7)]
planner = ArmPlanner(L.world_config(objects=objects0, shelf=True, container=True, fillers=True), time_dilation=args.speed)
HOME = planner.home

# ======================================================================================================================
# 7. BUILD THE SIMULATION
#   Final config tweaks (cameras off for --no-video, extra cameras for --stills, simplified gripper, start pose),
#   then create the env, apply the look (looks.py) and get handles to the robot, objects and drawers.
# ======================================================================================================================
# --stills: extra high-resolution cameras, name -> (eye, look-at, focal length [mm]), world frame
STILLS = {
    "front": ((0.9, -2.5, 2.3), (0.0, 0.0, 0.85), 18.0),
    "front_high": ((1.2, -1.9, 2.9), (0.0, 0.0, 0.8), 18.0),
    "end_px": ((2.4, -0.4, 2.4), (0.0, -0.05, 0.85), 18.0),
    "end_nx": ((-2.4, -0.6, 2.4), (0.0, -0.05, 0.85), 18.0),
    "top": ((0.0, -0.6, 3.6), (0.0, 0.0, 0.75), 14.0),
    "shelf": ((-0.35, -0.25, 1.55), (0.0, 0.47, 1.05), 14.0),
    "drawers": ((0.6, 0.15, 1.95), (-0.05, -0.62, 0.8), 16.0),
}
cfg = EnvCfg()
if args.no_video:
    cfg.scene.cam_a = cfg.scene.cam_b = cfg.scene.cam_wrist = None
if args.stills:
    if args.stills_cams:
        STILLS = {k: STILLS[k] for k in args.stills_cams.split(",")}
    cfg.scene.cam_a = cfg.scene.cam_b = None  # the episode cameras are not needed for stills
    sw, sh = map(int, args.stills_size.lower().split("x"))
    for nm, (eye, look, focal) in STILLS.items():
        c = E._camera(f"still_{nm}", eye, focal, rot=E.look_at_quat(eye, look))
        c.width, c.height, c.update_period = sw, sh, 0.0
        setattr(cfg.scene, f"still_{nm}", c)
E.use_simple_gripper(cfg, grip_force=args.grip_force)  # two force-limited sliding pads (see gripper_action.py)
cfg.scene.robot.init_state.joint_pos.update(dict(zip(n.ARM_JOINTS, map(float, HOME))))
env = ManagerBasedRLEnv(cfg)  # builds the scene in Isaac Sim
LK.apply(env.scene.stage)  # materials, lights, backdrop room, scanner pad (visual only)
env.reset()
dev = env.device
robot = env.scene["robot"]
ARM_IDS = robot.find_joints(list(n.ARM_JOINTS), preserve_order=True)[0]
OBJS = [env.scene[f"obj{i}"] for i in range(7)]
BINS = [env.scene[f"bin{i}"] for i in range(3)]


# same friction on the gripper and all objects (--friction)
def set_friction(asset, mu):
    m = asset.root_physx_view.get_material_properties()
    m[..., 0], m[..., 1] = mu, mu
    asset.root_physx_view.set_material_properties(m, torch.arange(m.shape[0]))


set_friction(robot, args.friction)
for o in OBJS:
    set_friction(o, args.friction)
print("object masses [kg]:", {KEYS[i]: round(float(o.root_physx_view.get_masses()[0].sum()), 3) for i, o in enumerate(OBJS)}, flush=True)

# ======================================================================================================================
# 8. CONTROL HELPERS: how the script moves the robot
#   step() sends ONE 10 Hz action to the env: 6 arm joint targets + 1 gripper value (and records the camera frames).
#   run() plays a planned trajectory step by step; hold() waits; set_grip() opens/closes the gripper gradually.
#   state = the last commanded arm configuration and gripper value.
# ======================================================================================================================
frames, motions = [], []
state = {"q": np.array(HOME, float), "grip": 0.0}


class Fail(RuntimeError):
    pass


def q_meas():
    return robot.data.joint_pos[0, ARM_IDS].cpu().numpy()


def step(q, grip):
    env.step(torch.tensor([[*q, grip]], dtype=torch.float32, device=dev))
    if args.stills or args.no_video:
        return
    frames.append(np.concatenate([env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
                                  for c in ("cam_a", "cam_b", "cam_wrist")], 1))


def run(plan: Plan, name, grip=None):
    g = state["grip"] if grip is None else grip
    err = 0.0
    for q in plan.q[1:]:
        step(q, g)
        err = max(err, float(np.abs(q_meas() - q).max()))
    state["q"] = plan.q[-1].copy()
    motions.append(dict(motion=name, seconds=round(plan.duration, 2), max_joint_err_deg=round(float(np.degrees(err)), 2)))


def hold(k):
    for _ in range(k):
        step(state["q"], state["grip"])


def set_grip(target, steps=15, settle=4):
    g0 = state["grip"]
    for k in range(1, steps + 1):
        step(state["q"], g0 + (target - g0) * k / steps)
    state["grip"] = target
    hold(settle)


PAD_IDS = robot.find_joints(["left_pad_slide", "right_pad_slide"], preserve_order=True)[0]
OBJ_MASS = [float(o.root_physx_view.get_masses()[0].sum()) for o in OBJS]


def set_grip_force(mass):
    """Squeeze force per pad for an object of this mass, as a real 2F-85's force setting (20-235 N): PhysX lets the pads
    push into light objects above ~300 N per kg (a 28 g foam brick at 20 N, a 0.1 kg cube at 40 N), so 250 N/kg, within
    3-30 N. Not more than 30 N: at 60 N a 0.35 kg soup can was not lifted either (the 1.13 kg bleach bottle can slip at
    30 N)."""
    f = float(np.clip(250.0 * mass, 3.0, 30.0))
    lim = torch.tensor([[f, f]], device=dev)
    robot.write_joint_effort_limit_to_sim(lim, joint_ids=PAD_IDS)
    robot.write_joint_stiffness_to_sim(lim / 0.005, joint_ids=PAD_IDS)  # full force 5 mm short of the target (= coupling)
    return round(f, 1)


# ======================================================================================================================
# 9. STATE HELPERS: read the simulation, keep the planner's world in sync, reset an episode
# ======================================================================================================================
# a failed plan aborts the current attempt (caught in the episode loop)
def need(plan, what):
    if plan is None:
        raise Fail(f"planning failed: {what}")
    return plan


def obj_T(i):  # pose of object i as a 4x4 matrix in the arm-base frame
    o = OBJS[i]
    return T_of(np.r_[o.data.root_pos_w[0].cpu().numpy() - to_base, o.data.root_quat_w[0].cpu().numpy()])


def slide(i):  # how far crate i is pulled out [m]
    return float(BINS[i].data.joint_pos[0, 0])


def latch(i, on):  # hold crate i where it is (on) or let it slide freely (off)
    BINS[i].write_joint_stiffness_to_sim(torch.tensor([[args.latch_stiffness if on else 0.0]], device=dev))
    if on:
        BINS[i].set_joint_position_target(BINS[i].data.joint_pos.clone())


def sync_world():  # copy crate and object poses from the simulation into the planner's world
    for i in range(3):
        for nm, c, _ in L.bin_boxes(i, slide(i)):
            if "crate" in nm:
                planner.set_obstacle_pose(nm, np.r_[c, 1.0, 0.0, 0.0, 0.0])
    for i in range(7):
        if planner.attached != f"obj{i}":
            planner.set_obstacle_pose(f"obj{i}", pose7(obj_T(i)))


def gap_to_angle(gap):  # finger opening [m] -> gripper command (finger angle, rad)
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, E.GRIPPER_CLOSED))


def reset_episode():  # close the crates, new random object poses, robot at home
    planner.detach()
    env.reset()
    for i in range(3):
        latch(i, False)
    state.update(q=np.array(HOME, float), grip=0.0)
    for i, p in enumerate(random_layout()):
        OBJS[i].write_root_pose_to_sim(torch.tensor([np.r_[p[:3], p[3:]]], dtype=torch.float32, device=dev))
        OBJS[i].write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    hold(25)
    sync_world()


# ======================================================================================================================
# 10. TASK PHASES: what the robot does in an episode
#   open_drawer(i) -> pick(i) -> place(...) -> go_home(). Each one plans with cuRobo, then executes with run().
#   A phase raises Fail when it cannot continue; the episode loop then tries the next object.
# ======================================================================================================================
def open_drawer(i):
    """Hook crate i's rim bar with the gripper and pull the crate out. Returns how far it is out [m]."""
    crate = L.crate_box_names(i)
    wall = np.r_[L.BIN_X[i], L.WALL_Y + slide(i), L.BAR_Z - L.TABLE_Z, Q_HOOK]
    # 1. move 12 cm above the rim bar, open the fingers to the hook gap
    sync_world()
    run(need(planner.plan_pose(state["q"], wall + np.r_[0, 0, 0.12, 0, 0, 0, 0], tag="to the rim bar"), "to the rim bar"), "to the rim bar", grip=0.0)
    set_grip(gap_to_angle(args.hook_gap), steps=8)
    with planner.relaxed(ignore=crate):  # the gripper has to touch the crate now
        # 2. lower straight onto the bar and close the gripper around it
        run(need(planner.plan_linear(state["q"], wall, frame="base", free_axis=2), "onto the bar"), "onto the bar")
        set_grip(E.GRIPPER_CLOSED, steps=10)
        # 3. plan the pull
        # the longest pull whose unhook also works and ends collision-free with the crate back in the planner's world
        # (bin 1, in front of the base: at 0.15 m the folded arm self-collides; at 0.135 m the unhooked arm overlaps the
        # crate and every later plan is refused as starting in collision)
        chosen = None
        for d in (args.pull, args.pull - 0.015, args.pull - 0.03, args.pull - 0.045):
            pull = planner.plan_cartesian(state["q"], wall + np.r_[0, d, 0, 0, 0, 0, 0], max_speed=0.05, quiet=True)
            if pull is None:
                continue
            off = planner.plan_cartesian(pull.q[-1], wall + np.r_[0, d, 0.12, 0, 0, 0, 0], max_speed=0.06, quiet=True)
            if off is None:
                continue
            # check the end state with the crate at slide d (its boxes are still where the crate is now)
            for nm, c, _ in L.bin_boxes(i, slide(i) + d):
                if "crate" in nm:
                    planner.set_obstacle_pose(nm, np.r_[c, 1.0, 0.0, 0.0, 0.0])
            for nm in crate:
                planner.enable_obstacle(nm, True)
            valid = bool(planner.mg.check_start_state(planner._js(off.q[-1]))[0])
            for nm in crate:
                planner.enable_obstacle(nm, False)
            sync_world()
            if valid:
                chosen = (d, pull, off)
                break
        if chosen is None:
            raise Fail("planning failed: no pull distance with a collision-free unhook")
        # 4. pull, hold the crate there, open the fingers and lift off the bar
        d, pull, off = chosen
        run(pull, f"pull {d:.3f} m")
        latch(i, True)
        set_grip(gap_to_angle(args.hook_gap), steps=8)
        run(off, "off the bar")
    set_grip(0.0, steps=8)
    return slide(i)


def pick(i):
    """Grasp object i and get it out of its place. Returns (grasp record, T_obj before the pick, support z)."""
    k, src = KEYS[i], SRC[i]
    sync_world()
    T0 = obj_T(i)
    support = (L.FLOOR_TOP - L.TABLE_Z) if src == "drawer" else (L.SHELF_LEVELS[(i - 3) // 2] - L.TABLE_Z)
    # 1. candidate grasps: the object's pre-computed grasps moved to its current pose, minus those that would hit the
    #    support (crate floor / shelf board); then a score-weighted random order
    inf = INFO[k]
    Ts = T0 @ inf["G"]
    keep = world_filter(Ts, floor_z=support, min_clear=0.004, widths=inf["widths"])
    if src == "drawer":  # inside a crate (17 cm walls, roof behind) only near-vertical approaches can work
        keep &= np.degrees(np.arccos(np.clip(-Ts[:, 2, 2], -1, 1))) <= 40.0
    idx = np.where(keep)[0]
    if len(idx) == 0:
        raise Fail("no grasp keeps the gripper above the support")
    sc = inf["scores"][idx]
    p = np.exp((sc - sc.max()) / 0.3)
    order = rng.choice(idx, size=min(args.candidates, len(idx)), replace=False, p=p / p.sum())
    cand = np.array([pose7(Ts[j]) for j in order])
    # 2. plan: the first candidate whose approach AND way out (lift + extraction, planning/tasks.py) both plan
    ext = {}

    def check(gi, to_pre, app):
        e = plan_extraction(planner, app.q[-1], Ts[order[gi]], T0, f"obj{i}", src)
        if e is not None:
            ext[gi] = e
        return e is not None

    t = time.time()
    to_pre, approach, gi = planner.plan_grasp(state["q"], cand, approach=0.10, target=f"obj{i}", order="given", max_tries=10, check=check)
    t_plan = time.time() - t
    if approach is None:
        raise Fail("planning failed: no grasp with a feasible approach and extraction")
    Tg = Ts[order[gi]]
    rec = dict(grasp=int(order[gi]), width_mm=round(float(inf["widths"][order[gi]]) * 1000, 1),
               approach_deg=round(float(np.degrees(np.arccos(np.clip(-Tg[2, 2], -1, 1)))), 1), plan_s=round(t_plan, 1))
    # 3. execute: to 10 cm before the grasp, approach, squeeze (force set from the object's mass), lift
    run(to_pre, "to pregrasp", grip=0.0)
    run(approach, "approach")
    rec["grip_force_N"] = set_grip_force(OBJ_MASS[i])
    set_grip(E.GRIPPER_CLOSED, steps=15, settle=4)
    lift, out = ext[gi]
    run(lift, "lift")
    rose = obj_T(i)[2, 3] - T0[2, 3]
    if rose < 0.015:
        raise Fail(f"object not in the gripper (rose {rose * 1000:.0f} mm)")
    # 4. tell the planner the object is now in the hand, then move it out of the crate / shelf
    planner.set_obstacle_pose(f"obj{i}", pose7(obj_T(i)))
    planner.attach(q_meas(), f"obj{i}")
    run(out, "out of the drawer" if src == "drawer" else "out of the shelf")
    return rec, T0, support


def place(i, T0, support, width):
    """Put the held object i into the container, in the orientation it had at its source. Returns a result record."""
    # 1. drop poses: positions x yaws in the container, raised where the fingers would hit its walls
    T_held = np.linalg.inv(T_of(planner.fk(q_meas()))) @ obj_T(i)
    drops = drop_poses(T_held, T0, support, rng, n_xy=3, yaws=(0, 45, 90, 135, 180, -135, -90, -45), footprint=footprint(KEYS[i], T0) + 0.01)
    drops, raised = clear_container_walls(drops, width)
    # 2. plan to the first reachable drop pose (from above, straight down) and execute
    sync_world()
    t = time.time()
    to_drop, down, di = planner.plan_grasp(state["q"], drops, approach=0.10, approach_dir=(0, 0, -1), order="given", max_tries=16, tag="place")
    if down is None:
        raise Fail("planning failed: no reachable drop pose in the container")
    t_plan = time.time() - t
    run(to_drop, "to above the container")
    run(down, "down into the container")
    # 3. still in the hand? then release and back off
    slip = float(np.linalg.norm((np.linalg.inv(T_of(planner.fk(q_meas()))) @ obj_T(i))[:3, 3] - T_held[:3, 3]))
    if slip > 0.03:
        raise Fail(f"object lost on the way to the container (moved {slip * 1000:.0f} mm in the hand)")
    set_grip(0.0, steps=12, settle=8)
    planner.detach()
    sync_world()
    p = planner.fk(q_meas())
    with planner.relaxed(ignore=[f"obj{i}"]):  # back off along the tool axis, else straight up
        back = planner.plan_cartesian(state["q"], pregrasp(p, 0.10), quiet=True) or planner.plan_cartesian(state["q"], p + np.r_[0, 0, 0.10, 0, 0, 0, 0], quiet=True)
    if back is not None:
        run(back, "back off")
    # 4. result: inside the container? how far from the planned spot, how much tilted?
    hold(10)
    T = obj_T(i)
    inside = (L.CONTAINER_INNER[0][0] < T[0, 3] < L.CONTAINER_INNER[0][1] and L.CONTAINER_INNER[1][0] < T[1, 3] < L.CONTAINER_INNER[1][1]
              and T[2, 3] < (L.CONTAINER_FLOOR_TOP - L.TABLE_Z) + 0.20)
    T_goal = T_of(drops[di]) @ T_held
    tilt = float(np.degrees(np.arccos(np.clip(T[:3, :3][:, 2] @ T_goal[:3, :3][:, 2], -1, 1))))  # change of the object's z axis
    return dict(in_container=bool(inside), place_plan_s=round(t_plan, 1), released_above_mm=round(float(raised[di]) * 1000 + 15, 0), xy_err_mm=round(float(np.hypot(*(T[:2, 3] - T_goal[:2, 3]))) * 1000, 1),
                tilt_deg=round(tilt, 1), backed_off=back is not None)


def go_home():
    """Plan back to the home pose; if no plan is found, put the arm there directly (not a real motion)."""
    sync_world()
    p = planner.plan_to_config(state["q"], HOME, tag="home")
    if p is not None:
        run(p, "home")
        return "planned"
    robot.write_joint_state_to_sim(torch.tensor([list(map(float, HOME))], dtype=torch.float32, device=dev), torch.zeros(1, 6, device=dev), joint_ids=ARM_IDS)
    state["q"] = np.array(HOME, float)
    env.action_manager.get_term("arm").reset()
    hold(5)
    return "arm reset (not a planned motion)"


# ======================================================================================================================
# 11. --stills MODE: render high-resolution pictures of the scene (crates closed, then open) and exit
# ======================================================================================================================
args.out.mkdir(parents=True, exist_ok=True)
if args.stills:
    import os

    def save_stills(state_tag):
        for nm in STILLS:
            img = env.scene[f"still_{nm}"].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            path = args.out / f"still_{nm}_{state_tag}{args.tag}.png"
            imageio.imwrite(path, img)
            print("wrote", path, flush=True)

    reset_episode()
    hold(20)  # a few more rendered frames, so the renderer has converged
    save_stills("closed")
    PULL = 0.15
    for i in map(int, args.stills_open.split(",")):  # pull the crates out (housing +x = world +y) and move their objects with them
        BINS[i].write_joint_state_to_sim(torch.tensor([[PULL]], device=dev), torch.zeros(1, 1, device=dev))
        p = OBJS[i].data.root_pose_w.clone()
        p[0, 1] += PULL
        OBJS[i].write_root_pose_to_sim(p)
        OBJS[i].write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    hold(25)
    save_stills("open")
    env.close()
    sys.stdout.flush()
    os._exit(0)

# ======================================================================================================================
# 12. MAIN LOOP: the episodes
#   Even episodes take an object from a drawer, odd ones from the shelf. Objects of that source are tried in random
#   order until one ends in the container. Each episode writes ep<NN>.mp4 and adds its record to episodes.json.
# ======================================================================================================================
records = []
for ep in range(args.episodes):
    reset_episode()
    frames.clear()
    motions.clear()
    src = "drawer" if ep % 2 == 0 else "shelf"
    cands = [i for i in range(7) if SRC[i] == src]
    rng.shuffle(cands)
    rec = dict(episode=ep, source=src, attempts=[])
    print(f"\n=== episode {ep}: from a {src} ===", flush=True)
    for i in cands:
        att = dict(obj=i, key=KEYS[i])
        rec["attempts"].append(att)
        try:
            if src == "drawer":
                att["crate_out"] = round(open_drawer(i), 3)
            g, T0, support = pick(i)
            att.update(g)
            att.update(place(i, T0, support, g["width_mm"] / 1000))
            att["ok"] = att["in_container"]
            print(f"   {KEYS[i]}: grasp {g['approach_deg']:.0f} deg from vertical, width {g['width_mm']:.0f} mm -> "
                  f"{'IN THE CONTAINER' if att['in_container'] else 'NOT in the container'} (xy {att['xy_err_mm']:.0f} mm, tilt {att['tilt_deg']:.0f} deg)", flush=True)
            att["home"] = go_home()
            break
        except Fail as ex:
            att.update(ok=False, error=str(ex))
            print(f"   {KEYS[i]}: {ex}", flush=True)
            set_grip(0.0, steps=8)
            planner.detach()
            att["home"] = go_home()
            if src == "drawer" and "crate_out" in att:
                break  # this crate is open now; the episode ends rather than opening a second one
    rec["ok"] = any(a.get("ok") for a in rec["attempts"])
    rec["motions"] = list(motions)
    records.append(rec)
    if frames:
        imageio.mimwrite(args.out / f"ep{ep:02d}{args.tag}.mp4", frames, fps=10, macro_block_size=1)
    (args.out / f"episodes{args.tag}.json").write_text(json.dumps(records, indent=1, default=float))

# summary of all episodes, then exit (os._exit: Isaac's normal shutdown can hang)
print("\nSUMMARY")
for r in records:
    a = r["attempts"][-1]
    print(f"   episode {r['episode']} ({r['source']}): {'OK ' if r['ok'] else 'FAIL'} {a['key']} {a.get('error', '')} after {len(r['attempts'])} object(s)")
print(f"{sum(r['ok'] for r in records)}/{len(records)} episodes put an object in the container", flush=True)
env.close()
sys.stdout.flush()
import os  # noqa: E402

os._exit(0)
