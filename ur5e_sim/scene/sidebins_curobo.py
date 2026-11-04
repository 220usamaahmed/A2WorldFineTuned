"""Side-bins task driven by cuRobo: open each drawer (hook the crate's rim bar, pull), pick the object, put it on the table.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python sidebins_curobo.py --headless [--bins 0,1,2] [--speed 0.5]

Same scene as sidebins_test.py (layout.py), but the arm is driven in joint space (ur5e_env.JointActionsCfg) along
trajectories from planning/planner.py instead of interpolated TCP poses through differential IK:
  * free-space motions: cuRobo MotionGen with the table, the housings, the crates (at their current slide) and the
    objects (YCB meshes at their current poses) as obstacles; a carried object is attached to the robot;
  * approach / lift / pull / place moves: straight TCP lines (planner.plan_linear / plan_cartesian);
  * grasp (--grasps rule): top-down at the object (fingers along +x or -x); (--grasps sampled): antipodal grasps from
    planning/grasps.py at the object's simulated pose, in score-weighted random order; planner.plan_grasp picks one;
  * place: the object goes down on the table in the orientation it rested in in the crate (so it is stable), straight
    down from 10 cm above; several yaw rotations of the whole placement are offered to the planner.
Objects (--objects): "cans" = a soup can standing in each crate (as sidebins_test.py); "mixed" = a different short YCB
object per crate in a random resting pose and yaw (--seed).
The planner's world is synced from the simulator before every phase.
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--bins", default="0,1,2")
ap.add_argument("--pull", type=float, default=0.15, help="how far to pull each crate [m]")
ap.add_argument("--speed", type=float, default=0.5, help="cuRobo time dilation (1 = time-optimal at the joint limits)")
ap.add_argument("--hook-gap", type=float, default=0.040, help="finger gap while straddling the end wall [m]")
ap.add_argument("--latch-stiffness", type=float, default=3000.0)
ap.add_argument("--slide-damping", type=float, default=15.0)
ap.add_argument("--grip-k", type=float, default=6.0)
ap.add_argument("--friction", type=float, default=1.0, help="pads and objects; 1.0 with the simple gripper (the four-bar model needed 4)")
ap.add_argument("--can-y", default="-0.04,0.0,0.04", help="crate-local y of the soup can in each bin (sidebins_test.py: -0.04,0,0.04)")
ap.add_argument("--objects", default="cans", help='"cans", "mixed" (061_foam_brick,009_gelatin_box,011_banana), or 3 comma-separated YCB keys')
ap.add_argument("--grasps", default="rule", choices=["rule", "sampled"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--candidates", type=int, default=40, help="sampled grasps handed to the planner per pick")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/curobo"))
ap.add_argument("--tag", default="")
ap.add_argument("--gripper", default="simple", choices=["simple", "linkage"], help="simple: sliding physics pads (make_simple_gripper.py); linkage: the original four-bar model")
ap.add_argument("--grip-force", type=float, default=20.0, help="squeeze force per pad with --gripper simple [N]")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

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
import robot_defs as n  # noqa: E402
import ur5e_env as E  # noqa: E402
from grasps import T_from_pose7, cached_grasps, graspable_stable_poses, load_mesh, world_filter  # noqa: E402
from planner import ArmPlanner, Plan, pregrasp  # noqa: E402

BIN_USD = HERE.parent / "assets" / "usd" / "pullbin.usd"
YCB = Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics")
MESHES = Path("/home/data/siddiquieu1/assets/ycb_meshes")
# table, in front of the arm (world x, y); a pulled-out crate reaches over the table to y = -0.17 (bin 2 spans x 0.27-0.67),
# so the spots stay >= 5 cm clear of it (the old -0.12 left 14 mm between the can and the crate)
PLACE_XY = [(0.40, 0.14), (0.40, 0.02), (0.42, -0.04)]  # bin 2 spot: at y -0.08 the wrist met the crate wall on set-down
PHYS = Path("/home/data/siddiquieu1/assets/ycb_physics")
rng = np.random.default_rng(args.seed)


def upright_quat(yaw_deg):
    x, y, z, w = (R.from_euler("z", yaw_deg, degrees=True) * R.from_euler("x", 90, degrees=True)).as_quat()
    return (float(w), float(x), float(y), float(z))


def object_specs():
    """Per bin: dict(key, usd, mesh, pose = world pose7 of the object resting in its (closed) crate)."""
    out = []
    if args.objects == "cans":  # standing upright (the files' long axis is y: +90 deg about x), as in sidebins_test.py
        for i, y in enumerate(args.can_y.split(",")):
            pos = L.crate_to_world(i, 0.10, float(y), L.FLOOR_TOP + 0.102 / 2 + 0.003)
            out.append(dict(key="005_tomato_soup_can", usd=YCB / "005_tomato_soup_can.usd", mesh=MESHES / "005_tomato_soup_can_physics.obj",
                            pose=np.r_[pos, upright_quat(0)]))
        return out
    keys = ["061_foam_brick", "009_gelatin_box", "011_banana"] if args.objects == "mixed" else args.objects.split(",")
    for i, k in enumerate(keys):
        m = load_mesh(MESHES / f"{k}.obj")
        poses = graspable_stable_poses(m, cached_grasps(k, MESHES / f"{k}.obj", n_pairs=500, n_roll=16))
        if not poses:
            raise SystemExit(f"{k}: no resting pose in which the 2F-85 can grasp it")
        Ts, ps, _ = zip(*poses)
        T = np.array(Ts[rng.choice(len(Ts), p=np.array(ps) / sum(ps))])  # rests on z = 0
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
        T = Rz @ T
        # inside the roof window of a crate pulled out ~0.12-0.15 m, away from the walls
        xc, yc = rng.uniform(0.06, 0.14), rng.uniform(-0.08, 0.08)
        T[:3, 3] += L.crate_to_world(i, xc, yc, L.FLOOR_TOP + 0.003)
        x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
        out.append(dict(key=k, usd=PHYS / f"{k}.usd", mesh=MESHES / f"{k}.obj", pose=np.r_[T[:3, 3], w, x, y, z]))
    return out


SPECS = object_specs()
for _s in SPECS:
    print(f"object {_s['key']} at {np.round(_s['pose'][:3], 3)}", flush=True)
Q_PICK = np.array([0.0, 1.0, 0.0, 0.0])  # tool down, fingers close along world x
Q_PICK_FLIP = np.array([0.0, 0.0, 1.0, 0.0])  # tool down, fingers along -x (180 deg about z from Q_PICK)
Q_HOOK = np.array([*R.from_matrix(np.stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]], 1)).as_quat()[[3, 0, 1, 2]]])  # fingers along y


def bin_cfg(i):
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/bin{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(BIN_USD), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(L.BIN_X[i], L.ORIGIN_Y, 0.0), rot=L.HOUSING_QUAT, joint_pos={"crate_slide": 0.0}),
        actuators={"slide": ImplicitActuatorCfg(joint_names_expr=["crate_slide"], stiffness=0.0, damping=args.slide_damping, effort_limit_sim=500.0)},
    )


def obj_cfg(i):
    sp = SPECS[i]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(sp["usd"]), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(map(float, sp["pose"][:3])), rot=tuple(map(float, sp["pose"][3:]))),
    )


EYE_A, EYE_B = (0.0, 1.05, 1.85), (1.55, -0.55, 1.45)


@configclass
class SceneCfg(E.SceneCfg):
    cube = None
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(size=(L.TABLE_SIZE[0], L.TABLE_SIZE[1], L.TABLE_Z), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, L.TABLE_Z / 2)),
    )
    bin0: ArticulationCfg = bin_cfg(0)
    bin1: ArticulationCfg = bin_cfg(1)
    bin2: ArticulationCfg = bin_cfg(2)
    obj0: RigidObjectCfg = obj_cfg(0)
    obj1: RigidObjectCfg = obj_cfg(1)
    obj2: RigidObjectCfg = obj_cfg(2)
    cam_dbg = E._camera("cam_dbg", EYE_A, 13.0, rot=E.look_at_quat(EYE_A, (0.0, -0.45, 0.85)))
    cam_side = E._camera("cam_side", EYE_B, 13.0, rot=E.look_at_quat(EYE_B, (0.0, -0.4, 0.85)))


@configclass
class EnvCfg(E.Ur5eTableJointEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=8.0)


# ---------------------------------------------------------------- planner (before the env: the robot starts at its home)
to_base = np.array([0.0, 0.0, L.TABLE_Z])
objects0 = [(f"obj{i}", SPECS[i]["mesh"], np.r_[SPECS[i]["pose"][:3] - to_base, SPECS[i]["pose"][3:]]) for i in range(3)]
GRASP_LIB = {sp["key"]: cached_grasps(sp["key"], sp["mesh"], n_pairs=500, n_roll=16) for sp in SPECS} if args.grasps == "sampled" else {}
planner = ArmPlanner(L.world_config(objects=objects0), time_dilation=args.speed)
HOME = planner.home
print("planner ready; home", np.round(HOME, 3), flush=True)

cfg = EnvCfg()
if args.gripper == "simple":
    E.use_simple_gripper(cfg, grip_force=args.grip_force)
cfg.scene.robot.init_state.joint_pos.update(dict(zip(n.ARM_JOINTS, map(float, HOME))))
if args.gripper == "linkage":
    cfg.scene.robot.actuators["gripper"].stiffness = args.grip_k
    cfg.scene.robot.actuators["gripper"].damping = 0.05 * args.grip_k
env = ManagerBasedRLEnv(cfg)
env.reset()
dev = env.device
robot = env.scene["robot"]
ARM_IDS = robot.find_joints(list(n.ARM_JOINTS), preserve_order=True)[0]


def set_friction(asset, mu):  # on the shapes themselves (binding a material to the prims had no effect)
    m = asset.root_physx_view.get_material_properties()
    m[..., 0], m[..., 1] = mu, mu
    asset.root_physx_view.set_material_properties(m, torch.arange(m.shape[0]))


set_friction(robot, args.friction)
for i in range(3):
    set_friction(env.scene[f"obj{i}"], args.friction)

frames, log = [], []
state = {"q": np.array(HOME, float), "grip": 0.0}


def q_meas():
    return robot.data.joint_pos[0, ARM_IDS].cpu().numpy()


def step(q, grip):
    env.step(torch.tensor([[*q, grip]], dtype=torch.float32, device=dev))
    rgb = [env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for c in ("cam_dbg", "cam_side", "cam_wrist")]
    frames.append(np.concatenate(rgb, 1))


def run(plan: Plan, grip=None, name=""):
    """Play a planned trajectory (one row per control step) and report how well the arm followed it."""
    g = state["grip"] if grip is None else grip
    err = 0.0
    for k, q in enumerate(plan.q[1:]):
        step(q, g)
        err = max(err, float(np.abs(q_meas() - q).max()))
    state["q"] = plan.q[-1].copy()
    log.append(dict(motion=name or plan.status, steps=len(plan.q) - 1, seconds=round(plan.duration, 2), max_joint_err_deg=round(np.degrees(err), 2)))
    print(f"   ran {name or plan.status}: {plan.duration:.1f} s, max joint tracking error {np.degrees(err):.2f} deg", flush=True)


def hold(steps, grip=None):
    for _ in range(steps):
        step(state["q"], state["grip"] if grip is None else grip)


def set_grip(target, steps=15, settle=4):
    g0 = state["grip"]
    for k in range(1, steps + 1):
        step(state["q"], g0 + (target - g0) * k / steps)
    state["grip"] = target
    hold(settle)


def gap_to_angle(gap):  # Robotiq 2F-85: 0 rad = 85 mm open, 0.8 rad = closed
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, E.GRIPPER_CLOSED))


def obj_pose_base(i):
    o = env.scene[f"obj{i}"]
    return np.r_[o.data.root_pos_w[0].cpu().numpy() - to_base, o.data.root_quat_w[0].cpu().numpy()]


def slide(i):
    return float(env.scene[f"bin{i}"].data.joint_pos[0, 0])


def sync_world():
    """Crate boxes and (unattached) objects in the planner's world <- simulator."""
    for i in range(3):
        for nm, c, _ in L.bin_boxes(i, slide(i)):
            if "crate" in nm:
                planner.set_obstacle_pose(nm, np.r_[c, 1.0, 0.0, 0.0, 0.0])
        if planner.attached != f"obj{i}":
            planner.set_obstacle_pose(f"obj{i}", obj_pose_base(i))


def latch(i, on):
    b = env.scene[f"bin{i}"]
    b.write_joint_stiffness_to_sim(torch.tensor([[args.latch_stiffness if on else 0.0]], device=dev))
    if on:
        b.set_joint_position_target(b.data.joint_pos.clone())


def tcp_err(p7_base):
    return float(np.linalg.norm(planner.fk(q_meas())[:3] - p7_base[:3]))


def need(plan, what):
    if plan is None:
        raise RuntimeError(f"planning failed: {what}")
    return plan


def _pose7(T):
    x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], w, x, y, z]


def grasp_candidates(i, p):
    """TCP poses [N, 7] (base frame) to try for object i at pose p, best / random first, plus a label per candidate."""
    if args.grasps == "rule":  # top-down at the centre, palm and wrist camera ~4 cm above the TCP
        h = float(load_mesh(SPECS[i]["mesh"]).extents[1]) if SPECS[i]["key"] == "005_tomato_soup_can" else 0.0
        gz = max(0.01, h / 2 - 0.042)
        return np.array([np.r_[p[:3] + [0, 0, gz], q] for q in (Q_PICK, Q_PICK_FLIP)]), ["top-down +x", "top-down -x"]
    G = GRASP_LIB[SPECS[i]["key"]]
    Ts = T_from_pose7(p) @ np.array([g.T for g in G])
    idx = np.where(world_filter(Ts, floor_z=L.FLOOR_TOP - L.TABLE_Z, min_clear=0.004, widths=[g.width for g in G]))[0]
    if len(idx) == 0:
        raise RuntimeError("no sampled grasp keeps the gripper above the crate floor")
    sc = np.array([G[k].score for k in idx])
    pr = np.exp((sc - sc.max()) / 0.3)
    order = rng.choice(idx, size=min(args.candidates, len(idx)), replace=False, p=pr / pr.sum())
    print(f"   {len(G)} sampled grasps, {len(idx)} keep the gripper above the crate floor, trying {len(order)}", flush=True)
    return np.array([_pose7(Ts[k]) for k in order]), [f"sampled #{k} (width {G[k].width * 1000:.0f} mm)" for k in order]


def recover(i):
    """After a failed phase: let go, back straight up out of the crate, go home; reset the arm if even that fails."""
    set_grip(0.0, steps=8)
    planner.detach()
    sync_world()
    with planner.relaxed(ignore=[f"obj{i}"] + L.crate_box_names(i)):
        p = planner.fk(q_meas())
        up = planner.plan_cartesian(q_meas(), p + np.r_[0, 0, 0.12, 0, 0, 0, 0], quiet=True)
    if up is not None:
        run(up, name="recovery: back off")
    home = planner.plan_to_config(state["q"], HOME, tag="recovery: home")
    if home is not None:
        run(home, name="recovery: home")
        return "planned"
    robot.write_joint_state_to_sim(torch.tensor([list(HOME)], device=dev), torch.zeros(1, 6, device=dev), joint_ids=ARM_IDS)
    state["q"] = np.array(HOME, float)
    env.action_manager.get_term("arm").reset()
    hold(5)
    return "arm reset to home (not a planned motion)"


hold(10)
results = []
for i in [int(b) for b in args.bins.split(",")]:
    name = SPECS[i]["key"]
    rec = dict(bin=i, object=name)
    print(f"\n=== bin {i} ({name}) ===", flush=True)
    try:
        # ---- 1. hook the crate's front end wall and pull it out ----
        crate = L.crate_box_names(i)
        wall = np.r_[L.BIN_X[i], L.WALL_Y + slide(i), L.BAR_Z - L.TABLE_Z, Q_HOOK]
        above = wall + np.r_[0, 0, 0.12, 0, 0, 0, 0]
        sync_world()
        run(need(planner.plan_pose(state["q"], above, tag="to the rim bar"), "to the rim bar"), grip=0.0, name="to the rim bar")
        set_grip(gap_to_angle(args.hook_gap), steps=8)
        with planner.relaxed(ignore=crate):  # the fingers straddle the crate's end wall
            run(need(planner.plan_linear(state["q"], wall, frame="base", free_axis=2), "down to the bar"), name="down onto the bar")
            e = tcp_err(wall)
            set_grip(E.GRIPPER_CLOSED, steps=10)
            # as far as the arm can pull without self-collision (bin 1, right in front of the base, folds the arm onto
            # itself near the end of a 0.15 m pull)
            for d in (args.pull, args.pull - 0.015, args.pull - 0.03):
                pulled = wall + np.r_[0, d, 0, 0, 0, 0, 0]
                pull_plan = planner.plan_cartesian(state["q"], pulled, max_speed=0.05, quiet=True)
                if pull_plan is not None:
                    break
            run(need(pull_plan, "pull"), name=f"pull {d:.3f} m")
            out = slide(i)
            latch(i, True)  # drawer detent stand-in: the crate stays where the arm leaves it
            set_grip(gap_to_angle(args.hook_gap), steps=8)
            run(need(planner.plan_linear(state["q"], pulled + np.r_[0, 0, 0.12, 0, 0, 0, 0], frame="base", free_axis=2), "lift off"), name="lift off the bar")
        print(f"   crate pulled out {out:.3f} m (target {args.pull:.2f}); TCP error at the bar {e * 1000:.1f} mm", flush=True)
        rec.update(crate_out=round(out, 3), pulled_cmd=round(d, 3), opened=out > d - 0.01, tcp_err_hook_mm=round(e * 1000, 1))

        # ---- 2. pick the object ----
        set_grip(0.0, steps=8)
        hold(5)
        sync_world()
        p = obj_pose_base(i)
        grasps, meta = grasp_candidates(i, p)
        to_pre, approach, gi = planner.plan_grasp(state["q"], grasps, approach=0.10, target=f"obj{i}",
                                                  order="given" if args.grasps == "sampled" else "nearest", max_tries=10)
        need(approach, "grasp")
        run(to_pre, grip=0.0, name="to pregrasp")
        run(approach, name="approach")
        e = tcp_err(grasps[gi])
        set_grip(E.GRIPPER_CLOSED, steps=15, settle=4)
        with planner.relaxed(ignore=[f"obj{i}"]):
            up = planner.fk(q_meas()) + np.r_[0, 0, 0.10, 0, 0, 0, 0]  # straight up, whatever the approach direction
            run(need(planner.plan_cartesian(state["q"], up, max_speed=0.05), "lift"), name="lift")
        lifted = obj_pose_base(i)[2] - p[2]
        z_ax = T_from_pose7(grasps[gi])[:3, 2]
        angle = float(np.degrees(np.arccos(np.clip(-z_ax[2], -1, 1))))
        print(f"   grasp {meta[gi]} (approach {angle:.0f} deg from vertical), TCP error at the grasp {e * 1000:.1f} mm, object lifted {lifted:.3f} m", flush=True)
        rec.update(grasp=meta[gi], approach_deg=round(angle, 1), grasped=lifted > 0.07, tcp_err_grasp_mm=round(e * 1000, 1))
        if lifted < 0.07:
            raise RuntimeError("object not in the gripper")

        # ---- 3. carry it to the table and put it down in the orientation it rested in ----
        sync_world()
        planner.set_obstacle_pose(f"obj{i}", obj_pose_base(i))
        assert planner.attach(state["q"], f"obj{i}")
        T_tcp_obj = np.linalg.inv(T_from_pose7(planner.fk(q_meas()))) @ T_from_pose7(obj_pose_base(i))  # as held (incl. slip)
        T_rest = T_from_pose7(p)
        T_rest[2, 3] += -(L.FLOOR_TOP - L.TABLE_Z) + 0.004  # same height above the table as above the crate floor (+4 mm)
        cands, objs_t = [], []
        for yaw in np.radians([0, 60, -60, 120, -120, 180]):  # turn the whole placement about the vertical
            Rz = np.eye(4)
            Rz[:3, :3] = R.from_euler("z", yaw).as_matrix()
            T_obj = Rz @ T_rest
            T_obj[:2, 3] = PLACE_XY[i]
            T_t = T_obj @ np.linalg.inv(T_tcp_obj)
            cands.append(_pose7(T_t) + np.r_[0, 0, 0.10, 0, 0, 0, 0])
            objs_t.append(T_obj)
        carry, k = planner.plan_goalset(state["q"], np.array(cands), tag="to above the place point")
        run(need(carry, "carry"), name="carry")
        place_tcp = cands[k] - np.r_[0, 0, 0.10, 0, 0, 0, 0]
        # down to 2 cm above the resting pose with full collision checking, the last 2 cm with the table ignored: at 4 mm
        # the carried object's spheres are inside cuRobo's 1 cm activation distance of the table top
        run(need(planner.plan_linear(state["q"], place_tcp + np.r_[0, 0, 0.02, 0, 0, 0, 0], frame="base", free_axis=2, max_speed=0.04), "put down"), name="put down")
        with planner.relaxed(ignore=["table"]):
            run(need(planner.plan_cartesian(state["q"], place_tcp, max_speed=0.02), "set down"), name="set down")
        e_place = tcp_err(place_tcp)
        set_grip(0.0, steps=12, settle=6)
        planner.detach()
        sync_world()
        with planner.relaxed(ignore=[f"obj{i}"]):  # back out along the tool axis (not straight up: angled grasps)
            run(need(planner.plan_cartesian(state["q"], pregrasp(place_tcp, 0.10)), "retreat"), name="retreat")
        hold(5)
        q = obj_pose_base(i)
        T_now, T_goal = T_from_pose7(q), objs_t[k]
        rot_err = float(np.degrees((R.from_matrix(T_now[:3, :3]) * R.from_matrix(T_goal[:3, :3]).inv()).magnitude()))
        xy_err = float(np.hypot(*(T_now[:2, 3] - T_goal[:2, 3])))
        ok = xy_err < 0.03 and abs(T_now[2, 3] - T_goal[2, 3]) < 0.015 and rot_err < 20
        print(f"   placed at {np.round(q[:3] + to_base, 3)}: {xy_err * 1000:.0f} mm / {rot_err:.0f} deg from the target pose, "
              f"TCP error at the place point {e_place * 1000:.1f} mm -> {'PLACED' if ok else 'NOT PLACED'}", flush=True)
        rec.update(placed=bool(ok), place_xy_err_mm=round(xy_err * 1000, 1), place_rot_err_deg=round(rot_err, 1), tcp_err_place_mm=round(e_place * 1000, 1))
        sync_world()
        run(need(planner.plan_to_config(state["q"], HOME, tag="home"), "home"), name="home")
    except RuntimeError as ex:
        print(f"   !! {ex}", flush=True)
        rec["error"] = str(ex)
        rec["recovery"] = recover(i)
    results.append({k: (v.item() if isinstance(v, np.generic) else v) for k, v in rec.items()})

print("\nSUMMARY")
for r in results:
    print("  ", r)
args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"sidebins_curobo{args.tag}.mp4"
imageio.mimwrite(path, frames, fps=10, macro_block_size=1)
(args.out / f"sidebins_curobo{args.tag}.json").write_text(json.dumps(dict(results=results, motions=log), indent=1))
print("video:", path, len(frames), "frames", flush=True)
env.close()
# Isaac Sim sometimes hangs in app.close() after a run (two such zombies held 22 GB of GPU 0 for hours): outputs are
# written, so leave hard
sys.stdout.flush()
import os  # noqa: E402

os._exit(0)
