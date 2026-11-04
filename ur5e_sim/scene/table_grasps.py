"""Sampled grasps on randomly placed objects: does the antipodal sampler + cuRobo pick them up in the simulator?

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python table_grasps.py --headless [--trials 5] [--seed 0]

Scene: the side-bins scene (drawers closed) with N short YCB objects on the table in front of the arm. Each trial puts
every object in a random resting pose (one of its stable poses, uniform yaw, random position, no overlaps), lets it
settle, then for each object in random order:
  1. the object's cached grasps (planning/grasps.py, object frame) are moved to its simulated pose, filtered so the open
     gripper stays above the table, and a random subset (weighted by grasp score) is handed to planner.plan_grasp in that
     order, so the chosen grasp varies rather than always being the single best or nearest one;
  2. execute: free motion to the pregrasp, straight approach, close, straight lift by 10 cm;
  3. success = the object rose with the gripper; then it is put back down, released, and the arm backs off.
Reported per pick: object, success, approach angle from vertical, grasp width, planning time.
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--objects", default="005_tomato_soup_can,008_pudding_box,010_potted_meat_can,061_foam_brick")
ap.add_argument("--speed", type=float, default=0.5)
ap.add_argument("--friction", type=float, default=1.0, help="pads and objects; 1.0 with the simple gripper (the four-bar model needed 4)")
ap.add_argument("--grip-k", type=float, default=6.0, help="gripper drive stiffness = squeeze force per rad of remaining closure")
ap.add_argument("--candidates", type=int, default=40, help="grasps handed to the planner per pick")
ap.add_argument("--temperature", type=float, default=0.3, help="softmax temperature on the grasp score for the random order")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/table_grasps"))
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
import trimesh  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "planning"))
import layout as L  # noqa: E402
import robot_defs as n  # noqa: E402
import ur5e_env as E  # noqa: E402
from grasps import T_from_pose7, cached_grasps, graspable_stable_poses, load_mesh, world_filter  # noqa: E402
from planner import ArmPlanner, Plan  # noqa: E402

PHYS = Path("/home/data/siddiquieu1/assets/ycb_physics")
MESHES = Path("/home/data/siddiquieu1/assets/ycb_meshes")
NAMES = args.objects.split(",")
N = len(NAMES)
AREA = ((0.28, 0.62), (-0.22, 0.24))  # world x, y range for object centres
rng = np.random.default_rng(args.seed)
meshes = [load_mesh(MESHES / f"{k}.obj") for k in NAMES]
grasp_lib = [cached_grasps(k, MESHES / f"{k}.obj", n_pairs=500, n_roll=16) for k in NAMES]
rest = [[(T, p) for T, p, _ in graspable_stable_poses(m, g)] for m, g in zip(meshes, grasp_lib)]
for k, r in zip(NAMES, rest):
    assert r, f"{k}: no resting pose in which the 2F-85 can grasp it"
radius = [float(np.linalg.norm(m.extents[:2]) / 2 + 0.01) for m in meshes]  # generous footprint radius (any yaw)
print("grasps per object:", {k: len(g) for k, g in zip(NAMES, grasp_lib)}, flush=True)


def random_layout():
    """[(pose7 world)] for all objects: stable pose x random yaw, centres in AREA, footprints apart."""
    poses, xy = [], []
    for i in range(N):
        Ts, ps = zip(*rest[i])
        T = np.array(Ts[rng.choice(len(Ts), p=np.array(ps) / sum(ps))])
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
        T = Rz @ T
        for _ in range(200):
            p = np.array([rng.uniform(*AREA[0]), rng.uniform(*AREA[1])])
            if all(np.linalg.norm(p - q) > radius[i] + radius[j] for j, q in enumerate(xy)):
                break
        xy.append(p)
        T[:2, 3] += p
        T[2, 3] += L.TABLE_Z + 0.003
        x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
        poses.append(np.r_[T[:3, 3], w, x, y, z])
    return poses


layout0 = random_layout()


def obj_cfg(i):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(PHYS / f"{NAMES[i]}.usd"), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(layout0[i][:3]), rot=tuple(layout0[i][3:])),
    )


EYE_A, EYE_B = (1.25, 0.55, 1.45), (0.45, -0.25, 1.75)
assert N <= 6, "at most 6 objects"


@configclass
class SceneCfg(E.SceneCfg):
    cube = None
    obj0: RigidObjectCfg = obj_cfg(0) if N > 0 else None
    obj1: RigidObjectCfg = obj_cfg(1) if N > 1 else None
    obj2: RigidObjectCfg = obj_cfg(2) if N > 2 else None
    obj3: RigidObjectCfg = obj_cfg(3) if N > 3 else None
    obj4: RigidObjectCfg = obj_cfg(4) if N > 4 else None
    obj5: RigidObjectCfg = obj_cfg(5) if N > 5 else None
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(size=(L.TABLE_SIZE[0], L.TABLE_SIZE[1], L.TABLE_Z), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, L.TABLE_Z / 2)),
    )
    cam_a = E._camera("cam_a", EYE_A, 13.0, rot=E.look_at_quat(EYE_A, (0.42, 0.0, 0.78)))
    cam_b = E._camera("cam_b", EYE_B, 13.0, rot=E.look_at_quat(EYE_B, (0.45, 0.0, 0.75)))


@configclass
class EnvCfg(E.Ur5eTableJointEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=8.0)


to_base = np.array([0.0, 0.0, L.TABLE_Z])
objects0 = [(f"obj{i}", MESHES / f"{NAMES[i]}.obj", np.r_[layout0[i][:3] - to_base, layout0[i][3:]]) for i in range(N)]
world = L.world_config(bins=(), objects=objects0)
planner = ArmPlanner(world, time_dilation=args.speed)
HOME = planner.home

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
FINGER_IDS = robot.find_joints(list(E.GRIPPER_DRIVEN) if args.gripper == "linkage" else ["left_pad_slide", "right_pad_slide"], preserve_order=True)[0]
PAD_IDS = robot.find_bodies(["left_inner_finger_pad", "right_inner_finger_pad"] if args.gripper == "linkage" else ["left_pad_phys", "right_pad_phys"], preserve_order=True)[0]
objs = [env.scene[f"obj{i}"] for i in range(N)]


def set_friction(asset, mu):
    m = asset.root_physx_view.get_material_properties()
    m[..., 0], m[..., 1] = mu, mu
    asset.root_physx_view.set_material_properties(m, torch.arange(m.shape[0]))


set_friction(robot, args.friction)
for o in objs:
    set_friction(o, args.friction)

frames = []
state = {"q": np.array(HOME, float), "grip": 0.0}


def q_meas():
    return robot.data.joint_pos[0, ARM_IDS].cpu().numpy()


def step(q, grip):
    env.step(torch.tensor([[*q, grip]], dtype=torch.float32, device=dev))
    frames.append(np.concatenate([env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for c in ("cam_a", "cam_b", "cam_wrist")], 1))


def run(plan: Plan, grip=None):
    g = state["grip"] if grip is None else grip
    for q in plan.q[1:]:
        step(q, g)
    state["q"] = plan.q[-1].copy()


def hold(k):
    for _ in range(k):
        step(state["q"], state["grip"])


def set_grip(target, steps=15, settle=4):
    g0 = state["grip"]
    for k in range(1, steps + 1):
        step(state["q"], g0 + (target - g0) * k / steps)
    state["grip"] = target
    hold(settle)


def pose_base(i):
    return np.r_[objs[i].data.root_pos_w[0].cpu().numpy() - to_base, objs[i].data.root_quat_w[0].cpu().numpy()]


def sync_world():
    for i in range(N):
        if planner.attached != f"obj{i}":
            planner.set_obstacle_pose(f"obj{i}", pose_base(i))


def place_objects(poses):
    for i, p in enumerate(poses):
        objs[i].write_root_pose_to_sim(torch.tensor([p], dtype=torch.float32, device=dev))
        objs[i].write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    hold(15)


def go_home():
    p = planner.plan_to_config(state["q"], HOME, tag="home")
    if p is not None:
        run(p)
        return True
    robot.write_joint_state_to_sim(torch.tensor([list(HOME)], device=dev), torch.zeros(1, 6, device=dev), joint_ids=ARM_IDS)
    state["q"] = np.array(HOME, float)
    env.action_manager.get_term("arm").reset()
    hold(5)
    return False


import time  # noqa: E402

picks = []
hold(10)
for trial in range(args.trials):
    if trial > 0:
        place_objects(random_layout())
    print(f"\n=== trial {trial} ===", flush=True)
    for i in rng.permutation(N):
        sync_world()
        p0 = pose_base(i)
        T_obj = T_from_pose7(p0)
        G = grasp_lib[i]
        Ts = np.array([T_obj @ g.T for g in G])
        keep = world_filter(Ts, floor_z=0.0, min_clear=0.004, widths=[g.width for g in G])
        idx = np.where(keep)[0]
        rec = dict(trial=trial, object=NAMES[i], n_grasps=len(G), above_table=int(len(idx)))
        if len(idx) == 0:
            rec.update(success=False, error="no grasp keeps the gripper above the table")
            picks.append(rec)
            print(f"   {NAMES[i]}: no grasp above the table", flush=True)
            continue
        sc = np.array([G[k].score for k in idx])
        pr = np.exp((sc - sc.max()) / args.temperature)
        order = rng.choice(idx, size=min(args.candidates, len(idx)), replace=False, p=pr / pr.sum())
        cand = np.array([np.r_[Ts[k][:3, 3], R.from_matrix(Ts[k][:3, :3]).as_quat()[[3, 0, 1, 2]]] for k in order])
        t = time.time()
        to_pre, app_, gi = planner.plan_grasp(state["q"], cand, approach=0.10, target=f"obj{i}", order="given", max_tries=10)
        rec["plan_s"] = round(time.time() - t, 2)
        if app_ is None:
            rec.update(success=False, error="planning failed")
            picks.append(rec)
            print(f"   {NAMES[i]}: planning failed ({rec['plan_s']} s)", flush=True)
            continue
        g = G[order[gi]]
        approach_dir = Ts[order[gi]][:3, 2]
        angle = float(np.degrees(np.arccos(np.clip(-approach_dir[2], -1, 1))))  # 0 = straight down
        run(to_pre, grip=0.0)
        moved_pre = np.linalg.norm(pose_base(i)[:3] - p0[:3])
        run(app_)
        moved_app = np.linalg.norm(pose_base(i)[:3] - p0[:3])
        tcp_e = np.linalg.norm(planner.fk(q_meas())[:3] - cand[gi][:3])
        set_grip(E.GRIPPER_CLOSED, steps=15, settle=4)
        fing = robot.data.joint_pos[0, FINGER_IDS].cpu().numpy()
        pads = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy() - to_base
        # pads in the grasp frame: x should be ~ +-width/2, y and z ~ 0
        Tg = T_from_pose7(cand[gi])
        pads_g = (np.linalg.inv(Tg) @ np.c_[pads, np.ones(2)].T).T[:, :3]
        moved_close = np.linalg.norm(pose_base(i)[:3] - p0[:3])
        print(f"      object moved {moved_pre * 1000:.0f} mm by the transit, {moved_app * 1000:.0f} mm after the approach, {moved_close * 1000:.0f} mm after closing; "
              f"TCP error {tcp_e * 1000:.1f} mm; finger joints {np.round(fing, 3)} (rad, or m for the simple pads); pads in grasp frame [mm] {np.round(pads_g * 1000).astype(int).tolist()}", flush=True)
        with planner.relaxed(ignore=[f"obj{i}"]):
            here = planner.fk(state["q"])
            lift = planner.plan_cartesian(state["q"], here + np.r_[0, 0, 0.10, 0, 0, 0, 0], max_speed=0.05, quiet=True)
            if lift is not None:
                run(lift)
                hold(5)
                rose = pose_base(i)[2] - p0[2]
                down = planner.plan_cartesian(state["q"], here + np.r_[0, 0, 0.005, 0, 0, 0, 0], max_speed=0.04, quiet=True)
                if down is not None:
                    run(down)
            else:
                rose = float("nan")
            set_grip(0.0, steps=10, settle=4)
            out = planner.plan_cartesian(state["q"], cand[gi] - np.r_[0.10 * approach_dir, 0, 0, 0, 0], quiet=True)
            if out is not None:
                run(out)
        ok = bool(rose > 0.07)
        rec.update(success=ok, lifted_m=round(float(rose), 3), approach_deg_from_vertical=round(angle, 1), width_mm=round(g.width * 1000, 1),
                   grasp_index=int(order[gi]), score=round(g.score, 3))
        picks.append(rec)
        print(f"   {NAMES[i]:22s} grasp #{order[gi]:4d} (approach {angle:5.1f} deg from vertical, width {g.width * 1000:4.1f} mm), "
              f"planned in {rec['plan_s']:.1f} s, lifted {rose:.3f} m -> {'OK' if ok else 'FAILED'}", flush=True)
        if not go_home():
            rec["arm_reset"] = True

ok = [p for p in picks if p.get("success")]
ang = np.array([p["approach_deg_from_vertical"] for p in picks if "approach_deg_from_vertical" in p])
print(f"\nSUMMARY: {len(ok)}/{len(picks)} picks lifted; planning failed {sum(p.get('error') == 'planning failed' for p in picks)}; "
      f"grasp executed but object not lifted {sum(('approach_deg_from_vertical' in p) and not p['success'] for p in picks)}")
if len(ang):
    print(f"approach angle from vertical: {np.histogram(ang, bins=[0, 15, 30, 45, 60, 75, 90, 180])[0].tolist()} for bins [0,15,30,45,60,75,90,180] deg")
for k in NAMES:
    s = [p for p in picks if p["object"] == k]
    print(f"   {k:22s} {sum(p.get('success', False) for p in s)}/{len(s)}")
args.out.mkdir(parents=True, exist_ok=True)
imageio.mimwrite(args.out / f"table_grasps{args.tag}.mp4", frames, fps=10, macro_block_size=1)
(args.out / f"table_grasps{args.tag}.json").write_text(json.dumps(picks, indent=1))
print("video:", args.out / f"table_grasps{args.tag}.mp4", len(frames), "frames", flush=True)
env.close()
# Isaac Sim sometimes hangs in app.close() after a run (two such zombies held 22 GB of GPU 0 for hours): outputs are
# written, so leave hard
sys.stdout.flush()
import os  # noqa: E402

os._exit(0)
