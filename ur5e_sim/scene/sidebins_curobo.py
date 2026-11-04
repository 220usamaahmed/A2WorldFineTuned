"""Side-bins task driven by cuRobo: open each drawer (hook the crate's rim bar, pull), pick the object, put it on the table.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python sidebins_curobo.py --headless [--bins 0,1,2] [--speed 0.5]

Same scene as sidebins_test.py (layout.py), but the arm is driven in joint space (ur5e_env.JointActionsCfg) along
trajectories from planning/planner.py instead of interpolated TCP poses through differential IK:
  * free-space motions: cuRobo MotionGen with the table, the housings, the crates (at their current slide) and the
    objects (YCB meshes at their current poses) as obstacles; a carried object is attached to the robot;
  * approach / lift / pull / place moves: straight TCP lines (planner.plan_linear / plan_cartesian);
  * grasp: top-down at the object (two candidates, fingers along +x or -x), chosen by planner.plan_grasp.
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
ap.add_argument("--friction", type=float, default=4.0, help="pads and objects (the simulated squeeze is weak)")
ap.add_argument("--can-y", default="-0.04,0.0,0.04", help="crate-local y of the soup can in each bin (sidebins_test.py: -0.04,0,0.04)")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/curobo"))
ap.add_argument("--tag", default="")
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
from planner import ArmPlanner, Plan  # noqa: E402

BIN_USD = HERE.parent / "assets" / "usd" / "pullbin.usd"
YCB = Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics")
MESHES = Path("/home/data/siddiquieu1/assets/ycb_meshes")
# table, in front of the arm (world x, y); a pulled-out crate reaches over the table to y = -0.17 (bin 2 spans x 0.27-0.67),
# so the spots stay >= 5 cm clear of it (the old -0.12 left 14 mm between the can and the crate)
PLACE_XY = [(0.40, 0.14), (0.40, 0.02), (0.40, -0.08)]
# (name, usd, mesh, height, crate-local (x, y), yaw): stood up by +90 deg about x (the files' long axis is y)
CAN = ("soup can", "005_tomato_soup_can.usd", "005_tomato_soup_can_physics.obj", 0.102)
OBJECTS = [(*CAN, (0.10, float(y)), 0) for y in args.can_y.split(",")]
Q_PICK = np.array([0.0, 1.0, 0.0, 0.0])  # tool down, fingers close along world x
Q_PICK_FLIP = np.array([0.0, 0.0, 1.0, 0.0])  # tool down, fingers along -x (180 deg about z from Q_PICK)
Q_HOOK = np.array([*R.from_matrix(np.stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]], 1)).as_quat()[[3, 0, 1, 2]]])  # fingers along y


def upright_quat(yaw_deg):
    x, y, z, w = (R.from_euler("z", yaw_deg, degrees=True) * R.from_euler("x", 90, degrees=True)).as_quat()
    return (float(w), float(x), float(y), float(z))


def bin_cfg(i):
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/bin{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(BIN_USD), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(L.BIN_X[i], L.ORIGIN_Y, 0.0), rot=L.HOUSING_QUAT, joint_pos={"crate_slide": 0.0}),
        actuators={"slide": ImplicitActuatorCfg(joint_names_expr=["crate_slide"], stiffness=0.0, damping=args.slide_damping, effort_limit_sim=500.0)},
    )


def obj_cfg(i):
    name, usd, _, h, (xc, yc), yaw = OBJECTS[i]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(YCB / usd), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(L.crate_to_world(i, xc, yc, L.FLOOR_TOP + h / 2 + 0.003)), rot=upright_quat(yaw)),
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
objects0 = [(f"obj{i}", MESHES / OBJECTS[i][2], np.r_[L.crate_to_world(i, *OBJECTS[i][4], L.FLOOR_TOP + OBJECTS[i][3] / 2 + 0.003) - to_base,
                                                        upright_quat(OBJECTS[i][5])]) for i in range(3)]
planner = ArmPlanner(L.world_config(objects=objects0), time_dilation=args.speed)
HOME = planner.home
print("planner ready; home", np.round(HOME, 3), flush=True)

cfg = EnvCfg()
cfg.scene.robot.init_state.joint_pos.update(dict(zip(n.ARM_JOINTS, map(float, HOME))))
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
    name, _, _, h, _, _ = OBJECTS[i]
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

        # ---- 2. pick the object from above ----
        set_grip(0.0, steps=8)
        hold(5)
        sync_world()
        p = obj_pose_base(i)
        z0 = p[2]
        gz = max(0.01, h / 2 - 0.042)  # palm and wrist camera sit ~4 cm above the TCP
        grasps = np.array([np.r_[p[:3] + [0, 0, gz], q] for q in (Q_PICK, Q_PICK_FLIP)])
        to_pre, approach, gi = planner.plan_grasp(state["q"], grasps, approach=0.10, target=f"obj{i}")
        need(approach, "grasp")
        run(to_pre, grip=0.0, name="to pregrasp")
        run(approach, name="approach")
        e = tcp_err(grasps[gi])
        set_grip(E.GRIPPER_CLOSED, steps=15, settle=4)
        with planner.relaxed(ignore=[f"obj{i}"]):
            up = grasps[gi] + np.r_[0, 0, 0.10, 0, 0, 0, 0]
            run(need(planner.plan_cartesian(state["q"], up, max_speed=0.05), "lift"), name="lift")
        lifted = obj_pose_base(i)[2] - z0
        print(f"   grasp candidate {gi}, TCP error at the grasp {e * 1000:.1f} mm, object lifted {lifted:.3f} m", flush=True)
        rec.update(grasped=lifted > 0.08, tcp_err_grasp_mm=round(e * 1000, 1))
        if lifted < 0.08:
            raise RuntimeError("object not in the gripper")

        # ---- 3. carry it to the table and put it down ----
        sync_world()
        planner.set_obstacle_pose(f"obj{i}", obj_pose_base(i))
        assert planner.attach(state["q"], f"obj{i}")
        T_tcp = planner.fk(q_meas())
        obj_now = obj_pose_base(i)
        dz_obj = obj_now[2] - T_tcp[2]  # object centre relative to the TCP (tool pointing down, object upright)
        place_tcp = np.r_[PLACE_XY[i][0], PLACE_XY[i][1], h / 2 + 0.006 - dz_obj, T_tcp[3:]]
        run(need(planner.plan_pose(state["q"], place_tcp + np.r_[0, 0, 0.10, 0, 0, 0, 0], tag="to the place point"), "carry"), name="carry")
        run(need(planner.plan_linear(state["q"], place_tcp, frame="base", free_axis=2, max_speed=0.04), "put down"), name="put down")
        e_place = tcp_err(place_tcp)
        set_grip(0.0, steps=12, settle=6)
        planner.detach()
        sync_world()
        with planner.relaxed(ignore=[f"obj{i}"]):
            run(need(planner.plan_linear(state["q"], place_tcp + np.r_[0, 0, 0.10, 0, 0, 0, 0], frame="base", free_axis=2), "retreat"), name="retreat")
        hold(5)
        q = obj_pose_base(i)
        upright = R.from_quat(q[3:7][[1, 2, 3, 0]]).apply([0, 1, 0])[2]
        ok = abs(q[2] - h / 2) < 0.02 and np.hypot(q[0] - PLACE_XY[i][0], q[1] - PLACE_XY[i][1]) < 0.03 and upright > 0.9
        print(f"   placed at {np.round(q[:3] + to_base, 3)}, upright {upright:.2f}, TCP error at the place point {e_place * 1000:.1f} mm -> {'PLACED' if ok else 'NOT PLACED'}", flush=True)
        rec.update(placed=bool(ok), tcp_err_place_mm=round(e_place * 1000, 1))
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
app.close()
