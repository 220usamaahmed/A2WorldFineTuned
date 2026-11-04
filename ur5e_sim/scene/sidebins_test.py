"""Three pull-out bins along one side of the table, arm at the table's centre: can it open each and pick from it?

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 python sidebins_test.py --headless [--bins 0,1,2] [--pull 0.15]

Layout (world frame, z up, arm base on the table top at the origin, table top z = 0.75):
  * table 1.8 x 0.8 m centred on the arm
  * three housings (build_pullbin.py, base plate at table height) in a row along the -y side, their fronts flush with the table
    edge (y = -0.40) and facing +y, so a pulled crate slides out over the table top
  * one YCB object stands in each crate
Per bin: hook the crate's end-wall rim bar and pull it out, then pick the object from above and put it on the table in front of the arm.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--bins", default="0,1,2", help="which bins to work on (0 = -x, 1 = centre, 2 = +x)")
ap.add_argument("--pull", type=float, default=0.15, help="how far to pull each crate [m]")
ap.add_argument("--table-edge-y", type=float, default=-0.30, help="world y of the table edge the housings touch")
ap.add_argument("--no-pick", action="store_true", help="only open the drawers")
ap.add_argument("--hook-gap", type=float, default=0.040, help="finger gap while straddling the wall [m]")
ap.add_argument("--latch-stiffness", type=float, default=3000.0, help="hold stiffness of the crate between pulls [N/m]")
ap.add_argument("--grip-k", type=float, default=6.0, help="gripper drive stiffness (squeeze/hold force); ur5e_env uses 6 for light objects")
ap.add_argument("--follow-mult", type=float, default=1.0, help="multiplier on the gripper follower (finger/knuckle) joint gains")
ap.add_argument("--slide-damping", type=float, default=15.0, help="viscous resistance of the crate slide [N s/m]; also what keeps it from creeping")
ap.add_argument("--friction", type=float, default=4.0, help="friction of the fingertips and objects; the simulated gripper squeezes weakly, so a 0.35 kg can needs about 4 (1.5 slips)")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/sidebins"))
ap.add_argument("--tag", default="")
ap.add_argument("--pick-flip-bins", default="2", help="bins whose pick uses the tool rolled by 180 deg (fingers swapped); avoids a wrist configuration the arm cannot reach on the +x side")
ap.add_argument("--cans", action="store_true", help="put a soup can in every crate (the other objects are tall and tip over under this gripper)")
ap.add_argument("--close-hold", type=int, default=3, help="control steps to wait after closing the fingers, before lifting")
ap.add_argument("--lift-lin", type=float, default=0.008, help="lift/carry speed with an object in the gripper [m per control step]")
ap.add_argument("--no-bind", action="store_true", help="debug: do not bind the high-friction material to fingers and objects")
ap.add_argument("--cube-test", action="store_true", help="debug: object 0 is the 5 cm cube the gripper was tuned on")
ap.add_argument("--table-grasp", action="store_true", help="debug: grasp each object standing on the plain table instead of in a crate")
ap.add_argument("--air-test", action="store_true", help="debug: close the gripper in free air and report the pad gap")
ap.add_argument("--flip-bins", default="", help="bins worked with the tool rolled by 180 deg (fingers swapped)")
ap.add_argument("--no-objects", action="store_true", help="debug: park the objects far away")
ap.add_argument("--creep-test", action="store_true", help="debug: crate set to 0.15 m, arm idle, watch whether it drifts")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation, Slerp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import ur5e_env as E  # noqa: E402

BIN_USD = HERE.parent / "assets" / "usd" / "pullbin.usd"
YCB = Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics")
# ---- must match build_pullbin.py ----
Z_B, CRATE_Z0, CH, T, IN_HALF_W = E.TABLE_Z, 0.002, 0.1703, 0.02, 0.215
WALL_IN_X, BAR_OUT_X, BAR_Z0 = 0.290, 0.3026, 0.144
FRONT_X = 0.32
FLOOR_TOP = Z_B + CRATE_Z0 + 0.006  # inside floor of the crate (world z)

PITCH = 2 * (IN_HALF_W + T)  # housings side by side
BIN_X = [-PITCH, 0.0, PITCH]
ORIGIN_Y = args.table_edge_y - FRONT_X  # housing frame origin (world y); the housing is yawed +90 deg: local +x -> world +y
WALL_Y = ORIGIN_Y + 0.5 * (WALL_IN_X + BAR_OUT_X)  # middle of the end wall incl. rim bar (world y), crate closed
BAR_Z = Z_B + CRATE_Z0 + 0.5 * (BAR_Z0 + CH)  # middle of the rim bar (world z)
PLACE_XY = [(0.32, -0.14), (0.32, 0.0), (0.32, 0.14)]  # where each object is put down: on the table in front of the arm
TRANSIT_Z = 1.08  # world z at which the tool travels (clears roofs, walls and the objects)

# objects: (name, usd, height, crate-local (x, y), yaw about the vertical [deg]); stood upright by a 90 deg rotation about x,
# then turned so that the thin side faces world x: the fingers close across the crate's width, away from the crate's end walls
# (the gripper's outer finger links stick out ~4 cm beyond the pads and hit the end wall when closing along the pull axis)
OBJECTS = [
    ("soup can", "005_tomato_soup_can.usd", 0.102, (0.10, 0.00), 0),
    ("mustard bottle", "006_mustard_bottle.usd", 0.192, (0.08, 0.04), 90),
    ("sugar box", "004_sugar_box.usd", 0.176, (0.10, -0.04), 90),
]


if args.cans:
    OBJECTS = [("soup can", "005_tomato_soup_can.usd", 0.102, (0.10, y), 0) for y in (-0.04, 0.0, 0.04)]


def upright_quat(yaw_deg):
    x, y, z, w = (Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("x", 90, degrees=True)).as_quat()
    return (float(w), float(x), float(y), float(z))


def crate_to_world(i, xc, yc, z):
    """Crate-local (x along the pull axis, y) of bin i, closed position -> world."""
    return (BIN_X[i] - yc, ORIGIN_Y + xc, z)


def bin_cfg(i):
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/bin{i}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BIN_USD),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(BIN_X[i], ORIGIN_Y, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), joint_pos={"crate_slide": 0.0}
        ),
        actuators={"slide": ImplicitActuatorCfg(joint_names_expr=["crate_slide"], stiffness=0.0, damping=args.slide_damping, effort_limit_sim=500.0)},
    )


def obj_cfg(i):
    name, usd, h, (xc, yc), yaw = OBJECTS[i]
    if args.cube_test and i == 0:
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.05,) * 3, rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12)),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5, restitution=0.0, friction_combine_mode="max"),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=crate_to_world(i, xc, yc, FLOOR_TOP + 0.025 + 0.003)),
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj{i}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(YCB / usd), rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0 + i, 3.0, 0.3) if args.no_objects else crate_to_world(i, xc, yc, FLOOR_TOP + h / 2 + 0.003), rot=upright_quat(yaw)),
    )


EYE_A, EYE_B = (0.0, 1.05, 1.85), (1.55, -0.55, 1.45)  # overview cameras


@configclass
class SideBinsSceneCfg(E.SceneCfg):
    cube = None
    robot: ArticulationCfg = E._robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 2 * abs(args.table_edge_y), E.TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, E.TABLE_Z / 2)),
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
class SideBinsEnvCfg(E.Ur5eTableEnvCfg):
    scene: SideBinsSceneCfg = SideBinsSceneCfg(num_envs=1, env_spacing=8.0)


def tool_quat(flip: bool = False) -> np.ndarray:
    """(w, x, y, z): tool z -> world -z (pointing down), tool x (the finger closing axis) -> world +y (the pull axis), or -y if flip.
    The two are the same grasp with the fingers swapped; they differ by 180 deg of wrist roll, which matters near joint limits."""
    cols = np.stack([[0, -1, 0] if flip else [0, 1, 0], [-1, 0, 0] if flip else [1, 0, 0], [0, 0, -1]], axis=1)
    x, y, z, w = Rotation.from_matrix(cols).as_quat()
    return np.array([w, x, y, z])


cfg = SideBinsEnvCfg()
cfg.scene.robot.actuators["gripper"].stiffness = args.grip_k
cfg.scene.robot.actuators["gripper"].damping = 0.05 * args.grip_k
for _a in ("gripper_fingers", "gripper_knuckles"):
    cfg.scene.robot.actuators[_a].stiffness *= args.follow_mult
    cfg.scene.robot.actuators[_a].damping *= args.follow_mult
env = ManagerBasedRLEnv(cfg)
env.reset()
dev = env.device
robot = env.scene["robot"]
# grippy fingers and objects. A friction set on the shapes themselves is what works (binding a material to the prims did not
# take effect); the default combine mode is the average, so both sides get the same value.
def set_friction(asset, mu):
    m = asset.root_physx_view.get_material_properties()  # [envs, shapes, (static, dynamic, restitution)]
    m[..., 0], m[..., 1] = mu, mu
    asset.root_physx_view.set_material_properties(m, torch.arange(m.shape[0]))


if not args.no_bind:
    set_friction(robot, args.friction)
    for i in range(3):
        set_friction(env.scene[f"obj{i}"], args.friction)
    _mr = robot.root_physx_view.get_material_properties()
    _mo = env.scene["obj0"].root_physx_view.get_material_properties()
    print("friction readback: robot shapes", tuple(_mr.shape), "static", _mr[..., 0].unique().tolist(), "| obj0 shapes", tuple(_mo.shape), "static", _mo[..., 0].unique().tolist())

_gj = robot.find_joints(list(E.GRIPPER_DRIVEN) + list(E.GRIPPER_FOLLOWERS))[0]
print("gripper joints:", [robot.joint_names[j] for j in _gj], "\n  effort limit", robot.data.joint_effort_limits[0, _gj].cpu().numpy(), "stiffness", robot.data.joint_stiffness[0, _gj].cpu().numpy(),
      "damping", robot.data.joint_damping[0, _gj].cpu().numpy(), "pos limits", robot.data.joint_pos_limits[0, _gj].cpu().numpy().tolist(), "friction", robot.data.joint_friction_coeff[0, _gj].cpu().numpy())
frames = []
state = {"pos": np.array(E.READY_TCP), "quat": np.array(E.TOOL_DOWN_QUAT), "grip": 0.0}
GRIP_MAX, Q = E.GRIPPER_CLOSED, tool_quat()
Q_PICK = np.array(E.TOOL_DOWN_QUAT)  # tool x -> world x: the fingers close across the crate's width
Q_PICK_FLIP = np.array([*Rotation.from_matrix(np.stack([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], axis=1)).as_quat()[[3, 0, 1, 2]]])  # same, rolled 180 deg
Q_FLIP = tool_quat(True)
CURQ = Q
PAD_IDS = robot.find_bodies(["left_inner_finger_pad", "right_inner_finger_pad"])[0]


def gap_to_angle(gap):  # Robotiq 2F-85: 0 rad = 85 mm open, 0.8 rad = closed
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, GRIP_MAX))


TRACE = {"on": False, "obj": None, "n": 0}


def step(pos, quat, grip):
    env.step(torch.tensor([[*pos, *quat, grip]], dtype=torch.float32, device=dev))
    if TRACE["on"]:
        TRACE["n"] += 1
        if TRACE["n"] % 2 == 0:
            fj = robot.find_joints(list(E.GRIPPER_DRIVEN))[0]
            pp = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy()
            print(f"      trace {TRACE['n']:3d} cmd {grip:.2f} tcp z {pos[2] + E.TABLE_Z:.3f} fingers {np.round(robot.data.joint_pos[0, fj].cpu().numpy(), 2)} "
                  f"pad x {np.round(pp[:, 0], 3)} obj {np.round(TRACE['obj'].data.root_pos_w[0].cpu().numpy(), 3)}")
    rgb = [env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for c in ("cam_dbg", "cam_side", "cam_wrist")]
    frames.append(np.concatenate(rgb, 1))


def move(w, grip, lin=0.01, ang=0.05, hold=3, smooth=False):
    """Move the TCP to world position w (x, y, z) with the tool pointing down, interpolating the commanded pose."""
    pos = np.array([w[0], w[1], w[2] - E.TABLE_Z], float)  # arm-base frame
    p0, q0, g0 = state["pos"], state["quat"], state["grip"]
    r0, r1 = Rotation.from_quat(q0[[1, 2, 3, 0]]), Rotation.from_quat(CURQ[[1, 2, 3, 0]])
    n = max(2, int(np.ceil(max(np.linalg.norm(pos - p0) / lin, (r1 * r0.inv()).magnitude() / ang, abs(grip - g0) / 0.05))))
    sl = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    for k in range(1, n + 1):
        t = k / n
        u = t * t * (3 - 2 * t) if smooth else t  # smoothstep: no jerk at the ends, so a carried object is not shaken out
        qx, qy, qz, qw = sl([u]).as_quat()[0]
        step(p0 + u * (pos - p0), np.array([qw, qx, qy, qz]), g0 + t * (grip - g0))
    for _ in range(hold):
        step(pos, CURQ, grip)
    state.update(pos=pos, quat=CURQ, grip=grip)


def grip_dump(tag):
    ids = robot.find_joints(list(E.GRIPPER_DRIVEN) + list(E.GRIPPER_FOLLOWERS))[0]
    names = [robot.joint_names[j] for j in ids]
    q, t = robot.data.joint_pos[0, ids].cpu().numpy(), robot.data.joint_pos_target[0, ids].cpu().numpy()
    print(f"   {tag}: " + ", ".join(f"{n_.replace('_joint', '')} {a:.2f}/{b:.2f}" for n_, a, b in zip(names, q, t)) + "  (pos/target)")


def pad_mid():
    return robot.data.body_pos_w[0, PAD_IDS].mean(0).cpu().numpy()


def track_err(w, tag):
    """How far the fingertip midpoint is from the commanded point (the pad frame sits ~1 cm below the TCP)."""
    e = pad_mid() - np.asarray(w) + np.array([0, 0, 0.011])
    flag = "" if np.linalg.norm(e) < 0.02 else "   <-- NOT REACHED"
    ti = robot.find_bodies("ur5e_tool0")[0][0]
    qt = robot.data.body_quat_w[0, ti].cpu().numpy()
    tilt = np.degrees(np.arccos(np.clip(-Rotation.from_quat(qt[[1, 2, 3, 0]]).apply([0, 0, 1])[2], -1, 1)))
    print(f"   {tag}: tracking error [mm] {np.round(e * 1000).astype(int)}, tool tilt from vertical {tilt:.1f} deg{flag}")
    return float(np.linalg.norm(e))


def latch(bin_, on: bool):
    """Stand-in for a drawer detent: hold the crate where it is (stiff position hold) or let it slide freely."""
    k = args.latch_stiffness if on else 0.0
    bin_.write_joint_stiffness_to_sim(torch.tensor([[k]], device=dev))
    if on:
        bin_.set_joint_position_target(bin_.data.joint_pos.clone())


def objinfo(obj, tag):
    q = obj.data.root_quat_w[0].cpu().numpy()
    up = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0, 1, 0])[2]  # the file's long axis is y; it points up when the object stands
    b = env.scene[f"bin{CUR}"]
    print(f"   [{tag}] object at {np.round(obj.data.root_pos_w[0].cpu().numpy(), 3)}, long axis up {up:.2f}, crate out {float(b.data.joint_pos[0, 0]):.3f}")


CUR = 0
if args.air_test:
    move((0.2, 0.2, TRANSIT_Z), 0.0, hold=15)
    for ang in (0.0, 0.3, 0.5, 0.75):
        move((0.2, 0.2, TRANSIT_Z), ang, hold=15)
        pp = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy()
        grip_dump(f"air command {ang:.2f}")
        print(f"air test: command {ang:.2f} -> pad centres y {np.round(pp[:, 1], 3)} (separation {abs(pp[0, 1] - pp[1, 1]) * 1000:.1f} mm, inner gap {abs(pp[0, 1] - pp[1, 1]) * 1000 - 6.3:.1f} mm), fingers {np.round(robot.data.joint_pos[0, robot.find_joints(list(E.GRIPPER_DRIVEN))[0]].cpu().numpy(), 2)}")
    env.close()
    app.close()
    sys.exit(0)
if args.creep_test:
    for b in range(3):
        env.scene[f"bin{b}"].write_joint_state_to_sim(torch.tensor([[0.15]], device=dev), torch.zeros(1, 1, device=dev))
    b1 = env.scene["bin1"]
    b1.write_joint_stiffness_to_sim(torch.tensor([[3000.0]], device=dev))
    b1.set_joint_position_target(torch.tensor([[0.10]], device=dev))
    print("stiffness readback:", b1.data.joint_stiffness[0].cpu().numpy(), "physx:", b1.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy(), "target", b1.data.joint_pos_target[0].cpu().numpy())
    for k in range(60):
        step(state["pos"], state["quat"], 0.0)
        if k % 10 == 0:
            print("creep test", k, [round(float(env.scene[f"bin{b}"].data.joint_pos[0, 0]), 4) for b in range(3)], flush=True)
    env.close()
    app.close()
    sys.exit(0)
results = []
for i in [int(b) for b in args.bins.split(",")]:
    name, _, h, _, _ = OBJECTS[i]
    if args.cube_test and i == 0:
        name, h = "cube", 0.05
    CUR = i
    CURQ = Q_FLIP if i in [int(b) for b in args.flip_bins.split(',') if b] else Q
    bin_ = env.scene[f"bin{i}"]
    sj = bin_.joint_names.index("crate_slide")
    obj = env.scene[f"obj{i}"]
    print(f"\n=== bin {i} at x = {BIN_X[i]:+.2f}  ({name}) ===")
    wall = np.array([BIN_X[i], WALL_Y, BAR_Z])
    print(f"   rim bar {np.round(wall, 3)}, distance from the arm axis {np.hypot(wall[0], wall[1]):.2f} m closed, "
          f"{np.hypot(wall[0], wall[1] + args.pull):.2f} m after the pull")
    g_hook = gap_to_angle(args.hook_gap)
    if args.table_grasp:
        pose7 = torch.tensor([[PLACE_XY[i][0], PLACE_XY[i][1], E.TABLE_Z + h / 2 + 0.004, *upright_quat(OBJECTS[i][4])]], device=dev, dtype=torch.float32)
        obj.write_root_pose_to_sim(pose7)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        for _ in range(20):
            step(state["pos"], state["quat"], 0.0)
        print(f"   [table grasp] {name} placed on the table at {np.round(obj.data.root_pos_w[0].cpu().numpy(), 3)}")
    rec = dict(bin=i, object=name)
    if not args.table_grasp:
        # ---- 1. hook the rim bar and pull ----
        move((wall[0], wall[1], TRANSIT_Z), 0.0, lin=0.012)
        move((wall[0], wall[1], wall[2] + 0.12), g_hook)
        move(wall, g_hook, lin=0.004, hold=12)
        e1 = track_err(wall, "at the wall")
        move(wall, GRIP_MAX, hold=8)
        for d in np.linspace(0, args.pull, 8)[1:]:
            move(wall + np.array([0, d, 0]), GRIP_MAX, hold=2)
        e2 = track_err(wall + np.array([0, args.pull, 0]), "end of the pull")
        out = float(bin_.data.joint_pos[0, sj])
        print(f"   crate pulled out {out:.3f} m (target {args.pull:.2f})  -> {'OPEN' if out > 0.8 * args.pull else 'FAILED'}")
        latch(bin_, True)  # a drawer detent: the crate stays where the arm left it
        move(wall + np.array([0, args.pull, 0]), g_hook, hold=4)  # let go
        for zz in (wall[2] + 0.03, wall[2] + 0.08, TRANSIT_Z):  # and lift off
            move((wall[0], wall[1] + args.pull, zz), 0.0, lin=0.012, hold=6)
        move((wall[0], wall[1] + args.pull, TRANSIT_Z), 0.0, hold=10)
        print(f"   crate stays at {float(bin_.data.joint_pos[0, sj]):.3f} m after the arm has left")
        rec.update(crate_out=out, opened=out > 0.8 * args.pull, track_hook=max(e1, e2))
    if not args.no_pick:
        # ---- 2. pick the object from above and put it on the table ----
        def pose():
            return obj.data.root_pos_w[0].cpu().numpy().copy()

        CURQ = Q_PICK_FLIP if i in [int(b) for b in args.pick_flip_bins.split(',') if b] else Q_PICK  # fingers along world x for the pick and place

        p = pose()
        in_crate = float(p[2])
        print(f"   {name} at {np.round(p, 3)} (distance from the arm axis {np.hypot(p[0], p[1]):.2f} m)")
        move((p[0], p[1], TRANSIT_Z), 0.0, lin=0.012, hold=10)  # arrive overhead and let everything settle
        p = pose()  # the object may have slid on the crate floor while the crate stopped
        move((p[0], p[1], p[2] + 0.13), 0.0, hold=10)
        p = pose()
        # grasp at mid-height, but for tall objects no lower than ~4 cm under the top: the palm and camera sit ~4 cm above the TCP
        gz = max(0.01, h / 2 - 0.042)
        grasp = np.array([p[0], p[1], p[2] + gz])
        move(grasp + np.array([0, 0, 0.10]), 0.0, lin=0.004, hold=12)
        objinfo(obj, "before descending")
        move(grasp, 0.0, lin=0.004, hold=12)
        objinfo(obj, "after descending")
        pp = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy()
        print(f"   pads (left, right) xyz {np.round(pp, 3).tolist()}   object xyz {np.round(pose(), 3).tolist()}   commanded TCP {np.round(grasp, 3).tolist()}")
        allb = {n_: np.round(robot.data.body_pos_w[0, k_].cpu().numpy(), 3).tolist() for k_, n_ in enumerate(robot.body_names) if robot.data.body_pos_w[0, k_, 2] < 0.97 and "ur5e" not in n_ or "tool" in n_}
        print("   low robot bodies:", allb)
        e3 = track_err(grasp, "at the object")
        TRACE.update(on=True, obj=obj, n=0)
        move(grasp, GRIP_MAX, hold=args.close_hold)
        objinfo(obj, "after closing")
        grip_dump("gripper joints after closing on the object")
        pp = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy()
        print(f"   after closing: pads x {np.round(pp[:, 0], 3)} y {np.round(pp[:, 1], 3)} z {np.round(pp[:, 2], 3)}, object {np.round(pose(), 3)}, fingers {np.round(robot.data.joint_pos[0, robot.find_joints(list(E.GRIPPER_DRIVEN))[0]].cpu().numpy(), 2)}")
        move((p[0], p[1], TRANSIT_Z), GRIP_MAX, lin=args.lift_lin, smooth=True, hold=6)
        TRACE["on"] = False
        lifted = float(obj.data.root_pos_w[0, 2]) - in_crate
        pp = robot.data.body_pos_w[0, PAD_IDS].cpu().numpy()
        print(f"   after the lift: pads {np.round(pp, 3).tolist()}, object {np.round(pose(), 3)}")
        print(f"   lifted by {lifted:.3f} m  -> {'GRASPED' if lifted > 0.15 else 'NOT GRASPED'}")
        tz = E.TABLE_Z + h / 2 + 0.004 + gz  # TCP height when the object stands on the table
        move((*PLACE_XY[i], TRANSIT_Z), GRIP_MAX, lin=args.lift_lin, smooth=True, hold=6)
        move((*PLACE_XY[i], tz), GRIP_MAX, lin=min(args.lift_lin, 0.006), smooth=True, hold=6)
        e4 = track_err(np.array([*PLACE_XY[i], tz]), "at the place point")
        move((*PLACE_XY[i], tz), 0.0, hold=8)
        move((*PLACE_XY[i], TRANSIT_Z), 0.0, lin=0.012)
        q = obj.data.root_pos_w[0].cpu().numpy()
        quat = obj.data.root_quat_w[0].cpu().numpy()
        # the file's y axis was turned up; the object is upright if that axis still points up
        up_axis = Rotation.from_quat(quat[[1, 2, 3, 0]]).apply([0, 1, 0])[2]
        on_table = abs(q[2] - (E.TABLE_Z + h / 2)) < 0.02 and abs(q[0] - PLACE_XY[i][0]) < 0.05 and abs(q[1] - PLACE_XY[i][1]) < 0.05
        print(f"   placed at {np.round(q, 3)}, upright {up_axis:.2f}  -> {'PLACED' if on_table and up_axis > 0.9 else 'NOT PLACED'}")
        rec.update(grasped=lifted > 0.15, placed=bool(on_table and up_axis > 0.9), track_pick=max(e3, e4))
    results.append(rec)

print("\nSUMMARY")
for r in results:
    print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"sidebins{args.tag}.mp4"
imageio.mimwrite(path, frames, fps=10, macro_block_size=1)
print("video:", path, len(frames), "frames")
env.close()
app.close()
