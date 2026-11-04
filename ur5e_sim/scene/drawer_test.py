"""Feasibility test: can the UR5e + Robotiq open a drawer of the Sektion cabinet by pinching its handle bar?

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 python drawer_test.py --headless [--drawer top|bottom] [--pull 0.25]

World frame: z up, arm base on the table top at the origin (z = 0.75). The cabinet stands on the floor to the arm's right
(-y) with its front facing the arm. The gripper approaches the handle horizontally (tool z -> world -y) with the fingers
closing vertically (tool y -> world z), so they straddle the 24 mm tall bar, and then pulls toward the arm (+y).
Writes <out>/drawer_<drawer>.mp4 (external view + wrist view) and prints the drawer travel.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--drawer", default="top", choices=["top", "bottom"])
ap.add_argument("--pull", type=float, default=0.25, help="how far to pull the drawer [m]")
ap.add_argument("--cab-y", type=float, default=-0.62, help="world y of the cabinet's front face")
ap.add_argument("--cab-x", type=float, default=0.35, help="world x of the cabinet's centre line")
ap.add_argument("--grip-k", type=float, default=6.0, help="gripper drive stiffness (squeeze force)")
ap.add_argument("--follow-mult", type=float, default=1.0, help="multiplier on the follower (finger/knuckle) joint gains")
ap.add_argument("--friction", type=float, default=2.0, help="finger pad friction coefficient")
ap.add_argument("--lift", type=float, default=0.0, help="raise the cabinet on a plinth by this much [m]")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/drawer"))
ap.add_argument("--grip-gap", type=float, default=0.040, help="finger gap before closing on the bar [m]")
ap.add_argument("--grasp-y", type=float, default=0.0, help="extra TCP offset along the approach axis [m] (+ = toward the arm)")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import ur5e_env as E  # noqa: E402

ASSETS = Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac")
CABINET_USD = ASSETS / "Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
CAB_HALF_H = 0.384  # cabinet origin height above the floor
HANDLE_Z = {"top": 0.327, "bottom": 0.198}  # handle centre height in the cabinet frame
HANDLE_X = 0.3555  # handle bar centre, from the cabinet origin, along the drawer axis
FRONT_X = 0.325  # drawer face


@configclass
class DrawerSceneCfg(E.SceneCfg):
    cube = None
    table = AssetBaseCfg(  # narrow slab; the real table must leave room for an opened drawer
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.1, 0.5, E.TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, E.TABLE_Z / 2)),
    )
    cabinet: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CABINET_USD),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(args.cab_x, args.cab_y - FRONT_X, CAB_HALF_H + args.lift),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),  # +90 deg about z: the cabinet front (+x) faces +y, toward the arm
            joint_pos={"door_left_joint": 0.0, "door_right_joint": 0.0, "drawer_top_joint": 0.0, "drawer_bottom_joint": 0.0},
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"], stiffness=0.0, damping=8.0, effort_limit_sim=200.0
            ),
            "doors": ImplicitActuatorCfg(joint_names_expr=["door_left_joint", "door_right_joint"], stiffness=200.0, damping=20.0),
        },
    )
    robot: ArticulationCfg = E._robot_cfg()
    cam_side = E._camera(  # profile view of the handle from the cabinet's side (world +x), 0.32 m away
        "cam_side", (args.cab_x + 0.32, args.cab_y - FRONT_X + HANDLE_X + 0.02, HANDLE_Z[args.drawer] + CAB_HALF_H + args.lift + 0.02),
        20.0, rot=E.look_at_quat((args.cab_x + 0.32, args.cab_y - FRONT_X + HANDLE_X + 0.02, HANDLE_Z[args.drawer] + CAB_HALF_H + args.lift + 0.02),
                                 (args.cab_x, args.cab_y - FRONT_X + HANDLE_X + 0.02, HANDLE_Z[args.drawer] + CAB_HALF_H + args.lift)))
    cam_dbg = E._camera(
        "cam_dbg", (0.75, 0.15, 1.35), 14.0, rot=E.look_at_quat((0.75, 0.15, 1.35), (0.35, -0.5, 0.72))
    )


@configclass
class DrawerEnvCfg(E.Ur5eTableEnvCfg):
    scene: DrawerSceneCfg = DrawerSceneCfg(num_envs=1, env_spacing=8.0)


def tool_quat() -> np.ndarray:
    """(w, x, y, z): tool z -> world -y (approach); the fingers close along tool x (measured), so tool x -> world z."""
    cols = np.stack([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], axis=1)
    x, y, z, w = Rotation.from_matrix(cols).as_quat()
    return np.array([w, x, y, z])


_cfg = DrawerEnvCfg()
_cfg.scene.robot.actuators["gripper"].stiffness = args.grip_k
_cfg.scene.robot.actuators["gripper"].damping = 0.05 * args.grip_k
for _a in ("gripper_fingers", "gripper_knuckles"):
    _cfg.scene.robot.actuators[_a].stiffness *= args.follow_mult
    _cfg.scene.robot.actuators[_a].damping *= args.follow_mult
env = ManagerBasedRLEnv(_cfg)
env.reset()
# grippy finger pads: with combine mode "max" the pad friction wins over the cabinet's
_pad = sim_utils.RigidBodyMaterialCfg(static_friction=args.friction, dynamic_friction=args.friction, friction_combine_mode="max")
_pad.func("/World/Materials/pad", _pad)
for _link in env.scene["robot"].body_names:
    if "finger" in _link or "pad" in _link:
        sim_utils.bind_physics_material(f"/World/envs/env_0/Robot/{_link}", "/World/Materials/pad")
dev = env.device
cab = env.scene["cabinet"]
jn = cab.joint_names
drawer_j = jn.index(f"drawer_{args.drawer}_joint")
robot = env.scene["robot"]
frames = []
state = {"pos": np.array(E.READY_TCP), "quat": np.array(E.TOOL_DOWN_QUAT), "grip": 0.0}


def gap_to_angle(gap):  # Robotiq 2F-85: 0 rad = 85 mm open, 0.8 rad = closed
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, GRIP_MAX))


GRIP_MAX = E.GRIPPER_CLOSED


def step(pos, quat, grip):
    act = torch.tensor([[*pos, *quat, grip]], dtype=torch.float32, device=dev)
    env.step(act)
    rgb = [env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for c in ("cam_dbg", "cam_side", "cam_wrist")]
    frames.append(np.concatenate(rgb, 1))


def tcp_world():
    from isaaclab.utils.math import combine_frame_transforms  # noqa: F401
    idx = robot.find_bodies("ur5e_tool0")[0][0]
    return robot.data.body_pos_w[0, idx].cpu().numpy()


def move(pos, quat, grip, lin=0.008, ang=0.05, hold=3):
    """Interpolate the commanded TCP pose (arm-base frame) from the previous command."""
    p0, q0, g0 = state["pos"], state["quat"], state["grip"]
    pos, quat = np.asarray(pos, float), np.asarray(quat, float)
    r0 = Rotation.from_quat(q0[[1, 2, 3, 0]])
    r1 = Rotation.from_quat(quat[[1, 2, 3, 0]])
    ang_d = (r1 * r0.inv()).magnitude()
    n = max(2, int(np.ceil(max(np.linalg.norm(pos - p0) / lin, ang_d / ang, abs(grip - g0) / 0.05))))
    sl = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    for i in range(1, n + 1):
        t = i / n
        qx, qy, qz, qw = sl([t]).as_quat()[0]
        step(p0 + t * (pos - p0), np.array([qw, qx, qy, qz]), g0 + t * (grip - g0))
    for _ in range(hold):
        step(pos, quat, grip)
    state.update(pos=pos, quat=quat, grip=grip)


def drawer_pos():
    return float(cab.data.joint_pos[0, drawer_j])


Q = tool_quat()
# handle centre in the arm-base frame
hx = args.cab_x
hy_outer = args.cab_y - FRONT_X + HANDLE_X  # bar centre along world y
hz = HANDLE_Z[args.drawer] + CAB_HALF_H + args.lift - E.TABLE_Z
print(f"handle (arm-base frame): x={hx:.3f} y={hy_outer:.3f} z={hz:.3f}  distance from the base axis {np.hypot(hx, hy_outer):.3f} m")

# 1. leave the ready pose: swing out in front of the handle with the tool horizontal
move((0.30, -0.25, 0.10), Q, 0.0)
# 2. open the fingers to a gap just wider than the bar and stand off in front of it (toward the arm)
g_open = gap_to_angle(args.grip_gap)
move((hx, hy_outer + 0.12, hz), Q, g_open)
print("pre-grasp reached, TCP world:", tcp_world().round(3))
# 3. slide onto the bar
move((hx, hy_outer + args.grasp_y, hz), Q, g_open, lin=0.004)
tcp_err = tcp_world() - np.array([hx, hy_outer, hz + E.TABLE_Z])
print("on the bar, TCP error vs handle [m]:", tcp_err.round(4))
_i = robot.find_bodies("ur5e_tool0")[0][0]
_qw = robot.data.body_quat_w[0, _i].cpu().numpy()
print("tool orientation error [deg]:", np.degrees((Rotation.from_quat(_qw[[1, 2, 3, 0]]) * Rotation.from_quat(Q[[1, 2, 3, 0]]).inv()).magnitude()).round(1))
_ar = robot.find_joints(list(E.n.ARM_JOINTS), preserve_order=True)[0]
print("arm joints [deg]:", np.degrees(robot.data.joint_pos[0, _ar].cpu().numpy()).round(0))
def report(tag):
    names = [b for b in robot.body_names if "finger" in b or "pad" in b or "knuckle" in b]
    ids = robot.find_bodies(names)[0]
    print(tag)
    for nme, i in zip(names, ids):
        print(f"   {nme:28s} world {robot.data.body_pos_w[0, i].cpu().numpy().round(3)}")
    hb = cab.body_names.index(f"drawer_handle_{args.drawer}")
    print("   handle body world", cab.data.body_pos_w[0, hb].cpu().numpy().round(3))


report("before closing")
# 4. close on the bar
move((hx, hy_outer + args.grasp_y, hz), Q, GRIP_MAX, hold=10)
report("after closing")
fj = robot.find_joints(list(E.GRIPPER_DRIVEN))[0]
print("finger angles after closing (0.75 = closed on nothing):", robot.data.joint_pos[0, fj].cpu().numpy().round(3))
# 5. pull toward the arm
y_start = hy_outer + args.grasp_y
for k, d in enumerate(np.linspace(0, args.pull, 11)[1:]):
    move((hx, y_start + d, hz), Q, GRIP_MAX, lin=0.01, hold=2)
    _pl = robot.data.body_pos_w[0, robot.find_bodies(["left_inner_finger_pad", "right_inner_finger_pad"])[0]].cpu().numpy()
    _hb = cab.data.body_pos_w[0, cab.body_names.index(f"drawer_handle_{args.drawer}")].cpu().numpy()
    print(f"  commanded pull {d:.2f} m  -> drawer at {drawer_pos():.3f} m   TCP world {tcp_world().round(3)}   fingers {robot.data.joint_pos[0, fj].cpu().numpy().round(2)}"
          f"  pad-handle (y,z) L {(_pl[0, 1:] - _hb[1:]).round(3)} R {(_pl[1, 1:] - _hb[1:]).round(3)}")
# 6. release and back off
move((hx, y_start + args.pull, hz), Q, 0.0, hold=5)
move((hx, y_start + args.pull + 0.12, hz), Q, 0.0)
final = drawer_pos()
print(f"RESULT drawer {args.drawer}: opened {final:.3f} m (target {args.pull:.2f} m) -> {'OK' if final > 0.8 * args.pull else 'FAILED'}")
print("other joints:", dict(zip(jn, cab.data.joint_pos[0].cpu().numpy().round(3))))

args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"drawer_{args.drawer}_lift{int(args.lift * 100)}.mp4"
imageio.mimwrite(path, frames, fps=10, macro_block_size=1)
print("video:", path, len(frames), "frames")
env.close()
app.close()
