"""Can the UR5e + Robotiq pull the crate out of the housing by hooking the end wall's rim bar?

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 python pullbin_test.py --headless [--pull 0.25]

Needs ../assets/usd/pullbin.usd (build_pullbin.py). World frame: z up, arm base on the table top at the origin (z = 0.75).
The housing stands on the floor on the arm's right (-y) with its front facing the arm. Strategy: tool pointing down, fingers
closing along the pull axis (y); one fingertip goes down INSIDE the crate behind the end wall, the other outside, in front of the
rim bar; then the arm moves toward +y. The inner finger pushes on the wall, so no friction is needed to pull.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--pull", type=float, default=0.25, help="how far to pull the crate [m]")
ap.add_argument("--front-y", type=float, default=-0.62, help="world y of the housing's front edge")
ap.add_argument("--x", type=float, default=0.35, help="world x of the housing's centre line")
ap.add_argument("--grip-gap", type=float, default=0.040, help="finger gap while straddling the wall [m]")
ap.add_argument("--dz", type=float, default=0.0, help="TCP height offset from the middle of the rim bar [m]")
ap.add_argument("--dy", type=float, default=0.0, help="TCP offset along the pull axis from the middle of the wall [m]")
ap.add_argument("--grip-k", type=float, default=6.0)
ap.add_argument("--no-squeeze", action="store_true", help="do not close the fingers on the wall: pull with the open fingers (pure hook)")
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/scene_previews/pullbin"))
ap.add_argument("--tag", default="")
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

BIN_USD = HERE.parent / "assets" / "usd" / "pullbin.usd"
Z_B, CRATE_Z0, CH = 0.60, 0.002, 0.1703  # must match build_pullbin.py
WALL_IN_X, BAR_OUT_X, BAR_Z0 = 0.290, 0.3026, 0.144
FRONT_X = 0.32  # housing front edge in the housing frame
ORIGIN_Y = args.front_y - FRONT_X  # housing frame origin, world y (the housing is yawed +90 deg: local +x -> world +y)
WALL_Y = ORIGIN_Y + 0.5 * (WALL_IN_X + BAR_OUT_X)  # middle of the end wall incl. the rim bar, world y
BAR_Z = Z_B + CRATE_Z0 + 0.5 * (BAR_Z0 + CH)  # middle of the rim bar, world z


EYE_D = (0.95, 0.25, 1.45)  # overview camera
EYE_S = (args.x + 0.55, WALL_Y + 0.12, BAR_Z + 0.02)  # profile camera on the end wall


@configclass
class BinSceneCfg(E.SceneCfg):
    cube = None
    robot: ArticulationCfg = E._robot_cfg()
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.1, 0.5, E.TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, E.TABLE_Z / 2)),
    )
    bin: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/bin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BIN_USD),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(args.x, ORIGIN_Y, 0.0), rot=(0.70710678, 0.0, 0.0, 0.70710678), joint_pos={"crate_slide": 0.0}
        ),
        actuators={"slide": ImplicitActuatorCfg(joint_names_expr=["crate_slide"], stiffness=0.0, damping=10.0, effort_limit_sim=500.0)},
    )
    cam_dbg = E._camera("cam_dbg", EYE_D, 14.0, rot=E.look_at_quat(EYE_D, (0.35, -0.55, 0.72)))
    cam_side = E._camera("cam_side", EYE_S, 20.0, rot=E.look_at_quat(EYE_S, (args.x, WALL_Y, BAR_Z - 0.02)))


@configclass
class BinEnvCfg(E.Ur5eTableEnvCfg):
    scene: BinSceneCfg = BinSceneCfg(num_envs=1, env_spacing=8.0)


def tool_quat() -> np.ndarray:
    """(w, x, y, z): tool z -> world -z (pointing down), tool x (the finger closing axis) -> world y (the pull axis)."""
    cols = np.stack([[0, 1, 0], [1, 0, 0], [0, 0, -1]], axis=1)
    x, y, z, w = Rotation.from_matrix(cols).as_quat()
    return np.array([w, x, y, z])


cfg = BinEnvCfg()
cfg.scene.robot.actuators["gripper"].stiffness = args.grip_k
cfg.scene.robot.actuators["gripper"].damping = 0.05 * args.grip_k
env = ManagerBasedRLEnv(cfg)
env.reset()
dev = env.device
robot, bin_ = env.scene["robot"], env.scene["bin"]
slide_j = bin_.joint_names.index("crate_slide")
frames = []
state = {"pos": np.array(E.READY_TCP), "quat": np.array(E.TOOL_DOWN_QUAT), "grip": 0.0}
GRIP_MAX = E.GRIPPER_CLOSED


def gap_to_angle(gap):  # Robotiq 2F-85: 0 rad = 85 mm open, 0.8 rad = closed
    return float(np.clip(0.8 * (1.0 - gap / 0.085), 0.0, GRIP_MAX))


def step(pos, quat, grip):
    env.step(torch.tensor([[*pos, *quat, grip]], dtype=torch.float32, device=dev))
    rgb = [env.scene[c].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8) for c in ("cam_dbg", "cam_side", "cam_wrist")]
    frames.append(np.concatenate(rgb, 1))


def move(pos, quat, grip, lin=0.008, ang=0.05, hold=3):
    """Interpolate the commanded TCP pose (arm-base frame) from the previous command."""
    p0, q0, g0 = state["pos"], state["quat"], state["grip"]
    pos, quat = np.asarray(pos, float), np.asarray(quat, float)
    r0, r1 = Rotation.from_quat(q0[[1, 2, 3, 0]]), Rotation.from_quat(quat[[1, 2, 3, 0]])
    n = max(2, int(np.ceil(max(np.linalg.norm(pos - p0) / lin, (r1 * r0.inv()).magnitude() / ang, abs(grip - g0) / 0.05))))
    sl = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    for i in range(1, n + 1):
        t = i / n
        qx, qy, qz, qw = sl([t]).as_quat()[0]
        step(p0 + t * (pos - p0), np.array([qw, qx, qy, qz]), g0 + t * (grip - g0))
    for _ in range(hold):
        step(pos, quat, grip)
    state.update(pos=pos, quat=quat, grip=grip)


def crate_out() -> float:
    return float(bin_.data.joint_pos[0, slide_j])


def pad_report(tag):
    ids = robot.find_bodies(["left_inner_finger_pad", "right_inner_finger_pad"])[0]
    p = robot.data.body_pos_w[0, ids].cpu().numpy()
    print(f"{tag}: pads world y = {p[:, 1].round(3)}  z = {p[:, 2].round(3)}   wall y = {WALL_Y:.3f} (+-{0.5 * (BAR_OUT_X - WALL_IN_X):.4f}) bar z = {BAR_Z:.3f}")


Q = tool_quat()
hx, hy = args.x, WALL_Y + args.dy
hz = BAR_Z + args.dz - E.TABLE_Z  # arm-base frame
print(f"rim bar centre (arm-base frame): x={hx:.3f} y={hy:.3f} z={hz:.3f}   distance from the base axis {np.hypot(hx, hy):.3f} m")
g_open = gap_to_angle(args.grip_gap)
# 1. go above the wall with the tool pointing down and the fingers open wider than the wall
move((hx, hy, hz + 0.14), Q, g_open, lin=0.01)
pad_report("above the wall")
# 2. lower the fingers over the wall: one ends up inside the crate, the other in front of the rim bar
move((hx, hy, hz), Q, g_open, lin=0.004)
pad_report("fingers straddle the wall")
# 3. squeeze (or not) and pull toward the arm
grip_pull = g_open if args.no_squeeze else GRIP_MAX
move((hx, hy, hz), Q, grip_pull, hold=10)
for d in np.linspace(0, args.pull, 11)[1:]:
    move((hx, hy + d, hz), Q, grip_pull, lin=0.01, hold=2)
    print(f"  commanded pull {d:.2f} m -> crate out {crate_out():.3f} m")
# 4. release and lift off
move((hx, hy + args.pull, hz), Q, g_open, hold=5)
move((hx, hy + args.pull, hz + 0.15), Q, g_open)
final = crate_out()
print(f"RESULT: crate pulled out {final:.3f} m (target {args.pull:.2f} m) -> {'OK' if abs(final - args.pull) < 0.04 else 'FAILED'}")

args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"pullbin{args.tag}.mp4"
imageio.mimwrite(path, frames, fps=10, macro_block_size=1)
print("video:", path, len(frames), "frames")
env.close()
app.close()
