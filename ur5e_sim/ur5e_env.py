"""Base Isaac Lab environment: the UR5e + Robotiq 2F-85 on a table, a wrist camera, joint-space control.
scene/pick_place.py extends the scene (furniture, objects, cameras).

Frames: world z up; the arm base sits on the table top at the world origin facing +x. All robot poses
reported by this module are in the arm-base frame (= world frame translated by TABLE_Z in z).

Action (dim 7): [6 arm joint positions to reach at the end of the control step, in robot_defs.ARM_JOINTS order |
finger angle [rad], 0 open ... 0.8 closed]. Control 10 Hz (physics 120 Hz); cameras render once per control step.

Import only after the Isaac Sim app has been launched.
"""
import sys
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import joint_pos, reset_scene_to_default, time_out
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # sibling modules import regardless of the working directory
from gripper_action import MimicGripperPositionActionCfg, SimpleGripperActionCfg  # noqa: E402  (gripper controllers, see their headers)
from joint_traj_action import JointTrajectoryActionCfg  # noqa: E402
import paths  # noqa: E402
import robot_defs as n  # noqa: E402

USD_PATH = paths.USD / "ur5e_gripper.usd"
USD_PATH_SIMPLE = paths.USD / "ur5e_gripper_simple.usd"  # make_simple_gripper.py + convert_arm.py --simple

# ---- geometry [m] ---------------------------------------------------------------------------------
TABLE_Z = 0.75  # table top height; the arm base is mounted on the table top at the world origin
TABLE_X = (-0.15, 0.95)  # table extent along x
TABLE_HALF_Y = 0.35  # table edge at y = +-0.35
CONTROL_DT = 0.1  # 10 Hz
PHYSICS_DT = 1.0 / 120.0
IMG = 256

GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.75
GRIPPER_DRIVEN = (n.GRIPPER_DRIVEN_JOINT, "right_outer_knuckle_joint")
GRIPPER_FOLLOWERS = {
    "left_inner_finger_joint": (n.GRIPPER_DRIVEN_JOINT, n.GRIPPER_MIMIC_JOINTS["left_inner_finger_joint"]),
    "left_inner_knuckle_joint": (n.GRIPPER_DRIVEN_JOINT, n.GRIPPER_MIMIC_JOINTS["left_inner_knuckle_joint"]),
    "right_inner_finger_joint": ("right_outer_knuckle_joint", n.GRIPPER_MIMIC_JOINTS["right_inner_finger_joint"]),
    "right_inner_knuckle_joint": ("right_outer_knuckle_joint", n.GRIPPER_MIMIC_JOINTS["right_inner_knuckle_joint"]),
}
# ARM_READY (from the Steve project): tool pointing straight down, TCP 0.395 m in front of the arm base, 0.034 m above its plane
READY_Q = n.ARM_READY
READY_TCP = (0.395, 0.0, 0.034)
TOOL_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)  # (w, x, y, z): 180 deg about x

# ---- wrist camera ------------------------------------------------------------------------------------
WRIST_OFFSET = (0.095, 0.0, 0.03)  # in the ur5e_tool0 frame
WRIST_TARGET = (0.0, 0.0, 0.30)  # point the wrist camera looks at, in the ur5e_tool0 frame
WRIST_FOCAL_MM = 12.0  # ~82 deg


# ---- cameras -------------------------------------------------------------------------------------------
def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """(w, x, y, z) for Isaac Lab's 'world' camera convention (forward +x, left +y, up +z)."""
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross(np.asarray(up, float), f)
    left /= np.linalg.norm(left)
    u = np.cross(f, left)
    x, y, z, w = Rotation.from_matrix(np.stack([f, left, u], axis=1)).as_quat()
    return (float(w), float(x), float(y), float(z))


def _camera(name: str, eye, focal: float, parent: str = "{ENV_REGEX_NS}", **kw) -> TiledCameraCfg:
    return TiledCameraCfg(
        prim_path=f"{parent}/{name}",
        update_period=CONTROL_DT,
        height=IMG,
        width=IMG,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.03, 8.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(eye), rot=kw["rot"], convention="world"),
    )


# ---- robot: UR5e + Robotiq 2F-85 from assets/usd, PD joint drives ---------------------------------------
def _robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            # gravity off for the robot's own links (as in the Steve project): the arm then holds its pose exactly, as a real UR5e
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_Z),
            joint_pos={
                **dict(zip(n.ARM_JOINTS, READY_Q)),
                n.GRIPPER_DRIVEN_JOINT: 0.0,
                **{j: 0.0 for j in n.GRIPPER_MIMIC_JOINTS},
            },
        ),
        actuators={
            "arm_big": ImplicitActuatorCfg(joint_names_expr=list(n.ARM_JOINTS[:3]), stiffness=4000.0, damping=400.0),
            "arm_wrist": ImplicitActuatorCfg(joint_names_expr=list(n.ARM_JOINTS[3:]), stiffness=1000.0, damping=100.0),
            # gripper gains as tuned in the Steve project (squeeze force ~ gain x remaining error)
            "gripper": ImplicitActuatorCfg(joint_names_expr=list(GRIPPER_DRIVEN), stiffness=6.0, damping=0.3),
            "gripper_fingers": ImplicitActuatorCfg(
                joint_names_expr=[j for j in GRIPPER_FOLLOWERS if "finger" in j], stiffness=300.0, damping=15.0
            ),
            "gripper_knuckles": ImplicitActuatorCfg(
                joint_names_expr=[j for j in GRIPPER_FOLLOWERS if "knuckle" in j], stiffness=20.0, damping=1.0
            ),
        },
    )


# ---- base scene (scene/pick_place.py extends it) ----------------------------------------------------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.95, 1.0)))
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_X[1] - TABLE_X[0], 2 * TABLE_HALF_Y, TABLE_Z),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.18), roughness=0.8),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5 * (TABLE_X[0] + TABLE_X[1]), 0.0, TABLE_Z / 2)),
    )
    robot: ArticulationCfg = _robot_cfg()
    # on the tool flange; tool +x is "up" in the image, so the fingers (which close along tool y) appear at the bottom
    cam_wrist = _camera(
        "cam_wrist", WRIST_OFFSET, WRIST_FOCAL_MM, parent="{ENV_REGEX_NS}/Robot/ur5e_tool0",
        rot=look_at_quat(WRIST_OFFSET, WRIST_TARGET, up=(1.0, 0.0, 0.0)),
    )


# ---- Isaac Lab managers: actions (the controller), observations, events, rewards, terminations ------------
@configclass
class ActionsCfg:
    """Arm: planned joint trajectories (planning/); gripper: linkage model (use_simple_gripper replaces it)."""

    arm = JointTrajectoryActionCfg(asset_name="robot", joint_names=list(n.ARM_JOINTS))
    gripper = MimicGripperPositionActionCfg(
        asset_name="robot", driven_joints=list(GRIPPER_DRIVEN), followers=GRIPPER_FOLLOWERS
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(
            func=joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(n.ARM_JOINTS), preserve_order=True)},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)


# ---- the environment config: scene + managers + timing --------------------------------------------------
@configclass
class Ur5eEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.sim.dt = PHYSICS_DT
        self.decimation = int(round(CONTROL_DT / PHYSICS_DT))  # 12
        self.sim.render_interval = self.decimation
        self.sim.log_dir = str(HERE / "logs")  # /tmp/isaaclab belongs to another user on this machine
        self.episode_length_s = 600.0  # scripted episodes must not be cut by the time-out
        self.viewer.eye = (1.6, -1.2, TABLE_Z + 0.9)
        self.viewer.lookat = (0.4, 0.0, TABLE_Z + 0.1)


# ---- simplified gripper physics (stage 2) ---------------------------------------------------------------
SIMPLE_LEFT = {n.GRIPPER_DRIVEN_JOINT: 1.0, "left_inner_knuckle_joint": 1.0, "left_inner_finger_joint": -1.0}
SIMPLE_RIGHT = {"right_outer_knuckle_joint": 1.0, "right_inner_knuckle_joint": 1.0, "right_inner_finger_joint": -1.0}


def use_simple_gripper(cfg, grip_force: float = 20.0, pad_speed: float = 0.10):
    """Switch an env cfg to the gripper with simplified physics: two sliding pads squeezing with
    ``grip_force`` [N] each (the real 2F-85: 20-235 N; in PhysX, 20-30 N hold a 5 cm cube with <= 1.5 mm penetration,
    40 N pushes the pads through it), closing at up to ``pad_speed`` [m/s] (real: 20-150 mm/s); the
    Robotiq linkage is visual only and follows the pads (gripper_action.SimpleGripperAction). Same action meaning."""
    r = cfg.scene.robot
    r.spawn.usd_path = str(USD_PATH_SIMPLE)
    r.init_state.joint_pos.update({"left_pad_slide": 0.0, "right_pad_slide": 0.0})
    r.actuators = {
        "arm_big": r.actuators["arm_big"],
        "arm_wrist": r.actuators["arm_wrist"],
        # stiff enough that 5 mm short of the target already gives the full force; the force limit sets the squeeze
        "pads": ImplicitActuatorCfg(joint_names_expr=["left_pad_slide", "right_pad_slide"], stiffness=grip_force / 0.005,
                                    damping=30.0, effort_limit_sim=grip_force, velocity_limit_sim=pad_speed),
        "linkage": ImplicitActuatorCfg(joint_names_expr=list(SIMPLE_LEFT) + list(SIMPLE_RIGHT), stiffness=100.0, damping=5.0),
    }
    cfg.actions.gripper = SimpleGripperActionCfg(asset_name="robot", info_path=str(paths.SIMPLE_GRIPPER_INFO),
                                                 left_linkage=SIMPLE_LEFT, right_linkage=SIMPLE_RIGHT, coupling=0.005)
    return cfg
