"""UR5e + Robotiq 2F-85 on a table (Isaac Lab): one cube, three RGB cameras, TCP-pose control.

Frames: world z up; the arm base sits on the table top at the world origin facing +x. All robot poses
reported by this module are in the arm-base frame (= world frame translated by TABLE_Z in z).

Action (dim 8): [TCP x y z qw qx qy qz in the arm-base frame | finger angle [rad], 0 open ... 0.8 closed]
Control 10 Hz (physics 120 Hz); cameras are rendered once per control step.

Import only after the Isaac Sim app has been launched.
"""
import sys
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    DifferentialInverseKinematicsActionCfg,
    joint_pos,
    reset_scene_to_default,
    time_out,
)
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
from gripper_action import MimicGripperPositionActionCfg  # noqa: E402  (tuned gripper controller, see its header)
from joint_traj_action import JointTrajectoryActionCfg  # noqa: E402
import robot_defs as n  # noqa: E402

USD_PATH = HERE / "assets" / "usd" / "ur5e_gripper.usd"

# ---- geometry [m] ---------------------------------------------------------------------------------
TABLE_Z = 0.75  # table top height; the arm base is mounted on the table top at the world origin
TABLE_X = (-0.15, 0.95)  # table extent along x
TABLE_HALF_Y = 0.35  # table edge at y = +-0.35
CUBE_SIZE = 0.05
CUBE_Z = CUBE_SIZE / 2  # cube centre height above the table (arm-base frame)
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

# ---- camera poses (world frame, look-at) ------------------------------------------------------------
LOOK_AT = (0.42, 0.0, TABLE_Z)
CAMERAS = {
    "ext1": dict(eye=(1.15, 0.62, TABLE_Z + 0.55)),
    "ext2": dict(eye=(1.15, -0.62, TABLE_Z + 0.55)),
}
CAM_FOCAL_MM = 15.0  # with the default 20.955 mm aperture: ~70 deg horizontal field of view
WRIST_OFFSET = (0.095, 0.0, 0.03)  # in the ur5e_tool0 frame
WRIST_TARGET = (0.0, 0.0, 0.30)  # point the wrist camera looks at, in the ur5e_tool0 frame
WRIST_FOCAL_MM = 12.0  # ~82 deg


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
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.5, restitution=0.0, friction_combine_mode="max"
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.2, TABLE_Z + CUBE_Z)),
    )
    cam_ext1 = _camera("cam_ext1", CAMERAS["ext1"]["eye"], CAM_FOCAL_MM, rot=look_at_quat(CAMERAS["ext1"]["eye"], LOOK_AT))
    cam_ext2 = _camera("cam_ext2", CAMERAS["ext2"]["eye"], CAM_FOCAL_MM, rot=look_at_quat(CAMERAS["ext2"]["eye"], LOOK_AT))
    # on the tool flange; tool +x is "up" in the image, so the fingers (which close along tool y) appear at the bottom
    cam_wrist = _camera(
        "cam_wrist", WRIST_OFFSET, WRIST_FOCAL_MM, parent="{ENV_REGEX_NS}/Robot/ur5e_tool0",
        rot=look_at_quat(WRIST_OFFSET, WRIST_TARGET, up=(1.0, 0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(n.ARM_JOINTS),
        body_name=n.EE_LINK,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, n.TCP_OFFSET_Z)),
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    )
    gripper = MimicGripperPositionActionCfg(
        asset_name="robot", driven_joints=list(GRIPPER_DRIVEN), followers=GRIPPER_FOLLOWERS
    )


@configclass
class JointActionsCfg:
    """Arm commanded in joint space (planned trajectories, see planning/), gripper as above. Action (dim 7):
    [6 arm joint positions to reach at the end of the control step, in robot_defs.ARM_JOINTS order | finger angle]."""

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


@configclass
class Ur5eTableEnvCfg(ManagerBasedRLEnvCfg):
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


@configclass
class Ur5eTableJointEnvCfg(Ur5eTableEnvCfg):
    """Same scene, arm driven by joint-space targets (JointActionsCfg) instead of the differential-IK TCP pose."""

    actions: JointActionsCfg = JointActionsCfg()
