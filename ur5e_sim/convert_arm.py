"""URDF -> USD for the UR5e + Robotiq arm (fixed base). Importer settings as in the Steve project's convert_urdf.py.

    OMNI_KIT_ACCEPT_EULA=YES python convert_arm.py --headless            # ur5e_gripper.usd (four-bar gripper physics)
    OMNI_KIT_ACCEPT_EULA=YES python convert_arm.py --headless --simple   # ur5e_gripper_simple.usd (make_simple_gripper.py)
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--simple", action="store_true", help="convert assets/ur5e_gripper_simple.urdf (sliding physics pads)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd  # noqa: E402

HERE = Path(__file__).resolve().parent
NAME = "ur5e_gripper_simple" if args.simple else "ur5e_gripper"
URDF = HERE / "assets" / f"{NAME}.urdf"
USD_DIR = HERE / "assets" / "usd"


def tune_mimic_joints(usd_path: str, freq: float = 200.0, damping_ratio: float = 1.0) -> int:
    stage = Usd.Stage.Open(usd_path)
    n = 0
    for prim in stage.Traverse():
        for schema in prim.GetAppliedSchemas():
            if schema.startswith("PhysxMimicJointAPI:"):
                axis = schema.split(":")[1]
                prim.GetAttribute(f"physxMimicJoint:{axis}:naturalFrequency").Set(freq)
                prim.GetAttribute(f"physxMimicJoint:{axis}:dampingRatio").Set(damping_ratio)
                n += 1
    stage.GetRootLayer().Save()
    return n


cfg = UrdfConverterCfg(
    asset_path=str(URDF),
    usd_dir=str(USD_DIR),
    usd_file_name=f"{NAME}.usd",
    fix_base=True,
    merge_fixed_joints=False,  # keep ur5e_tool0 as a body
    convert_mimic_joints_to_normal_joints=True,  # (sic) = import <mimic> as PhysX mimic joints
    self_collision=False,
    collider_type="convex_hull",
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    ),
)
conv = UrdfConverter(cfg)
print("[convert_arm] mimic joints tuned:", tune_mimic_joints(conv.usd_path), "->", conv.usd_path, flush=True)

stage = Usd.Stage.Open(conv.usd_path)
print("[convert_arm] default prim:", stage.GetDefaultPrim().GetPath())
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if prim.GetTypeName() in ("Xform", "PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsPrismaticJoint") and p.count("/") <= 3:
        print("  ", prim.GetTypeName(), p)
simulation_app.close()
