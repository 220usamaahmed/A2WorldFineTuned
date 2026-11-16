"""Where files live. Large files (downloaded assets, generated objects, episodes) go under A2W_DATA
(environment variable; default: <repo>/data). Everything else is relative to this repository."""
import os
from pathlib import Path

SIM = Path(__file__).resolve().parent  # ur5e_sim/
DATA = Path(os.environ.get("A2W_DATA", SIM.parent / "data")).expanduser()

# in the repository
ROBOT_URDF = SIM / "assets" / "ur5e_gripper.urdf"
ROBOT_URDF_SIMPLE = SIM / "assets" / "ur5e_gripper_simple.urdf"
SIMPLE_GRIPPER_INFO = SIM / "assets" / "simple_gripper.json"
USD = SIM / "assets" / "usd"  # generated: robot (setup/convert_arm.py) and drawer (setup/build_pullbin.py)
ROBOT_YML = SIM / "planning" / "ur5e_2f85.yml"  # cuRobo robot config (setup/make_robot_cfg.py)

# under A2W_DATA
ASSETS = DATA / "assets"
ISAAC = ASSETS / "Isaac/5.1/Isaac"  # mirror of NVIDIA's Isaac assets (setup/fetch_assets.py)
NVIDIA = ASSETS / "Isaac/5.1/NVIDIA"  # materials and light map (scene/looks.py)
YCB_PHYSICS = ASSETS / "ycb_physics"  # setup/make_ycb_physics.py
YCB_MESHES = ASSETS / "ycb_meshes"  # setup/export_ycb_meshes.py
GRASP_CACHE = ASSETS / "grasp_cache"  # filled on first use by planning/grasps.py
EPISODES = DATA / "episodes"  # default output of scene/pick_place.py
