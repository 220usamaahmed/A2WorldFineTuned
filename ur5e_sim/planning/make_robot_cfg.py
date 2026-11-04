"""Generate the cuRobo robot config for our UR5e + Robotiq 2F-85 (planning/ur5e_2f85.yml) and check it against the sim.

    python make_robot_cfg.py            # (env_isaaclab, no Isaac app needed)

* Kinematics: our own URDF (assets/ur5e_gripper.urdf, the one the Isaac asset is converted from), base link
  ``ur5e_base_link`` (= the articulation root, i.e. the "arm-base frame" used everywhere in ur5e_sim), end-effector
  ``grasp_frame`` = ``ur5e_tool0`` shifted by robot_defs.TCP_OFFSET_Z along tool z: the same TCP the IK action used.
* Arm collision spheres: cuRobo's UR5e spheres (content/configs/robot/ur5e_robotiq_2f_140.yml). Our URDF has the same arm
  joint origins as cuRobo's UR5e (checked when writing this script), so the per-link sphere centres carry over with the
  ``ur5e_`` link prefix.
* Gripper collision spheres: a conservative cover of the 2F-85 collision meshes / pad boxes (sphere_cover below).
* ``finger_joint`` is locked open (0): planning with open fingers is the conservative case for approach motions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
import yourdfpy

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
import robot_defs as n  # noqa: E402

from curobo.util_file import get_robot_configs_path, join_path, load_yaml  # noqa: E402

URDF = SIM / "assets" / "ur5e_gripper.urdf"
OUT = HERE / "ur5e_2f85.yml"
P = "ur5e_"  # arm link prefix in our URDF
ARM_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
GRIPPER_LINKS = [
    "robotiq_arg2f_base_link",
    "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger", "right_outer_finger",
    "left_inner_knuckle", "right_inner_knuckle", "left_inner_finger", "right_inner_finger",
    "left_inner_finger_pad", "right_inner_finger_pad",
]


def link_collision_mesh(robot: yourdfpy.URDF, link: str) -> trimesh.Trimesh:
    """All collision geometry of one link, in the link frame."""
    parts = []
    for c in robot.link_map[link].collisions:
        g = c.geometry
        if g.mesh is not None:
            m = trimesh.load(str(URDF.parent / g.mesh.filename), force="mesh")
            if g.mesh.scale is not None:
                m.apply_scale(np.broadcast_to(np.asarray(g.mesh.scale, float), 3)[0])
        elif g.box is not None:
            m = trimesh.creation.box(extents=g.box.size)
        else:
            raise ValueError(f"{link}: unsupported collision geometry")
        if c.origin is not None:
            m.apply_transform(c.origin)
        parts.append(m)
    return trimesh.util.concatenate(parts)


def sphere_cover(mesh: trimesh.Trimesh, cell_max: float, tol: float = 0.001, n_pts: int = 8000):
    """Spheres that cover the whole surface of ``mesh`` (conservative collision model).

    cuRobo's own fitter (fit_spheres_to_mesh) covered under 25 % of these links' surfaces (95th-percentile gaps of
    1-3 cm), so instead: grid centres over the oriented bounding box (cells at most ``cell_max``), moved by a few Lloyd
    iterations to the centroid of the surface points nearest to them; each sphere's radius reaches its farthest assigned
    point. Every sampled surface point is inside a sphere by construction."""
    ob = mesh.bounding_box_oriented
    T, e = ob.primitive.transform, ob.primitive.extents
    nn = np.maximum(1, np.ceil(e / min(max(e.min(), 0.004), cell_max))).astype(int)
    g = [(np.arange(k) + 0.5) / k * ei - ei / 2 for k, ei in zip(nn, e)]
    c = trimesh.transform_points(np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3), T)
    pts, _ = trimesh.sample.sample_surface_even(mesh, n_pts, seed=1)
    for _ in range(5):
        a = np.argmin(np.linalg.norm(pts[:, None] - c[None], axis=2), 1)
        c = np.array([pts[a == k].mean(0) for k in np.unique(a)])
    a = np.argmin(np.linalg.norm(pts[:, None] - c[None], axis=2), 1)
    r = np.array([np.linalg.norm(pts[a == k] - c[k], axis=1).max() + tol for k in range(len(c))])
    return c, r


def cover_quality(mesh, c, r):
    """(fraction of surface points inside a sphere, 95th percentile of how far the spheres stick out of the mesh [m])."""
    pts, _ = trimesh.sample.sample_surface_even(mesh, 3000, seed=0)
    inside = ((np.linalg.norm(pts[:, None] - c[None], axis=2) - r[None]).min(1) < 0.002).mean()
    sph = np.concatenate([ci + ri * trimesh.sample.sample_surface_sphere(200) for ci, ri in zip(c, r)])
    sph = sph[~(np.linalg.norm(sph[:, None] - c[None], axis=2) < r[None] - 1e-6).any(1)]
    return float(inside), float(np.percentile(trimesh.proximity.closest_point(mesh, sph)[1], 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-max", type=float, default=0.02, help="largest grid cell of the gripper sphere cover [m]")
    args = ap.parse_args()

    robot = yourdfpy.URDF.load(str(URDF), build_scene_graph=False, load_meshes=False)
    template = load_yaml(join_path(get_robot_configs_path(), "ur5e_robotiq_2f_140.yml"))["robot_cfg"]["kinematics"]

    spheres = {P + k: template["collision_spheres"][k] for k in ARM_LINKS}
    for link in GRIPPER_LINKS:
        mesh = link_collision_mesh(robot, link)
        pts, rad = sphere_cover(mesh, args.cell_max)
        cov, over = cover_quality(mesh, pts, rad)
        spheres[link] = [{"center": [round(float(v), 4) for v in p], "radius": round(float(r), 4)} for p, r in zip(pts, rad)]
        print(f"{link:26s} extents {np.round(mesh.extents, 3)} -> {len(pts):3d} spheres, r <= {rad.max() * 1000:.0f} mm, "
              f"covers {cov:.0%} of the surface, sticks out <= {over * 1000:.0f} mm (95th pct)")

    grip = list(GRIPPER_LINKS)
    ignore = {
        P + "shoulder_link": [P + "upper_arm_link"],
        P + "upper_arm_link": [P + "forearm_link"],
        P + "forearm_link": [P + "wrist_1_link", P + "wrist_2_link", P + "wrist_3_link"],
        P + "wrist_1_link": [P + "wrist_2_link", P + "wrist_3_link", "robotiq_arg2f_base_link"],
        P + "wrist_2_link": [P + "wrist_3_link", "robotiq_arg2f_base_link"] + grip[1:3],
        P + "wrist_3_link": grip,
    }
    for i, a in enumerate(grip):  # the gripper never collides with itself in a way we can plan around (fingers are locked)
        if grip[i + 1:]:
            ignore[a] = grip[i + 1:]

    arm_joints = list(n.ARM_JOINTS)
    cfg = {"robot_cfg": {"kinematics": {
        "urdf_path": str(URDF),
        "asset_root_path": str(URDF.parent),
        "usd_path": None,
        "base_link": P + "base_link",
        "ee_link": "grasp_frame",
        "link_names": None,
        "lock_joints": {n.GRIPPER_DRIVEN_JOINT: 0.0},
        "extra_links": {"grasp_frame": {
            "parent_link_name": n.EE_LINK, "link_name": "grasp_frame", "joint_name": "grasp_frame_joint", "joint_type": "FIXED",
            "fixed_transform": [0.0, 0.0, float(n.TCP_OFFSET_Z), 1.0, 0.0, 0.0, 0.0],  # x y z qw qx qy qz
        }},
        "collision_link_names": list(spheres),
        "collision_spheres": spheres,
        "collision_sphere_buffer": 0.002,
        "extra_collision_spheres": {"attached_object": 40},
        "self_collision_ignore": ignore,
        "self_collision_buffer": {P + "shoulder_link": 0.01},
        "use_global_cumul": True,
        "mesh_link_names": [P + k for k in ARM_LINKS] + grip,
        "cspace": {
            "joint_names": arm_joints + [n.GRIPPER_DRIVEN_JOINT],
            "retract_config": list(n.ARM_READY) + [0.0],
            "null_space_weight": [1.0] * 7,
            "cspace_distance_weight": [1.0] * 7,
            "max_jerk": 500.0,
            "max_acceleration": 12.0,
            "position_limit_clip": 0.0,
        },
    }}}
    OUT.write_text("# generated by make_robot_cfg.py -- edit that script, not this file\n" + yaml.safe_dump(cfg, sort_keys=False, width=140))
    print("wrote", OUT)

    # ---- check: cuRobo's TCP at the ready pose must equal the sim's (robot_defs / ur5e_env READY_TCP, tool down) ----
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType

    tdt = TensorDeviceType()
    model = CudaRobotModel(CudaRobotModelConfig.from_robot_yaml_file(load_yaml(str(OUT))["robot_cfg"], tensor_args=tdt))
    print("cuRobo joints:", model.joint_names)
    q = tdt.to_device(torch.tensor([list(n.ARM_READY)]))
    st = model.get_state(q)
    pos, quat = st.ee_position[0].cpu().numpy(), st.ee_quaternion[0].cpu().numpy()
    print(f"ready pose: TCP {np.round(pos, 4)} (sim READY_TCP 0.395 0 0.034), quat wxyz {np.round(quat, 4)} (tool down: +-(0 1 0 0))")
    err = np.linalg.norm(pos - np.array([0.395, 0.0, 0.034]))
    assert err < 0.005, f"TCP mismatch {err:.4f} m"
    assert abs(abs(quat[1]) - 1.0) < 0.01, "tool not pointing down at the ready pose"
    print("OK: cuRobo kinematics match the sim's ready pose")


if __name__ == "__main__":
    main()
