"""Pick-and-place building blocks on top of ArmPlanner: getting a grasped object out of a drawer crate or a shelf, and
drop poses in the container. Used by scene/pick_place.py.

Frames as in planner.py (arm-base frame); 4x4 matrices for poses, pose7 = (x y z qw qx qy qz).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scene"))
import layout as L  # noqa: E402
from planner import ArmPlanner, Plan  # noqa: E402

LIFT = 0.03  # first lift off the support, object ignored by the planner (it is still touching the support)


def pose7(T) -> np.ndarray:
    x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], w, x, y, z]


def T_of(p7) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(p7[3:7])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = p7[:3]
    return T


def plan_extraction(pl: ArmPlanner, q_grasp, T_grasp, T_obj, obj: str, source: str, keep_attached: bool = False):
    """Plans that take an object, grasped at q_grasp (TCP pose T_grasp, object pose T_obj), out of its place:
    (1) straight up by LIFT with the object ignored; (2) attach it; (3) out: drawer -> straight up 8 cm more (clear of the
    crate walls); shelf -> straight towards the arm (-y) until the object is 10 cm in front of the shelf.
    Returns [lift, out] or None. Restores the planner's world (object detached, back at T_obj) unless keep_attached."""
    p_g = pose7(T_grasp)
    with pl.relaxed(ignore=[obj]):
        lift = pl.plan_cartesian(q_grasp, p_g + np.r_[0, 0, LIFT, 0, 0, 0, 0], max_speed=0.04, quiet=True)
    if lift is None:
        return None
    T_up = T_obj.copy()
    T_up[2, 3] += LIFT
    pl.set_obstacle_pose(obj, pose7(T_up))
    pl.attach(lift.q[-1], obj)
    p_up = p_g + np.r_[0, 0, LIFT, 0, 0, 0, 0]
    p_out = p_up.copy()
    if source == "shelf":
        p_out[1] = min(p_up[1], L.SHELF_FRONT_Y - 0.10 - (T_obj[1, 3] - T_grasp[1, 3]))
    else:
        p_out[2] += 0.08
    out = pl.plan_cartesian(lift.q[-1], p_out, max_speed=0.05, quiet=True) if np.linalg.norm(p_out - p_up) > 1e-3 else Plan(lift.q[-1:], lift.dt, "none")
    if out is None or not keep_attached:
        pl.detach()
        pl.set_obstacle_pose(obj, pose7(T_obj))
    return None if out is None else [lift, out]


def drop_poses(T_tcp_obj, T_obj_rest, support_z, rng, n_xy: int = 3, yaws=(0, 90, -90, 180), clearance: float = 0.015,
               footprint: float = 0.05):
    """TCP poses [N, 7] that hold the object (held as T_tcp_obj = TCP^-1 * object) ``clearance`` above the container
    floor, in the orientation it rested in at its source (T_obj_rest, resting on a support at height support_z, base
    frame), turned about the vertical by each yaw, at n_xy random positions inside the reachable part of the container
    (``footprint``: margin from the walls). The orientation it rested in is stable on a flat floor."""
    out = []
    (x0, x1), (y0, y1) = L.CONTAINER_DROP
    (ix0, ix1), (iy0, iy1) = L.CONTAINER_INNER
    lo_x, hi_x = max(x0, ix0 + footprint), min(x1, ix1 - footprint)  # footprint: margin from the walls only
    lo_y, hi_y = max(y0, iy0 + footprint), min(y1, iy1 - footprint)
    T_held_inv = np.linalg.inv(T_tcp_obj)
    for _ in range(n_xy):
        xy = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
        for yaw in np.radians(yaws):
            Rz = np.eye(4)
            Rz[:3, :3] = R.from_euler("z", yaw).as_matrix()
            T = Rz @ T_obj_rest
            T[:2, 3] = xy
            T[2, 3] = T_obj_rest[2, 3] - support_z + (L.CONTAINER_FLOOR_TOP - L.TABLE_Z) + clearance
            out.append(pose7(T @ T_held_inv))
    return np.array(out)


def clear_container_walls(drops: np.ndarray, width: float, margin: float = 0.015) -> tuple[np.ndarray, np.ndarray]:
    """Raise drop TCP poses [N, 7] so that the whole gripper (closed on an object ``width`` wide) is ``margin`` above
    the container's walls: a tall object held from the side at mid-height would otherwise bring the fingers down to the
    9 cm wall tops. Returns (raised poses, how much each was raised [m]); the object then falls that far."""
    from grasps import closing_angle, gripper_boxes

    a = float(np.ceil(closing_angle(width) * 10) / 10)
    corners = np.array([[[lo[0], hi[0]][i], [lo[1], hi[1]][j], [lo[2], hi[2]][k]] for f in (0.0, a) for lo, hi in gripper_boxes(f)
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    wall_top = max(c[2] + d[2] / 2 for nm, c, d in L.container_boxes() if "wall" in nm)
    out, dz = [], []
    (x0, x1), (y0, y1) = L.CONTAINER_INNER  # world x, y = base x, y
    for p in drops:
        T = T_of(p)
        c = corners @ T[:3, :3].T + T[:3, 3]
        # only gripper parts near or beyond the walls count; fingers well inside the tray may go below the wall tops
        near = (c[:, 0] < x0 + 0.02) | (c[:, 0] > x1 - 0.02) | (c[:, 1] < y0 + 0.02) | (c[:, 1] > y1 - 0.02)
        zmin = float(c[near, 2].min()) if near.any() else np.inf
        lift = max(0.0, wall_top + margin - zmin)
        out.append(p + np.r_[0, 0, lift, 0, 0, 0, 0])
        dz.append(lift)
    return np.array(out), np.array(dz)
