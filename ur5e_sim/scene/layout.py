"""Geometry of the side-bins scene as plain numbers (no Isaac imports): table, three drawer housings with sliding crates.

Used by the Isaac scene (sidebins_curobo.py) and by the cuRobo collision world. Frames:
  * world: z up, arm base on the table top at the origin (as in ur5e_env / sidebins_test.py)
  * base: the arm-base frame = world shifted down by TABLE_Z (what the planner and the arm actions use)
  * housing i: origin on the floor under the housing centre, local +x = pull direction; placed at (BIN_X[i], ORIGIN_Y, 0)
    and yawed +90 deg, so local +x -> world +y and local +y -> world -x
  * crate: origin at the centre of the crate bottom, offset by the slide along housing +x

Numbers MUST match build_pullbin.py (the USD's colliders) and sidebins_test.py (the placement).
"""
from __future__ import annotations

import numpy as np

TABLE_Z = 0.75
TABLE_SIZE = (1.8, 0.6)  # x, y; centred on the arm base
TABLE_EDGE_Y = -0.30  # the housings stand against this edge

# ---- build_pullbin.py ----
Z_B = TABLE_Z  # housing base plate top
T, IN_HALF_W, BACK_X, FRONT_X, WALL_H, WINDOW = 0.02, 0.215, -0.315, 0.32, 0.25, 0.20
W = IN_HALF_W + T
ROOF_X = (BACK_X - T, FRONT_X - WINDOW)
CL, CW, CH = 0.6052, 0.4052, 0.1703
WALL_IN_X, SIDE_IN_Y, FLOOR_T, BAR_Z0, BAR_OUT_X, CRATE_Z0 = 0.290, 0.1965, 0.006, 0.144, 0.3026, 0.002
SLIDE_MAX = 0.35

# ---- placement (sidebins_test.py) ----
PITCH = 2 * W
BIN_X = (-PITCH, 0.0, PITCH)
ORIGIN_Y = TABLE_EDGE_Y - FRONT_X
FLOOR_TOP = Z_B + CRATE_Z0 + FLOOR_T  # inside floor of a crate (world z)
WALL_Y = ORIGIN_Y + 0.5 * (WALL_IN_X + BAR_OUT_X)  # middle of the crate's front end wall incl. rim bar (world y), closed
BAR_Z = Z_B + CRATE_Z0 + 0.5 * (BAR_Z0 + CH)  # middle of the rim bar (world z)

HOUSING_BOXES = {  # name: (lo, hi) in the housing frame
    "plinth": ((BACK_X - T, -W, 0.0), (FRONT_X, W, Z_B - T)),
    "base": ((BACK_X - T, -W, Z_B - T), (FRONT_X, W, Z_B)),
    "side_l": ((BACK_X - T, IN_HALF_W, Z_B), (FRONT_X, W, Z_B + WALL_H)),
    "side_r": ((BACK_X - T, -W, Z_B), (FRONT_X, -IN_HALF_W, Z_B + WALL_H)),
    "back": ((BACK_X - T, -IN_HALF_W, Z_B), (BACK_X, IN_HALF_W, Z_B + WALL_H)),
    "roof": ((ROOF_X[0], -W, Z_B + WALL_H), (ROOF_X[1], W, Z_B + WALL_H + T)),
}
CRATE_BOXES = {  # name: (lo, hi) in the crate frame
    "floor": ((-CL / 2, -CW / 2, 0.0), (CL / 2, CW / 2, FLOOR_T)),
    "front_wall": ((WALL_IN_X, -CW / 2, 0.0), (WALL_IN_X + 0.0075, CW / 2, CH)),
    "front_bar": ((WALL_IN_X + 0.0075, -CW / 2, BAR_Z0), (BAR_OUT_X, CW / 2, CH)),
    "rear_wall": ((-WALL_IN_X - 0.0075, -CW / 2, 0.0), (-WALL_IN_X, CW / 2, CH)),
    "rear_bar": ((-BAR_OUT_X, -CW / 2, BAR_Z0), (-WALL_IN_X - 0.0075, CW / 2, CH)),
    "side_l": ((-CL / 2, SIDE_IN_Y, 0.0), (CL / 2, CW / 2, CH)),
    "side_r": ((-CL / 2, -CW / 2, 0.0), (CL / 2, -SIDE_IN_Y, CH)),
}
HOUSING_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)  # +90 deg about z (w, x, y, z)


def housing_to_world(i: int, p) -> np.ndarray:
    """Point in housing i's frame -> world (yaw +90 deg: (x, y) -> (-y, x))."""
    x, y, z = p
    return np.array([BIN_X[i] - y, ORIGIN_Y + x, z])


def crate_to_world(i: int, xc: float, yc: float, z: float, slide: float = 0.0) -> np.ndarray:
    """Crate-local (x along the pull axis, y) of bin i (crate pulled out by ``slide``) -> world, z given in world."""
    return np.array([BIN_X[i] - yc, ORIGIN_Y + xc + slide, z])


def _box_world(i, lo, hi, shift=(0.0, 0.0, 0.0)):
    lo, hi = np.asarray(lo, float) + shift, np.asarray(hi, float) + shift
    c = housing_to_world(i, (lo + hi) / 2)
    d = hi - lo
    return c, np.array([d[1], d[0], d[2]])  # yaw 90: the housing's x extent lies along world y


def bin_boxes(i: int, slide: float = 0.0, frame: str = "base"):
    """[(name, centre, dims)] of housing i and its crate (pulled out by ``slide``), axis aligned in the world."""
    dz = TABLE_Z if frame == "base" else 0.0
    out = []
    for nm, (lo, hi) in HOUSING_BOXES.items():
        c, d = _box_world(i, lo, hi)
        out.append((f"bin{i}_housing_{nm}", c - [0, 0, dz], d))
    for nm, (lo, hi) in CRATE_BOXES.items():
        c, d = _box_world(i, lo, hi, shift=(slide, 0.0, Z_B + CRATE_Z0))
        out.append((f"bin{i}_crate_{nm}", c - [0, 0, dz], d))
    return out


def crate_box_names(i: int):
    return [f"bin{i}_crate_{nm}" for nm in CRATE_BOXES]


def table_box(frame: str = "base"):
    dz = TABLE_Z if frame == "base" else 0.0
    return ("table", np.array([0.0, 0.0, TABLE_Z / 2 - dz]), np.array([TABLE_SIZE[0], TABLE_SIZE[1], TABLE_Z]))


# ---- shelf (opposite the drawers, +y side): two levels, open top, wooden boards like the housings ----
SHELF_FRONT_Y = 0.32  # just behind the container's far wall (the table edge is at +0.30)
SHELF_DEPTH = 0.30  # inner depth (y)
SHELF_X = (-0.45, 0.45)  # outer x extent
SHELF_LEVELS = (TABLE_Z + 0.15, TABLE_Z + 0.47)  # top faces of the two boards (world z), 0.30 m clear between them. Not at table height: side grasps 8-12 cm above the table put the wrist into it (straight retreats failed for 74 of 80 grasps)
SHELF_SIDE_H = 1.50  # top of the side walls (world z)
SHELF_BOXES = {  # name: (lo, hi) in the world frame
    "plinth": ((SHELF_X[0], SHELF_FRONT_Y, 0.0), (SHELF_X[1], SHELF_FRONT_Y + SHELF_DEPTH, SHELF_LEVELS[0] - T)),
    "board0": ((SHELF_X[0] + T, SHELF_FRONT_Y, SHELF_LEVELS[0] - T), (SHELF_X[1] - T, SHELF_FRONT_Y + SHELF_DEPTH, SHELF_LEVELS[0])),
    "board1": ((SHELF_X[0] + T, SHELF_FRONT_Y, SHELF_LEVELS[1] - T), (SHELF_X[1] - T, SHELF_FRONT_Y + SHELF_DEPTH, SHELF_LEVELS[1])),
    "side_l": ((SHELF_X[0], SHELF_FRONT_Y, 0.0), (SHELF_X[0] + T, SHELF_FRONT_Y + SHELF_DEPTH, SHELF_SIDE_H)),
    "side_r": ((SHELF_X[1] - T, SHELF_FRONT_Y, 0.0), (SHELF_X[1], SHELF_FRONT_Y + SHELF_DEPTH, SHELF_SIDE_H)),
    "back": ((SHELF_X[0], SHELF_FRONT_Y + SHELF_DEPTH, 0.0), (SHELF_X[1], SHELF_FRONT_Y + SHELF_DEPTH + T, SHELF_SIDE_H)),
}
SHELF_SPOTS_X = (-0.32, -0.11, 0.11, 0.32)  # nominal object x positions on a level (jittered per episode)
SHELF_SPOTS_Y = (SHELF_FRONT_Y + 0.05, SHELF_FRONT_Y + 0.14)  # object centre y range on a level

# ---- container: Isaac's shallow grey tray (Props/PackingTable/props/container_h20), on the table at the +x end ----
# 0.612 x 0.412 x 0.09 m, origin at the bottom centre, long side along x, its +x edge flush with the table's +x end.
# Across the table it has to fit between a pulled-out crate (reaches y = -0.17) and the shelf (y >= 0.32).
CONTAINER_USD = "Props/PackingTable/props/container_h20/container_h20.usd"
CONTAINER_POS = (TABLE_SIZE[0] / 2 - 0.306, 0.105, TABLE_Z)  # world
CONTAINER_BOXES = {  # its own box colliders (from the USD), container frame
    "floor": ((-0.3, -0.2, 0.0005), (0.3, 0.2, 0.0085)),
    "wall_yp": ((-0.3, 0.1925, 0.01), (0.3, 0.2075, 0.09)),
    "wall_ym": ((-0.3, -0.2075, 0.01), (0.3, -0.1925, 0.09)),
    "wall_xm": ((-0.3075, -0.19, 0.01), (-0.2925, 0.19, 0.09)),
    "wall_xp": ((0.2925, -0.19, 0.01), (0.3075, 0.19, 0.09)),
}
CONTAINER_FLOOR_TOP = TABLE_Z + 0.0085  # world z
CONTAINER_INNER = ((CONTAINER_POS[0] - 0.2925, CONTAINER_POS[0] + 0.2925), (CONTAINER_POS[1] - 0.1925, CONTAINER_POS[1] + 0.1925))
CONTAINER_DROP_X = (CONTAINER_INNER[0][0], CONTAINER_POS[0] + 0.05)  # the far half of the tray is out of the arm's reach


def shelf_boxes(frame: str = "base"):
    dz = TABLE_Z if frame == "base" else 0.0
    out = []
    for nm, (lo, hi) in SHELF_BOXES.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out.append((f"shelf_{nm}", (lo + hi) / 2 - [0, 0, dz], hi - lo))
    return out


def container_boxes(frame: str = "base"):
    dz = TABLE_Z if frame == "base" else 0.0
    out = []
    for nm, (lo, hi) in CONTAINER_BOXES.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out.append((f"container_{nm}", (lo + hi) / 2 + np.asarray(CONTAINER_POS) - [0, 0, dz], hi - lo))
    return out


def world_config(bins=(0, 1, 2), slides=(0.0, 0.0, 0.0), objects=(), shelf=False, container=False):
    """cuRobo WorldConfig in the base frame. ``objects``: [(name, obj_file, pose7 in the base frame)]."""
    from curobo.geom.types import Cuboid, Mesh, WorldConfig

    boxes = [table_box()] + [b for i in bins for b in bin_boxes(i, slides[i])]
    boxes += shelf_boxes() if shelf else []
    boxes += container_boxes() if container else []
    cub = [Cuboid(name=nm, pose=[*map(float, c), 1.0, 0.0, 0.0, 0.0], dims=[*map(float, d)]) for nm, c, d in boxes]
    meshes = [Mesh(name=nm, file_path=str(f), pose=[*map(float, p)]) for nm, f, p in objects]
    return WorldConfig(cuboid=cub, mesh=meshes)
