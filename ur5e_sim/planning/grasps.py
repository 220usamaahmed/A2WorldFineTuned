"""Antipodal grasp sampler for the Robotiq 2F-85 on object meshes (ground-truth geometry, any object pose).

Grasp frame = the planner's TCP frame (``grasp_frame``): origin midway between the finger pads, x = closing axis (the pads
move along +-x), z = approach direction (from the palm towards the fingertips).

Sampling (object frame):
  1. points p1 + outward normals n1 on the surface;
  2. a ray from p1 into the object (-n1) finds the opposite surface point p2 (normal n2) -> grasp axis a = p2 - p1;
  3. keep pairs with width < the open gap minus a margin and both contact normals inside the friction cone around a;
  4. each pair gives ``n_roll`` grasps: the approach direction z is swept around the axis a (the roll of the gripper about
     the closing axis) and the TCP can sit ``depths`` deeper/shallower along z;
  5. reject grasps whose open gripper (palm + finger boxes in the grasp frame, swept back along -z over the approach
     distance) overlaps the object's surface points.
Robot-side checks (reachability, collisions with table / crate / shelf) are left to the planner: ArmPlanner.ik_reachable
and plan_grasp, with the full world model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

OPEN_GAP = 0.085  # pad-to-pad distance, fully open
PAD = dict(x=0.00635, y=0.022, z=0.0375)  # pad box (thickness along the closing axis, width, length along approach)

# Open gripper as boxes in the grasp frame: (lo, hi). Measured from assets/ur5e_gripper.urdf at finger_joint = 0 with
# ``gripper_boxes_from_urdf`` below (rounded outward); pads and inner fingers hang below the palm on each side.
GRIPPER_BOXES = {}  # finger angle -> boxes, filled lazily by gripper_boxes()

# Pad travel of the 2F-85 four-bar linkage (from the URDF, finger_joint 0 ... 0.8 rad): inner gap between the pads and how
# far the pad centres move along the approach axis (+z) from their open position. Closing on a 65 mm can moves the pads
# 7 mm towards the fingertips; fully closed, 14 mm.
PAD_GAP = np.array([86.0, 77.1, 67.4, 57.2, 46.6, 35.6, 24.3, 12.9, 1.5]) / 1000
PAD_DZ = np.array([0.0, 3.5, 6.6, 9.2, 11.2, 12.8, 13.7, 14.1, 13.9]) / 1000


PAD_DZ_LINKAGE = PAD_DZ  # the real / four-bar pad path (above)
# The simulated gripper (make_simple_gripper.py) slides its physics pads on the straight line from open to closed, so
# their travel along the approach axis is linear in the gap: 13.9 mm over 86 -> 1.5 mm (the visual linkage stays within
# 4.7 mm of it). Grasps are placed for these pads.
PAD_DZ = PAD_DZ[-1] * (PAD_GAP[0] - PAD_GAP) / (PAD_GAP[0] - PAD_GAP[-1])


def closing_dz(width: float) -> float:
    """How far the (simulated) pads move along the approach axis while closing from open onto an object ``width`` wide."""
    return float(np.interp(width, PAD_GAP[::-1], PAD_DZ[::-1]))


GRIPPER_BOX_LINKS = ("ur5e_wrist_3_link", "robotiq_coupling", "robotiq_arg2f_base_link", "left_outer_knuckle",
                     "right_outer_knuckle", "left_outer_finger", "right_outer_finger", "left_inner_knuckle",
                     "right_inner_knuckle", "left_inner_finger", "right_inner_finger", "left_inner_finger_pad",
                     "right_inner_finger_pad")


def gripper_boxes_from_urdf(margin: float = 0.003, slices: int = 4, finger: float = 0.0) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Boxes (grasp frame) covering the open gripper's collision geometry (finger_joint = 0), inflated by ``margin``.
    Each link is cut into ``slices`` along its longest principal axis with one axis-aligned box per slice: a single
    box around a slanted link (the inner knuckles run diagonally from the palm to the fingers) would block the space
    between the fingers that the object needs."""
    import sys

    import yourdfpy

    sim = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(sim))
    import robot_defs as n

    import paths

    urdf = yourdfpy.URDF.load(str(paths.ROBOT_URDF), build_collision_scene_graph=True,
                              load_collision_meshes=True, build_scene_graph=False, load_meshes=False)
    cfg = {j: 0.0 for j in urdf.actuated_joint_names}
    cfg[n.GRIPPER_DRIVEN_JOINT] = finger
    urdf.update_cfg(cfg)
    T_grasp = urdf.get_transform(n.EE_LINK, collision_geometry=True) @ trimesh.transformations.translation_matrix([0, 0, n.TCP_OFFSET_Z])
    inv = np.linalg.inv(T_grasp)
    sc = urdf.collision_scene
    boxes = []
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        link = sc.graph.transforms.parents.get(node, "")
        if link not in GRIPPER_BOX_LINKS:
            continue
        pts = trimesh.transform_points(sc.geometry[gname].vertices, inv @ T)
        c = pts - pts.mean(0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]
        t = c @ axis
        edges = np.linspace(t.min(), t.max(), slices + 1)
        for k in range(slices):
            sel = (t >= edges[k] - 1e-9) & (t <= edges[k + 1] + 1e-9)
            if sel.sum() < 3:
                continue
            boxes.append((pts[sel].min(0) - margin, pts[sel].max(0) + margin, link))
    return boxes


def gripper_boxes(finger: float = 0.0):
    if finger not in GRIPPER_BOXES:
        GRIPPER_BOXES[finger] = [(lo, hi) for lo, hi, _ in gripper_boxes_from_urdf(finger=finger)]
    return GRIPPER_BOXES[finger]


@dataclass
class Grasp:
    T: np.ndarray  # 4x4 grasp frame in the object frame
    width: float  # object width between the contacts [m]
    score: float
    axis_id: int  # which antipodal pair it came from (grasps with the same id differ only in roll / depth)

    def pose7(self, T_obj: np.ndarray | None = None) -> np.ndarray:
        """(x y z qw qx qy qz) of the grasp, in the object frame or, with T_obj (object pose), in that frame's parent."""
        T = self.T if T_obj is None else T_obj @ self.T
        x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
        return np.r_[T[:3, 3], w, x, y, z]


def T_from_pose7(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(p[3:7])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


class AntipodalSampler:
    def __init__(self, mesh: trimesh.Trimesh, friction: float = 0.5, width_margin: float = 0.008, min_width: float = 0.004,
                 n_surface: int = 2500, approach: float = 0.10, seed: int = 0):
        self.mesh = mesh
        self.cos_cone = np.cos(np.arctan(friction))
        self.max_w, self.min_w = OPEN_GAP - width_margin, min_width
        self.approach = approach
        self.rng = np.random.default_rng(seed)
        self.cloud, _ = trimesh.sample.sample_surface_even(mesh, n_surface, seed=seed)
        self.boxes = gripper_boxes()
        self.boxes_swept = [(np.r_[lo[:2], lo[2] - approach], hi) for lo, hi in self.boxes]  # + the approach path
        self.centroid = mesh.centroid if mesh.is_watertight else mesh.bounding_box.centroid

    # ------------------------------------------------------------------ antipodal pairs
    def pairs(self, n_points: int = 600):
        p1, fid = trimesh.sample.sample_surface(self.mesh, n_points, seed=int(self.rng.integers(1 << 31)))
        n1 = self.mesh.face_normals[fid]
        origins = p1 - 1e-4 * n1
        locs, ray_id, tri = self.mesh.ray.intersects_location(origins, -n1, multiple_hits=True)
        out = []
        for k in range(n_points):
            sel = np.where(ray_id == k)[0]
            if len(sel) == 0:
                continue
            d = np.linalg.norm(locs[sel] - p1[k], axis=1)
            sel, d = sel[d > 1e-3], d[d > 1e-3]
            if len(sel) == 0:
                continue
            j = sel[np.argmin(d)]  # first surface the ray meets on the way through
            p2, n2 = locs[j], self.mesh.face_normals[tri[j]]
            a = p2 - p1[k]
            w = np.linalg.norm(a)
            if not (self.min_w < w < self.max_w):
                continue
            a /= w
            # friction cones; |dot| because some YCB meshes have inconsistently oriented faces
            if abs(a @ n1[k]) < self.cos_cone or abs(a @ n2) < self.cos_cone:
                continue
            out.append((p1[k], p2, a, w, min(abs(a @ n1[k]), abs(a @ n2))))
        return out

    # ------------------------------------------------------------------ grasp frames
    def collides(self, Ts: np.ndarray) -> np.ndarray:
        """[M] bool: open gripper (swept back along -z over the approach distance) of each grasp frame Ts [M, 4, 4]
        overlaps the object's surface points."""
        inv = np.linalg.inv(Ts)
        pts = np.einsum("mij,pj->mpi", inv[:, :3, :3], self.cloud) + inv[:, None, :3, 3]  # [M, P, 3]
        hit = np.zeros(len(Ts), bool)
        for lo, hi in self.boxes_swept:
            hit |= np.any(np.all((pts > lo) & (pts < hi), axis=2), axis=1)
        return hit

    def sample(self, n_pairs: int = 600, n_roll: int = 12, depths=(0.0, -0.008, -0.016), max_grasps: int | None = None) -> list[Grasp]:
        grasps = []
        for pid, (p1, p2, a, w, anti) in enumerate(self.pairs(n_pairs)):
            c = 0.5 * (p1 + p2)
            # an arbitrary vector perpendicular to a, then sweep the approach direction around a
            ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
            u = np.cross(a, ref)
            u /= np.linalg.norm(u)
            v = np.cross(a, u)
            th = np.linspace(0, 2 * np.pi, n_roll, endpoint=False) + self.rng.uniform(0, 2 * np.pi / n_roll)
            Ts = []
            for t in th:
                z = np.cos(t) * u + np.sin(t) * v
                y = np.cross(z, a)
                for dz in depths:
                    T = np.eye(4)
                    T[:3, :3] = np.stack([a, y, z], axis=1)
                    # the pads move towards the fingertips while closing: put the TCP (open pad centre) that much short
                    # of the contact midpoint, so the pads meet the object at the sampled contacts
                    T[:3, 3] = c + (dz - closing_dz(w)) * z
                    Ts.append(T)
            Ts = np.array(Ts)
            free = ~self.collides(Ts)
            off = np.linalg.norm(np.cross(self.centroid - c, a))
            # score: antipodality, closeness of the grasp axis to the object's centre, width margin
            score = anti - 2.0 * off + 0.5 * (self.max_w - w) / self.max_w
            grasps += [Grasp(T, float(w), float(score), pid) for T in Ts[free]]
        grasps.sort(key=lambda g: -g.score)
        return grasps[:max_grasps] if max_grasps else grasps


# objects whose simulated collider is a convex hull (setup/make_ycb_physics.py HULL, and Isaac's own physics variants of
# the cracker box, sugar box and soup can): grasps must be sampled on the hull, not on the visual mesh (whose can rims
# offered 4 mm "grasps" that do not exist in the physics)
HULL_COLLIDER = {"002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "007_tuna_fish_can",
                 "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "036_wood_block", "040_large_marker",
                 "061_foam_brick", "003_cracker_box_physics", "004_sugar_box_physics", "005_tomato_soup_can_physics"}


def collider_mesh(key: str, obj_file) -> trimesh.Trimesh:
    m = load_mesh(obj_file)
    return m.convex_hull if key in HULL_COLLIDER else m


def cached_grasps(key: str, obj_file, cache_dir=None, **kw) -> list[Grasp]:
    """Object-frame grasps of one object (sampled on its simulated collider shape), stored as <cache_dir>/<key>.npz."""
    if cache_dir is None:
        import paths

        cache_dir = paths.GRASP_CACHE
    f = Path(cache_dir) / f"{key}.npz"
    if f.exists():
        d = np.load(f)
        return [Grasp(T, float(w), float(s), int(a)) for T, w, s, a in zip(d["T"], d["width"], d["score"], d["axis_id"])]
    g = AntipodalSampler(collider_mesh(key, obj_file), **{k: v for k, v in kw.items() if k in ("friction", "width_margin", "approach", "seed")}).sample(
        **{k: v for k, v in kw.items() if k in ("n_pairs", "n_roll", "depths")})
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, T=np.array([x.T for x in g]).reshape(-1, 4, 4), width=np.array([x.width for x in g]),
             score=np.array([x.score for x in g]), axis_id=np.array([x.axis_id for x in g]))
    return g


def load_mesh(obj_file) -> trimesh.Trimesh:
    m = trimesh.load(str(obj_file), force="mesh")
    m.merge_vertices()
    return m


def stable_poses(mesh: trimesh.Trimesh, min_prob: float = 0.05):
    """Resting orientations of the mesh on a plane: [(4x4 transform putting the mesh on z = 0, probability)]."""
    Ts, probs = mesh.compute_stable_poses(n_samples=1, threshold=0.0)
    return [(T, float(p)) for T, p in zip(Ts, probs) if p >= min_prob]


def closing_angle(width: float) -> float:
    """finger_joint angle at which the pads are ``width`` apart."""
    return float(np.interp(width, PAD_GAP[::-1], np.linspace(0.8, 0.0, len(PAD_GAP))))


def world_filter(grasps_T: np.ndarray, floor_z: float, min_clear: float = 0.005, widths=None, extra_close: float = 0.010) -> np.ndarray:
    """Cheap pre-filter in the planner frame (before IK): keep grasps [N, 4, 4] whose gripper stays ``min_clear`` above
    ``floor_z`` (table top / crate floor), checked open and closed to ``width - extra_close`` (the pads travel up to
    14 mm towards the fingertips while closing; a finger that hits the table stalls before it reaches the object).
    Without ``widths``, fully closed."""
    grasps_T = np.asarray(grasps_T).reshape(-1, 4, 4)
    widths = np.zeros(len(grasps_T)) if widths is None else np.asarray(widths, float)
    angles = np.ceil(np.array([closing_angle(max(w - extra_close, 0.0)) for w in widths]) * 10) / 10  # grid of 0.1 rad
    keep = np.zeros(len(grasps_T), bool)
    for a in np.unique(angles):
        sel = angles == a
        corners = np.array([[[lo[0], hi[0]][i], [lo[1], hi[1]][j], [lo[2], hi[2]][k]] for f in (0.0, float(a)) for lo, hi in gripper_boxes(f)
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        T = grasps_T[sel]
        zc = np.einsum("nij,mj->nmi", T[:, :3, :3], corners)[..., 2] + T[:, None, 2, 3]
        keep[sel] = zc.min(axis=1) > floor_z + min_clear
    return keep


def graspable_stable_poses(mesh: trimesh.Trimesh, grasps: list[Grasp], min_grasps: int = 20, min_prob: float = 0.02):
    """Stable resting poses [(4x4 on z = 0, probability, n_grasps)] in which at least ``min_grasps`` of the object's grasps
    keep the gripper above the support (e.g. a pudding box lying flat offers only its vertical 38 mm side and 90 / 114 mm
    sides wider than the gripper: no grasp)."""
    Gs = np.array([g.T for g in grasps]).reshape(-1, 4, 4)
    out = []
    for T, p in stable_poses(mesh, min_prob=min_prob):
        k = int(world_filter(T @ Gs, floor_z=0.0, min_clear=0.004, widths=[g.width for g in grasps]).sum()) if len(Gs) else 0
        if k >= min_grasps:
            out.append((T, p, k))
    return out
