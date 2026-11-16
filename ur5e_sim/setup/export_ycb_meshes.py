"""Export the Isaac YCB objects as plain triangle meshes (.obj, metres, in the object's root frame) plus a summary json.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python export_ycb_meshes.py --headless [--inspect]

Why: the grasp sampler (trimesh) and the cuRobo collision world need the object geometry outside Isaac Sim. The frame of
each exported mesh is the frame of the USD's default prim, which is the frame Isaac Lab reports as the object's root pose
(``RigidObject.data.root_pos_w`` / ``root_quat_w``) when the file is spawned with UsdFileCfg. ``--inspect`` also prints the
prim tree (which prim carries the rigid body / collision APIs) so that assumption can be checked per file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ur5e_sim/
import paths  # noqa: E402
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--ycb", type=Path, default=paths.ISAAC / "Props/YCB")
ap.add_argument("--out", type=Path, default=paths.YCB_MESHES)
ap.add_argument("--inspect", action="store_true", help="print the prim tree of every file")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def mesh_in_root_frame(stage, purposes):
    """All Mesh prims under the default prim with one of the given purposes, triangulated, in the default prim's frame."""
    root = stage.GetDefaultPrim()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    to_root = np.array(xc.GetLocalToWorldTransform(root)).T  # USD matrices are row-major with row vectors
    to_root = np.linalg.inv(to_root)
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    parts = []
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputePurpose() not in purposes:
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), float)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), int)
        faces, k = [], 0
        for c in counts:  # fan triangulation (faces are convex polygons in these assets)
            faces += [[idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1)]
            k += c
        T = to_root @ np.array(xc.GetLocalToWorldTransform(p)).T
        pts_h = np.c_[pts, np.ones(len(pts))] @ T.T
        parts.append(trimesh.Trimesh(pts_h[:, :3] * mpu, np.asarray(faces), process=True))
    return trimesh.util.concatenate(parts) if parts else None


args.out.mkdir(parents=True, exist_ok=True)
summary = {}
for variant in ("Axis_Aligned", "Axis_Aligned_Physics"):
    for f in sorted((args.ycb / variant).glob("*.usd")):
        stage = Usd.Stage.Open(str(f))
        root = stage.GetDefaultPrim()
        if args.inspect:
            print(f"\n== {variant}/{f.name}: mpu {UsdGeom.GetStageMetersPerUnit(stage)} up {UsdGeom.GetStageUpAxis(stage)} default {root.GetPath()}")
            for p in Usd.PrimRange(root):
                apis = [a for a in p.GetAppliedSchemas() if "Physics" in a or "Physx" in a]
                pur = UsdGeom.Imageable(p).ComputePurpose() if p.IsA(UsdGeom.Imageable) else ""
                print(f"   {p.GetPath()} [{p.GetTypeName()}] {pur} {apis}")
        mesh = mesh_in_root_frame(stage, {"default", "render"})
        if mesh is None:
            print(f"!! {variant}/{f.name}: no render mesh")
            continue
        name = f.stem
        key = name if variant == "Axis_Aligned" else name + "_physics"
        out = args.out / f"{key}.obj"
        mesh.export(out)
        lo, hi = mesh.bounds
        summary[key] = dict(usd=str(f), obj=str(out), bounds_lo=lo.round(4).tolist(), bounds_hi=hi.round(4).tolist(),
                            extents=(hi - lo).round(4).tolist(), watertight=bool(mesh.is_watertight), faces=len(mesh.faces))
        print(f"{key:32s} extents {np.round(hi - lo, 3)}  centre {np.round((hi + lo) / 2, 3)}  faces {len(mesh.faces)}  watertight {mesh.is_watertight}")
(args.out / "summary.json").write_text(json.dumps(summary, indent=1))
print("wrote", args.out / "summary.json", flush=True)
app.close()
