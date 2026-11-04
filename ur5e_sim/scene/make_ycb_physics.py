"""Physics-enabled copies of the Isaac YCB objects (Isaac ships physics for only 4 of its 21 YCB objects).

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python make_ycb_physics.py --headless

For every Props/YCB/Axis_Aligned/*.usd this writes <out>/<name>.usd: a new stage whose default prim references the
original file and adds RigidBodyAPI + MassAPI (the official YCB mass) on the root and CollisionAPI + MeshCollisionAPI on
the mesh: a convex hull for box/can shapes, a convex decomposition for the others. Isaac Lab's spawner only *modifies*
these schemas when they already exist, so they have to be in the file. The root frame is unchanged, so the meshes from
export_ycb_meshes.py (same root frame) stay valid for the grasp sampler and the planner.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--ycb", type=Path, default=Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned"))
ap.add_argument("--out", type=Path, default=Path("/home/data/siddiquieu1/assets/ycb_physics"))
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

# official YCB masses [kg] (Calli et al. 2015)
MASS = {
    "002_master_chef_can": 0.414, "003_cracker_box": 0.411, "004_sugar_box": 0.514, "005_tomato_soup_can": 0.349,
    "006_mustard_bottle": 0.603, "007_tuna_fish_can": 0.171, "008_pudding_box": 0.187, "009_gelatin_box": 0.097,
    "010_potted_meat_can": 0.370, "011_banana": 0.066, "019_pitcher_base": 0.178, "021_bleach_cleanser": 1.131,
    "024_bowl": 0.147, "025_mug": 0.118, "035_power_drill": 0.895, "036_wood_block": 0.729, "037_scissors": 0.082,
    "040_large_marker": 0.0158, "051_large_clamp": 0.125, "052_extra_large_clamp": 0.202, "061_foam_brick": 0.028,
}
HULL = {"002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "007_tuna_fish_can",
        "008_pudding_box", "009_gelatin_box", "010_potted_meat_can", "036_wood_block", "040_large_marker", "061_foam_brick"}

args.out.mkdir(parents=True, exist_ok=True)
for f in sorted(args.ycb.glob("*.usd")):
    name = f.stem
    src = Usd.Stage.Open(str(f))
    src_root = src.GetDefaultPrim()
    meshes = [p.GetPath().MakeRelativePath(src_root.GetPath()) for p in Usd.PrimRange(src_root) if p.IsA(UsdGeom.Mesh)]
    out = args.out / f"{name}.usd"
    st = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(st, UsdGeom.GetStageUpAxis(src))
    UsdGeom.SetStageMetersPerUnit(st, UsdGeom.GetStageMetersPerUnit(src))
    root = st.DefinePrim("/Root", "Xform")
    st.SetDefaultPrim(root)
    root.GetReferences().AddReference(str(f))  # the original's default prim
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(MASS[name])
    for rel in meshes:
        mp = st.GetPrimAtPath(root.GetPath().AppendPath(rel))
        UsdPhysics.CollisionAPI.Apply(mp)
        mc = UsdPhysics.MeshCollisionAPI.Apply(mp)
        if name in HULL:
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            # PhysX's default hull has <= 64 vertices; on rounded boxes (foam brick) that differs from the exact hull the
            # grasp sampler uses by enough to miss grasps near the edges. 255 is the maximum.
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mp).CreateHullVertexLimitAttr(255)
        else:
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)
            cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mp)
            cd.CreateMaxConvexHullsAttr(32)
            cd.CreateHullVertexLimitAttr(64)
            cd.CreateVoxelResolutionAttr(300000)
            cd.CreateShrinkWrapAttr(True)
    st.GetRootLayer().Save()
    print(f"{name:24s} mass {MASS[name]:.3f} kg, {'convex hull' if name in HULL else 'convex decomposition'}, {len(meshes)} mesh(es) -> {out}")
app.close()
