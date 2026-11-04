"""Render every candidate asset on its own and tile the images into labelled contact sheets.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=1 python contact_sheet.py --headless --out <dir>

Assets are read from the local mirror made by fetch_assets.py. Output: <out>/sheet_<group>.png, one per group.
"""
import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--assets", type=Path, default=Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac"))
ap.add_argument("--tile", type=int, default=360)
ap.add_argument("--steps", type=int, default=12)
ap.add_argument("--only", default="", help="comma separated group names")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux  # noqa: E402

A = args.assets
GROUPS = {
    "backdrop": ["Environments/Simple_Warehouse/warehouse.usd", "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"],
    "table": ["Props/PackingTable/packing_table.usd", "Props/Mounts/SeattleLabTable/table_instanceable.usd",
              "Props/Mounts/ThorlabsTable/table_instanceable.usd"],
    "furniture": ["Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd", "Environments/Simple_Warehouse/Props/SM_RackFrame_03.usd",
                  "Environments/Simple_Warehouse/Props/SM_RackShelf_01.usd"],
    "containers": ["Props/KLT_Bin/small_KLT.usd", "Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd",
                   "Environments/Simple_Warehouse/Props/SM_CratePlastic_B_01.usd", "Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd"],
    "pullbins": ["Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd", "Environments/Simple_Warehouse/Props/SM_CratePlastic_B_01.usd",
                 "Environments/Simple_Warehouse/Props/SM_CratePlastic_C_01.usd", "Environments/Simple_Warehouse/Props/SM_CratePlastic_D_01.usd",
                 "Environments/Simple_Warehouse/Props/SM_CratePlastic_E_01.usd", "Environments/Simple_Warehouse/Props/SM_CratePlasticNote_A_01.usd",
                 "Environments/Simple_Warehouse/Props/SM_CratePlasticNote_B_03.usd", "Props/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd",
                 "Props/PackingTable/props/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01.usd", "Props/PackingTable/props/container_h20/container_h20.usd",
                 "Environments/Digital_Twin_Warehouse/Materials/Projects/gtc_sj_2020/original_assets/Sortbot_FBX/NormalKLT.usd", "Props/KLT_Bin/small_KLT.usd"],
    "pullbins2": ["Props/PackingTable/props/SM_Crate_A07_Yellow_01/SM_Crate_A07_Yellow_01.usd", "Props/PackingTable/props/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01.usd",
                  "Environments/Simple_Warehouse/Props/SM_CratePlastic_D_01.usd", "Props/PackingTable/props/container_h20/container_h20.usd"],
    "ycb": sorted(str(p.relative_to(A)) for p in (A / "Props/YCB/Axis_Aligned").glob("*.usd")),
    "other_objects": sorted(str(p.relative_to(A)) for p in (A / "Props/Mugs").glob("*.usd"))
    + [f"Environments/Simple_Warehouse/Props/SM_BottlePlastic{c}_01.usd" for c in "ABCDE"]
    + ["Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", "Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd"],
}
LARGE = {"backdrop"}  # rooms: camera goes inside the bounding box

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
dome = UsdLux.DomeLight.Define(stage, "/Lights/dome")
dome.CreateIntensityAttr(1200)
sun = UsdLux.DistantLight.Define(stage, "/Lights/sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 35))

CAM = "/World/cam"
camera = UsdGeom.Camera.Define(stage, CAM)
camera.CreateFocalLengthAttr(24.0)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 1000.0))  # USD default near plane is 1 m, which cuts off small objects
camera.CreateHorizontalApertureAttr(20.955)
rp = rep.create.render_product(CAM, (args.tile, args.tile))
print("render product created", flush=True)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])


MULTI = {"pullbins2": {"iso": (0.8, -0.9, 0.7), "end_-x": (-1.0, 0.05, 0.35), "end_+x": (1.0, 0.05, 0.35), "side": (0.0, -1.0, 0.3)},
         "pullbins": {"iso": (0.75, -0.9, 0.55), "front": (0.0, -1.0, 0.12), "top": (0.02, -0.02, 1.0)}}


def frame(rel: str, large: bool, direction=(0.75, -0.9, 0.55)):
    """Add the asset (once per asset), return (eye, target, scale of its bounding box)."""
    prim = stage.GetPrimAtPath("/World/asset")
    if not prim.IsValid():
        prim = stage.DefinePrim("/World/asset", "Xform")
        prim.GetReferences().AddReference(str(A / rel))
        _st = Usd.Stage.Open(str(A / rel))
        print("   metersPerUnit", UsdGeom.GetStageMetersPerUnit(_st), "upAxis", UsdGeom.GetStageUpAxis(_st), flush=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    ctr, size = (lo + hi) / 2, hi - lo
    if large:  # stand inside the room near one end, look along the long axis
        ax = int(np.argmax(size[:2]))
        eye = ctr.copy(); eye[ax] = lo[ax] + 0.15 * size[ax]; eye[2] = lo[2] + 1.7
        tgt = ctr.copy(); tgt[2] = lo[2] + 1.0
        return eye, tgt, size
    d = 1.15 * np.linalg.norm(size) / (2 * np.tan(np.radians(20)))
    v = np.array(direction, float)
    return ctr + d * v / np.linalg.norm(v), ctr, size


def look_at(eye, tgt):
    """Set the camera pose directly on the USD prim (z up, camera looks down its -z axis)."""
    m = Gf.Matrix4d(1).SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1)).GetInverse()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(CAM))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def render(eye, tgt):
    look_at(eye, tgt)
    for i in range(args.steps):  # let textures and materials stream in
        app.update()
    img = rgb.get_data()
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


args.out.mkdir(parents=True, exist_ok=True)
for group, items in GROUPS.items():
    if args.only and group not in args.only.split(","):
        continue
    tiles = []
    for rel in items:
        if not (A / rel).exists():
            print("missing", rel); continue
        print("loading", rel, flush=True)
        for vname, vdir in MULTI.get(group, {"iso": (0.75, -0.9, 0.55)}).items():
            eye, tgt, size = frame(rel, group in LARGE, vdir)
            img = render(eye, tgt)
            stage.RemovePrim("/World/asset") if vname == list(MULTI.get(group, {"iso": 0}))[-1] else None
            label = f"{Path(rel).stem} {vname}  [{size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m]"
            cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
            print(group, label, "mean", img.mean().round(1))
    cols = 4 if group in ("pullbins2", "ycb", "other_objects", "containers") else 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(args.out / f"sheet_{group}.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print("wrote", args.out / f"sheet_{group}.png")
app.close()
