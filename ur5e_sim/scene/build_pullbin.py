"""Generate pullbin.usd: a wooden housing with one Isaac Sim produce crate that slides out of it (prototype of a drawer).

    OMNI_KIT_ACCEPT_EULA=YES python build_pullbin.py --headless [--out ../assets/usd/pullbin.usd]

Why a generated asset: the crate has no physics of its own and a convex hull of its mesh would be a solid block, so the
visual is the real mesh (SM_Crate_A08_Blue_01) while the collision is a set of invisible boxes that keep the inside
open and the end wall's rim bar (the handle) at its true place.

Frames. Housing frame: origin on the floor below the centre of the housing footprint, x = the direction the crate is
pulled out ("front"), z up. The crate frame is the mesh frame: origin at the centre of the crate's bottom, x along its
long axis. The crate's end wall (x = +0.29) faces the front. Joint `crate_slide` (prismatic, along +x) moves the crate out.

Crate measurements (from the mesh, metres): outer 0.605 x 0.405 x 0.170; end wall = 3 mm plate (inner face x = 0.290,
outer 0.293) with a 26 mm tall rim bar on top (z 0.144..0.170) whose outer face is at x = 0.3026; the floor is 6 mm thick.
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402  (plain numbers, no Isaac imports)

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=HERE.parent / "assets" / "usd" / "pullbin.usd")
ap.add_argument("--assets", type=Path, default=Path("/home/data/siddiquieu1/assets/Isaac/5.1/Isaac"))
ap.add_argument("--base-z", type=float, default=0.75, help="height of the housing's base plate (top face) above the floor [m]; 0.75 = table top")
ap.add_argument("--wall-h", type=float, default=0.25, help="side wall height above the base plate = clearance under the roof [m]; the crate itself is 0.17")
ap.add_argument("--window", type=float, default=layout.WINDOW, help="depth of the roof opening at the front, over the crate [m] (default: layout.WINDOW, which the planner uses)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

CRATE_USD = args.assets / "Props/PackingTable/props/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01.usd"
WOOD = (0.45, 0.30, 0.18)  # same brown as the table slab in ur5e_env.py; change both together

# ---- crate (mesh frame) ----
CL, CW, CH = 0.6052, 0.4052, 0.1703  # outer length (x), width (y), height (z)
WALL_IN_X = 0.290  # inner face of the end walls
SIDE_IN_Y = 0.1965  # inner face of the long side walls
FLOOR_T = 0.006
BAR_Z0 = 0.144  # rim bar of the end walls
BAR_OUT_X = 0.3026
CRATE_MASS = 1.2

# ---- housing (housing frame) ----
T = 0.02  # panel thickness
IN_HALF_W = 0.215  # inner half-width (crate half-width 0.2026 + 12 mm clearance)
BACK_X = -0.315  # inner face of the back wall
FRONT_X = 0.32  # front edge of the housing
WALL_H = args.wall_h  # side wall height above the base plate (items stand up to ~0.2 m tall in the crate)
ROOF_X = (BACK_X - T, FRONT_X - args.window)  # roof over everything except a window at the front, for reaching in from above
CRATE_Z0 = 0.002  # crate bottom above the base plate


def material(stage, name, rgb, rough=0.7):
    m = UsdShade.Material.Define(stage, f"/PullBin/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/PullBin/Looks/{name}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def box(stage, path, lo, hi, visible=True, collide=True, mat=None):
    """Axis-aligned box from corner lo to corner hi (parent frame) as a scaled unit cube."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not visible:
        UsdGeom.Imageable(cube).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    elif mat is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube


import numpy as np  # noqa: E402

args.out.parent.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.CreateNew(str(args.out))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/PullBin")
stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
wood = material(stage, "wood", WOOD, 0.8)

# ---- housing: plinth + base plate + side walls + back wall + roof over the rear part ----
Z_B = args.base_z
hous = UsdGeom.Xform.Define(stage, "/PullBin/housing")
UsdPhysics.RigidBodyAPI.Apply(hous.GetPrim())
UsdPhysics.MassAPI.Apply(hous.GetPrim()).CreateMassAttr(60.0)
W = IN_HALF_W + T
box(stage, "/PullBin/housing/plinth", (BACK_X - T, -W, 0.0), (FRONT_X, W, Z_B - T), mat=wood)
box(stage, "/PullBin/housing/base", (BACK_X - T, -W, Z_B - T), (FRONT_X, W, Z_B), mat=wood)
box(stage, "/PullBin/housing/side_l", (BACK_X - T, IN_HALF_W, Z_B), (FRONT_X, W, Z_B + WALL_H), mat=wood)
box(stage, "/PullBin/housing/side_r", (BACK_X - T, -W, Z_B), (FRONT_X, -IN_HALF_W, Z_B + WALL_H), mat=wood)
box(stage, "/PullBin/housing/back", (BACK_X - T, -IN_HALF_W, Z_B), (BACK_X, IN_HALF_W, Z_B + WALL_H), mat=wood)
box(stage, "/PullBin/housing/roof", (ROOF_X[0], -W, Z_B + WALL_H), (ROOF_X[1], W, Z_B + WALL_H + T), mat=wood)

# ---- crate: real mesh as the visual, invisible boxes as the collision ----
crate = UsdGeom.Xform.Define(stage, "/PullBin/crate")
UsdGeom.Xformable(crate).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, Z_B + CRATE_Z0))
UsdPhysics.RigidBodyAPI.Apply(crate.GetPrim())
UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(CRATE_MASS)
vis = UsdGeom.Xform.Define(stage, "/PullBin/crate/visual")
UsdGeom.Xformable(vis).AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))  # the crate file is in centimetres
vis.GetPrim().GetReferences().AddReference(Sdf.Reference(str(CRATE_USD)))
box(stage, "/PullBin/crate/col_floor", (-CL / 2, -CW / 2, 0.0), (CL / 2, CW / 2, FLOOR_T), visible=False)
for sx, nm in ((+1, "front"), (-1, "rear")):  # end walls: thin plate + the rim bar on the outside
    x_in, x_out = sorted((sx * WALL_IN_X, sx * (WALL_IN_X + 0.0075)))
    box(stage, f"/PullBin/crate/col_{nm}_wall", (x_in, -CW / 2, 0.0), (x_out, CW / 2, CH), visible=False)
    b_in, b_out = sorted((sx * (WALL_IN_X + 0.0075), sx * BAR_OUT_X))
    box(stage, f"/PullBin/crate/col_{nm}_bar", (b_in, -CW / 2, BAR_Z0), (b_out, CW / 2, CH), visible=False)
for sy, nm in ((+1, "l"), (-1, "r")):
    y_in, y_out = sorted((sy * SIDE_IN_Y, sy * CW / 2))
    box(stage, f"/PullBin/crate/col_side_{nm}", (-CL / 2, y_in, 0.0), (CL / 2, y_out, CH), visible=False)

# ---- joints ----
fixed = UsdPhysics.FixedJoint.Define(stage, "/PullBin/rootJoint")
fixed.CreateBody1Rel().SetTargets([Sdf.Path("/PullBin/housing")])
slide = UsdPhysics.PrismaticJoint.Define(stage, "/PullBin/crate_slide")
slide.CreateBody0Rel().SetTargets([Sdf.Path("/PullBin/housing")])
slide.CreateBody1Rel().SetTargets([Sdf.Path("/PullBin/crate")])
slide.CreateAxisAttr("X")
slide.CreateLowerLimitAttr(0.0)
slide.CreateUpperLimitAttr(0.35)
# a linear drive must exist for the simulator's gains to apply (Isaac Lab only overwrites an existing drive); without it
# the crate slides with no resistance at all. Stiffness 0 + damping: a pure damper, tuned at run time.
drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
drive.CreateTypeAttr("force")
drive.CreateStiffnessAttr(0.0)
drive.CreateDampingAttr(100.0)
drive.CreateMaxForceAttr(500.0)
drive.CreateTargetPositionAttr(0.0)
slide.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, Z_B + CRATE_Z0))  # crate origin, in the housing frame, when closed
slide.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
slide.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
slide.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
stage.GetRootLayer().Save()
print("wrote", args.out)
app.close()
