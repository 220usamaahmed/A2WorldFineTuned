"""Look of the pick-and-place scene: furniture materials, lights and the backdrop room. Visual only (nothing here
collides or changes what the planner sees), applied by pick_place.py.

    python looks.py        # download the material and light map below (no Isaac Sim needed)

- Wood (NVIDIA's Base/Wood/Walnut, world-space projected so the grain has the same size on every board): the surfaces
  things stand on: the desk top, the two shelf boards and the counter top (drawer housing roofs + counter end tops).
- Paint (flat satin white): everything else of the furniture: desk body, shelf plinth/sides/back, counter body (drawer
  housings below the roof, counter end bodies).
- Bar code scanner on the desk (layout.SCANNER_*): brushed-metal frame around a dark glass window, visual only.
- Lights: the photo-studio light map on a dome (hidden from the cameras, so it only lights) plus a directional key light
  for shadows, both warm (colour temperature).
- Backdrop: an open-top room with a polished concrete floor (NVIDIA's Base/Masonry/Concrete_Polished), painted drywall
  walls (Base/Wall_Board/Plaster: even colour, fine surface texture) and a baseboard; the shelf stands close to the +y wall. The walls cast no shadows and do not block the dome, so they only
  fill the background. The default grid ground plane stays for physics but is hidden.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ur5e_sim/
import layout as L  # noqa: E402
import paths  # noqa: E402

NVIDIA = paths.NVIDIA
BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA"
WOOD_MDL = NVIDIA / "Materials/Base/Wood/Walnut.mdl"  # Oak.mdl (light, orange) is fetched too
WALL_MDL = NVIDIA / "Materials/Base/Wall_Board/Plaster.mdl"
FLOOR_MDL = NVIDIA / "Materials/Base/Masonry/Concrete_Polished.mdl"  # Concrete_Smooth.mdl (matte, almost uniform) is fetched too
STUDIO_HDR = NVIDIA / "Assets/Skies/Studio/photo_studio_01_4k.hdr"
FILES = ["Assets/Skies/Studio/photo_studio_01_4k.hdr"] + \
        [f"Materials/Base/{d}/{m}{f}" for d, m, n in (("Wood", "Oak", "N"), ("Wood", "Walnut", "N"), ("Wall_Board", "Plaster", "N"),
                                                      ("Masonry", "Concrete_Smooth", "Normal"), ("Masonry", "Concrete_Polished", "Normal"))
         for f in (".mdl", f"/{m}_BaseColor.png", f"/{m}_ORM.png", f"/{m}_{n}.png")]

WOOD_SCALE = 1.0  # texture_scale of the wood (world-space projection; larger = bigger grain)
FLOOR_SCALE = 2.0  # texture_scale of the concrete
PAINT = (0.80, 0.80, 0.78)  # counter body, linear RGB (white)
PAINT_ROUGH = 0.45
WALL_TINT, WALL_BRIGHTNESS = (1.0, 0.98, 0.94), 1.55  # the plaster texture is an even mid grey: tint and brighten it
BASEBOARD = (0.85, 0.85, 0.83)  # semi-gloss white
SCANNER_METAL = (0.70, 0.70, 0.70)  # metallic, brushed (roughness 0.3)
SCANNER_GLASS = (0.010, 0.012, 0.015)  # near black, glossy
BASEBOARD_H, BASEBOARD_T = 0.10, 0.015
DOME_INTENSITY, DOME_KELVIN = 250.0, 4500.0
SUN_INTENSITY, SUN_KELVIN = 450.0, 3800.0
SUN_TILT, SUN_AZIMUTH = 35.0, -60.0  # degrees from vertical; direction the light comes from (0 = +x, -90 = -y customer side)
BACK_GAP = 0.03  # shelf back to the +y wall (the baseboard takes BASEBOARD_T of it)
ROOM_X, ROOM_H = (-3.6, 3.6), 4.0  # inner faces of the +-x walls, wall height
ROOM_Y = (-3.95, L.SHELF_BOXES["back"][1][1] + BACK_GAP)

# prims under the env namespace (names from pick_place.SceneCfg and pullbin.usd)
WOOD_PRIMS = ["desk_top", "shelf_board0", "shelf_board1", "filler_top_l", "filler_top_r"] + [f"bin{i}/housing/roof" for i in range(3)]
PAINT_PRIMS = ["desk_body", "shelf_plinth", "shelf_side_l", "shelf_side_r", "shelf_back", "filler_end_l", "filler_end_r"] + \
              [f"bin{i}/housing/{p}" for i in range(3) for p in ("plinth", "base", "side_l", "side_r", "back")]


def fetch():
    for f in FILES:
        dst = NVIDIA / f
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(urllib.request.urlopen(f"{BUCKET}/{f}", timeout=120).read())
        print("fetched", f)


def sun_quat():
    """(w, x, y, z) turning the distant light's -z (its shining direction) to point down from SUN_TILT / SUN_AZIMUTH."""
    from scipy.spatial.transform import Rotation as R
    x, y, z, w = (R.from_euler("z", SUN_AZIMUTH, degrees=True) * R.from_euler("y", SUN_TILT, degrees=True)).as_quat()
    return (float(w), float(x), float(y), float(z))


def light_cfgs():
    """AssetBaseCfg for the dome and the key light (scene attributes `sky` and `sun`)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(
        intensity=DOME_INTENSITY, texture_file=str(STUDIO_HDR), visible_in_primary_ray=False,
        enable_color_temperature=True, color_temperature=DOME_KELVIN))
    sun = AssetBaseCfg(prim_path="/World/sun", spawn=sim_utils.DistantLightCfg(
        intensity=SUN_INTENSITY, angle=2.0, enable_color_temperature=True, color_temperature=SUN_KELVIN),
                       init_state=AssetBaseCfg.InitialStateCfg(rot=sun_quat()))
    return sky, sun


def _box(stage, path, lo, hi, mat, shadows=False):
    from pxr import Gf, Sdf, UsdGeom

    import isaaclab.sim as sim_utils

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
    xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo)))
    if not shadows:
        UsdGeom.PrimvarsAPI(cube).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
    sim_utils.bind_visual_material(path, mat, stage=stage)


def apply(stage, ns: str = "/World/envs/env_0"):
    """Create the materials and the backdrop room and bind the materials to the furniture (call after the scene exists)."""
    from pxr import Gf, Sdf

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

    looks = "/World/Looks"
    for nm, mdl, scale, extra in (("wood", WOOD_MDL, WOOD_SCALE, {}), ("floor", FLOOR_MDL, FLOOR_SCALE, {}),
                                  ("wall", WALL_MDL, 1.0, {"diffuse_tint": WALL_TINT, "albedo_brightness": WALL_BRIGHTNESS})):
        cfg = sim_utils.MdlFileCfg(mdl_path=str(mdl), project_uvw=True, texture_scale=(scale, scale))
        shader = cfg.func(f"{looks}/{nm}", cfg)
        for k, v in {"world_or_object": True, **extra}.items():
            if isinstance(v, tuple):  # colours must be typed color3f for the MDL parameter to bind
                shader.CreateAttribute(f"inputs:{k}", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
            else:
                safe_set_attribute_on_usd_prim(shader, f"inputs:{k}", v, camel_case=False)
    for nm, rgb, rough, metal in (("paint", PAINT, PAINT_ROUGH, 0.0), ("baseboard", BASEBOARD, 0.35, 0.0),
                                  ("scanner_metal", SCANNER_METAL, 0.3, 1.0), ("scanner_glass", SCANNER_GLASS, 0.05, 0.0)):
        c = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough, metallic=metal)
        c.func(f"{looks}/{nm}", c)
    for mat, prims in (("wood", WOOD_PRIMS), ("paint", PAINT_PRIMS)):
        for p in prims:
            if not stage.GetPrimAtPath(f"{ns}/{p}").IsValid():
                raise ValueError(f"looks: no prim {ns}/{p}")
            sim_utils.bind_visual_material(f"{ns}/{p}", f"{looks}/{mat}", stage=stage)
    (x0, x1), (y0, y1), t = ROOM_X, ROOM_Y, 0.05
    _box(stage, "/World/backdrop/floor", (x0 - t, y0 - t, -0.02), (x1 + t, y1 + t, 0.0), f"{looks}/floor")
    _box(stage, "/World/backdrop/wall_px", (x1, y0 - t, 0.0), (x1 + t, y1 + t, ROOM_H), f"{looks}/wall")
    _box(stage, "/World/backdrop/wall_nx", (x0 - t, y0 - t, 0.0), (x0, y1 + t, ROOM_H), f"{looks}/wall")
    _box(stage, "/World/backdrop/wall_py", (x0, y1, 0.0), (x1, y1 + t, ROOM_H), f"{looks}/wall")
    _box(stage, "/World/backdrop/wall_ny", (x0, y0 - t, 0.0), (x1, y0, ROOM_H), f"{looks}/wall")
    b, h = BASEBOARD_T, BASEBOARD_H
    _box(stage, "/World/backdrop/base_px", (x1 - b, y0, 0.0), (x1, y1, h), f"{looks}/baseboard")
    _box(stage, "/World/backdrop/base_nx", (x0, y0, 0.0), (x0 + b, y1, h), f"{looks}/baseboard")
    _box(stage, "/World/backdrop/base_py", (x0, y1 - b, 0.0), (x1, y1, h), f"{looks}/baseboard")
    _box(stage, "/World/backdrop/base_ny", (x0, y0, 0.0), (x1, y0 + b, h), f"{looks}/baseboard")
    # bar code scanner: four frame bars around the glass
    (cx, cy), (sx, sy), z0, h, r = L.SCANNER_XY, L.SCANNER_SIZE, L.TABLE_Z, L.SCANNER_H, L.SCANNER_RIM
    xa, xb, ya, yb = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
    for nm, lo, hi in (("rim_nx", (xa, ya, z0), (xa + r, yb, z0 + h)), ("rim_px", (xb - r, ya, z0), (xb, yb, z0 + h)),
                       ("rim_ny", (xa + r, ya, z0), (xb - r, ya + r, z0 + h)), ("rim_py", (xa + r, yb - r, z0), (xb - r, yb, z0 + h))):
        _box(stage, f"/World/scanner/{nm}", lo, hi, f"{looks}/scanner_metal", shadows=True)
    _box(stage, "/World/scanner/glass", (xa + r, ya + r, z0), (xb - r, yb - r, z0 + h - 0.001), f"{looks}/scanner_glass", shadows=True)


if __name__ == "__main__":
    fetch()
