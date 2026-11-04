"""Reachability of the shelf + container layout, planner only (no simulator).

    CUDA_VISIBLE_DEVICES=0 python test_shelf_container.py [--trials 2]

For each tall object standing upright at each nominal spot on both shelf levels (random yaw): sampled grasps at the
object's pose -> planner.plan_grasp; then, with the object attached, a free-space plan to 10 cm above a drop pose in the
container (same orientation as on the shelf) and a straight move down. Reports which spots / objects work.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scene"))
import layout as L  # noqa: E402
from grasps import T_from_pose7, cached_grasps, collider_mesh, graspable_stable_poses, world_filter  # noqa: E402
from planner import ArmPlanner  # noqa: E402
from tasks import drop_poses, plan_extraction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--levels", default="01")
args = ap.parse_args()
LEVELS = args.levels
rng = np.random.default_rng(args.seed)
MESHES = Path("/home/data/siddiquieu1/assets/ycb_meshes")
TALL = ["006_mustard_bottle", "004_sugar_box", "003_cracker_box", "021_bleach_cleanser"]
summ = json.load(open(MESHES / "summary.json"))
lib = {}
for k in TALL:
    m = collider_mesh(k, summ[k]["obj"])
    g = cached_grasps(k, summ[k]["obj"], n_pairs=500, n_roll=16)
    poses = graspable_stable_poses(m, g)
    h = [(T @ np.c_[m.vertices, np.ones(len(m.vertices))].T)[2].max() for T, _, _ in poses]
    lib[k] = (g, poses[int(np.argmax(h))][0], float(max(h)))  # the upright (tallest) resting pose
    print(f"{k}: upright height {max(h) * 100:.0f} cm, {len(g)} grasps", flush=True)


def pose7(T):
    x, y, z, w = R.from_matrix(T[:3, :3]).as_quat()
    return np.r_[T[:3, 3], w, x, y, z]


pl = ArmPlanner(L.world_config(shelf=True, container=True), time_dilation=0.5)
res = []
for trial in range(args.trials):
    for lvl, z_board in [(i, z) for i, z in enumerate(L.SHELF_LEVELS) if str(i) in LEVELS]:
        for sx in L.SHELF_SPOTS_X:
            for k in TALL:
                g, T_rest, h = lib[k]
                Rz = np.eye(4)
                Rz[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
                T = Rz @ T_rest
                T[:3, 3] += [sx, rng.uniform(*L.SHELF_SPOTS_Y), z_board + 0.002 - L.TABLE_Z]
                pl.update_world(L.world_config(shelf=True, container=True, objects=[("obj", MESHES / f"{k}.obj", pose7(T))]))
                Ts = T @ np.array([x.T for x in g])
                idx = np.where(world_filter(Ts, floor_z=z_board - L.TABLE_Z, min_clear=0.004, widths=[x.width for x in g]))[0]
                sc = np.array([g[i].score for i in idx])
                p = np.exp((sc - sc.max()) / 0.3)
                order = rng.choice(idx, size=min(40, len(idx)), replace=False, p=p / p.sum())
                cand = np.array([pose7(Ts[i]) for i in order])
                ext = {}

                def check(gi, to_pre, app):
                    e = plan_extraction(pl, app.q[-1], Ts[order[gi]], T, "obj", "shelf")
                    if e is not None:
                        ext[gi] = e
                    return e is not None

                t = time.time()
                to_pre, app, gi = pl.plan_grasp(pl.home, cand, target="obj", order="given", max_tries=10, check=check)
                rec = dict(level=lvl, x=sx, obj=k, grasp_ok=app is not None, t_grasp=round(time.time() - t, 1))
                if app is not None:
                    Tg = Ts[order[gi]]
                    rec["approach_deg"] = round(float(np.degrees(np.arccos(np.clip(-Tg[2, 2], -1, 1)))), 0)
                    lift, out = plan_extraction(pl, app.q[-1], Tg, T, "obj", "shelf", keep_attached=True)
                    foot = 0.5 * float(np.linalg.norm(np.ptp((T[:3, :3] @ collider_mesh(k, summ[k]["obj"]).vertices.T)[:2], axis=1)))
                    drops = drop_poses(np.linalg.inv(Tg) @ T, T, z_board - L.TABLE_Z, rng, footprint=foot + 0.01)
                    t = time.time()
                    to_drop, down, di = pl.plan_grasp(out.q[-1], drops, approach=0.10, approach_dir=(0, 0, -1), order="given", max_tries=12, tag="place")
                    rec.update(place_ok=down is not None, t_place=round(time.time() - t, 1))
                    if down is None:
                        rec["fail"] = "place"
                    pl.detach()
                res.append(rec)
                print(rec, flush=True)

print("\nSUMMARY (grasp planned / grasp + place planned)")
for lvl in (0, 1):
    for sx in L.SHELF_SPOTS_X:
        r = [x for x in res if x["level"] == lvl and x["x"] == sx]
        print(f"  level {lvl} x {sx:+.2f}: {sum(x['grasp_ok'] for x in r)}/{len(r)}  {sum(x.get('place_ok', False) for x in r)}/{len(r)}")
for k in TALL:
    r = [x for x in res if x["obj"] == k]
    print(f"  {k:22s}: {sum(x['grasp_ok'] for x in r)}/{len(r)}  {sum(x.get('place_ok', False) for x in r)}/{len(r)}")
