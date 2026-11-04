"""Smoke test of the cuRobo planner on the side-bins world, without Isaac Sim.

    CUDA_VISIBLE_DEVICES=0 python test_planner.py

Checks: the ready and home configurations are collision-free; free-space plans to tool-down poses above the table;
top-down grasp plans (free motion to a pregrasp + straight 10 cm descent) at 3 positions in each of the 3 open crates,
from both start configurations; straightness of the descent; planning time.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scene"))
import layout as L  # noqa: E402
import robot_defs as n  # noqa: E402
from planner import ArmPlanner  # noqa: E402

DOWN = (0.0, 1.0, 0.0, 0.0)  # tool pointing down, fingers closing along base x
SLIDE = 0.15

t0 = time.time()
world = L.world_config(slides=(SLIDE, SLIDE, SLIDE))
pl = ArmPlanner(world, control_dt=0.1, time_dilation=0.5)
print(f"planner ready in {time.time() - t0:.1f} s ({len(world.cuboid)} cuboids)")
starts = {"ready": np.array(n.ARM_READY), "home": pl.home}
for k, q in starts.items():
    print(f"{k:5s} q {np.round(q, 3)} TCP {np.round(pl.fk(q)[:3], 3)} valid {pl.mg.check_start_state(pl._js(q))[0]}")

fails = 0
for x, y in [(0.32, 0.14), (0.55, 0.2), (0.45, -0.05)]:
    p = np.r_[x, y, 0.05, DOWN]
    t = time.time()
    plan = pl.plan_pose(pl.home, p, tag=f"table ({x}, {y})")
    if plan is None:
        fails += 1
        continue
    err = np.linalg.norm(pl.fk(plan.q[-1])[:3] - p[:3])
    print(f"table ({x:.2f}, {y:+.2f}): {plan.duration:4.1f} s of motion, planned in {time.time() - t:4.2f} s, TCP error {err * 1000:.1f} mm")

times = []
for start, q0 in starts.items():
    for b in range(3):
        row = []
        for xc in (0.15, 0.10, 0.05):  # crate-local x: the roof window of a crate pulled out 0.15 m is x > -0.03
            g = np.r_[L.crate_to_world(b, xc, 0.0, L.FLOOR_TOP + 0.05, slide=SLIDE) - [0, 0, L.TABLE_Z], DOWN]
            t = time.time()
            to_pre, app, _ = pl.plan_grasp(q0, g[None], approach=0.10)
            times.append(time.time() - t)
            if app is None:
                fails += 1
                row.append(f"x {xc:+.2f}: FAIL")
                continue
            tcp = np.array([pl.fk(q)[:3] for q in app.q])
            dev = np.abs(tcp[:, :2] - g[:2]).max()
            row.append(f"x {xc:+.2f}: {to_pre.duration:.1f}+{app.duration:.1f} s, dev {dev * 1000:.1f} mm")
        print(f"grasp from {start:5s} bin{b}: " + " | ".join(row), flush=True)
print(f"grasp planning time: median {np.median(times):.2f} s, max {np.max(times):.2f} s")
print("ALL OK" if fails == 0 else f"{fails} FAILURES")
