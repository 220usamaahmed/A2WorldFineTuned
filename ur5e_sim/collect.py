"""Scripted UR5e pick-and-place episodes (successes and failures) with 3 cameras, saved per episode.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python collect.py --headless \
        --plan success:3,drop_on_table:2,release_off_edge:2,push_off_edge:2,miss_grasp:2 --out data/pilot

Episode kinds
  success           pick the cube, carry it to a target on the table, release it there
  drop_on_table     the gripper opens while carrying: the cube drops onto the table
  release_off_edge  the cube is carried over the table edge and released: it falls to the floor
  push_off_edge     closed gripper sweeps the cube across the table edge: it falls to the floor
  miss_grasp        the gripper closes beside the cube and drags / knocks it, then lifts

Per episode (T control steps, T+1 observations) a folder with ext1/ext2/wrist .mp4 and ``data.npz``:
  rgb_ext1 rgb_ext2 rgb_wrist [T+1,256,256,3] uint8 | joint_pos joint_vel [T+1,6] | finger_pos [T+1,2] gripper01 [T+1]
  tcp_pos [T+1,3] tcp_quat [T+1,4 wxyz] (arm-base frame, TCP = pad midpoint) | cube_pos [T+1,3] cube_quat [T+1,4]
  action_tcp [T,7] action_grip [T] (commands applied after observation t) | phase [T] | meta (json string)
"""
import argparse
import json
import math
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--plan", default="success:1", help="kind:count,kind:count,...")
parser.add_argument("--out", default="data/pilot")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--first-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import ur5e_env as E  # noqa: E402
import robot_defs as n  # noqa: E402

KINDS = ("success", "drop_on_table", "release_off_edge", "push_off_edge", "miss_grasp")
HOME = (0.40, 0.0, 0.25)
Z_GRASP, Z_HOVER, Z_CARRY = 0.030, 0.18, 0.20  # TCP heights above the table top [m]
MAX_STEP = 0.012  # TCP speed limit [m per control step] (LIBERO demos move ~0.009 m per step on average)
SLOW_STEP = 0.008  # while a cube hangs in the gripper (a faster carry shakes it out of the fingers)
PHASES = ["warmup", "move", "descend", "close", "hold", "lift", "carry", "release", "retreat", "push", "idle"]


# ---------------------------------------------------------------------------------------------------
# scripted programs: lists of segments (target TCP position, yaw, finger command, number of steps, phase)
# ---------------------------------------------------------------------------------------------------
def seg(pos, yaw, grip, phase, steps=None, prev=None):
    if steps is None:
        dist = np.linalg.norm(np.asarray(pos) - np.asarray(prev)) if prev is not None else 0.0
        steps = max(6, math.ceil(dist / (SLOW_STEP if phase in ("lift", "carry") else MAX_STEP)))
    return dict(pos=tuple(float(v) for v in pos), yaw=float(yaw), grip=float(grip), phase=phase, steps=int(steps))


class Program:
    def __init__(self):
        self.segs, self._pos, self._yaw, self._grip = [], HOME, 0.0, E.GRIPPER_OPEN

    def go(self, pos, phase, yaw=None, grip=None, steps=None):
        yaw = self._yaw if yaw is None else yaw
        grip = self._grip if grip is None else grip
        self.segs.append(seg(pos, yaw, grip, phase, steps, prev=self._pos))
        self._pos, self._yaw, self._grip = tuple(pos), yaw, grip

    def wait(self, steps, phase, grip=None):
        self.go(self._pos, phase, grip=grip, steps=steps)


def sample_scene(rng, kind):
    while True:
        cx, cy = rng.uniform(0.38, 0.62), rng.uniform(-0.18, 0.18)
        if np.hypot(cx - E.READY_TCP[0], cy) > 0.13:
            break
    yaw = float(rng.uniform(-0.5, 0.5))
    if kind == "push_off_edge":  # near one side edge, to be swept across it
        side = float(rng.choice([-1.0, 1.0]))
        cy, yaw = side * float(rng.uniform(0.12, 0.20)), 0.0
    return dict(cube_xy=(float(cx), float(cy)), cube_yaw=yaw)


def build_program(kind, rng, cube=None):
    """cube = (x, y, yaw) measured after the cube has settled; the random draws are the same with or without it."""
    sc = sample_scene(rng, kind)
    if cube is not None:
        sc = dict(cube_xy=(float(cube[0]), float(cube[1])), cube_yaw=float(cube[2]))
    cx, cy = sc["cube_xy"]
    yaw = sc["cube_yaw"]
    p = Program()
    info = dict(sc, kind=kind)

    def pick(offset_y=0.0):
        gx, gy = cx - math.sin(yaw) * offset_y, cy + math.cos(yaw) * offset_y  # offset along the closing axis
        p.go((gx, gy, Z_HOVER), "move", yaw=yaw)
        p.go((gx, gy, Z_GRASP), "descend")
        p.go((gx, gy, Z_GRASP), "close", grip=E.GRIPPER_CLOSED, steps=10)
        p.wait(3, "hold")
        p.go((gx, gy, Z_CARRY), "lift")
        p.wait(3, "hold")

    if kind in ("success", "drop_on_table"):
        while True:
            tx, ty = rng.uniform(0.38, 0.66), rng.uniform(-0.24, 0.24)
            if np.hypot(tx - cx, ty - cy) > 0.20 and np.hypot(tx - E.READY_TCP[0], ty) > 0.13:
                break
        info["target_xy"] = (float(tx), float(ty))
        pick()
        if kind == "success":
            p.go((tx, ty, Z_CARRY), "carry")
            p.go((tx, ty, Z_GRASP + 0.004), "descend")
            p.go((tx, ty, Z_GRASP + 0.004), "release", grip=E.GRIPPER_OPEN, steps=6)
            p.wait(2, "hold")
            p.go((tx, ty, Z_CARRY), "retreat")
            p.wait(4, "idle")
        else:  # open the gripper half way, still 10 cm above the table
            mx, my = 0.5 * (cx + tx), 0.5 * (cy + ty)
            p.go((mx, my, Z_CARRY), "carry")
            p.go((mx, my, 0.12), "carry")
            p.go((mx, my, 0.12), "release", grip=E.GRIPPER_OPEN, steps=3)
            info["release_xy"] = (float(mx), float(my))
            p.go((tx, ty, Z_CARRY), "retreat")  # the arm carries on as if nothing happened
            p.wait(6, "idle")
    elif kind == "release_off_edge":
        side = float(rng.choice([-1.0, 1.0]))
        tx, ty = rng.uniform(0.40, 0.58), side * 0.40  # cube centre 5 cm beyond the table edge (+-0.35)
        info["target_xy"] = (float(tx), float(ty))
        pick()
        p.go((tx, ty, Z_CARRY), "carry")
        p.go((tx, ty, 0.08), "descend")
        p.go((tx, ty, 0.08), "release", grip=E.GRIPPER_OPEN, steps=4)
        p.go((tx, ty, Z_CARRY), "retreat")
        p.wait(8, "idle")
    elif kind == "push_off_edge":
        side = float(np.sign(cy))
        p._grip = E.GRIPPER_CLOSED
        p.go((cx, cy - side * 0.085, Z_HOVER), "move", yaw=0.0, grip=E.GRIPPER_CLOSED)
        p.go((cx, cy - side * 0.085, 0.035), "descend")
        p.go((cx, side * 0.46, 0.035), "push")
        p.go((cx, side * 0.46, Z_HOVER), "retreat")
        p.wait(8, "idle")
    elif kind == "miss_grasp":
        side = float(rng.choice([-1.0, 1.0]))
        pick(offset_y=side * 0.07)
        p.wait(6, "idle")
    else:
        raise ValueError(kind)
    return info, p.segs


# ---------------------------------------------------------------------------------------------------
# simulation helpers
# ---------------------------------------------------------------------------------------------------
def quat_wxyz(yaw: float) -> np.ndarray:
    """Tool pointing straight down, rotated by ``yaw`` about the vertical."""
    r = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", math.pi)
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


class Sim:
    def __init__(self):
        self.env = ManagerBasedRLEnv(E.Ur5eTableEnvCfg())
        self.robot = self.env.scene["robot"]
        self.cube = self.env.scene["cube"]
        self.dev = self.env.device
        self.tool = self.robot.find_bodies(n.EE_LINK)[0][0]
        self.arm_ids = self.robot.find_joints(n.ARM_JOINTS, preserve_order=True)[0]
        self.finger_ids = self.robot.find_joints(list(E.GRIPPER_DRIVEN), preserve_order=True)[0]

    def reset(self, cube_xy, cube_yaw):
        # the cube is born at its episode pose: teleporting it after the reset leaves a faint temporal-render ghost of
        # it at the default position in the first frames (which would be the model's conditioning frame)
        pose = torch.tensor(
            [cube_xy[0], cube_xy[1], E.TABLE_Z + E.CUBE_Z + 0.003, math.cos(cube_yaw / 2), 0.0, 0.0, math.sin(cube_yaw / 2)],
            dtype=torch.float32, device=self.dev,
        )
        self.cube.data.default_root_state[:, :7] = pose
        self.cube.data.default_root_state[:, 7:] = 0.0
        self.env.reset()

    def step(self, tcp_pos, yaw, grip):
        act = torch.tensor(
            [[*tcp_pos, *quat_wxyz(yaw), grip]], dtype=torch.float32, device=self.dev
        )
        self.env.step(act)
        return act

    def state(self):
        d = self.robot.data
        root = d.root_pos_w[0]
        off = torch.tensor([0.0, 0.0, n.TCP_OFFSET_Z], device=self.dev)
        tcp = d.body_pos_w[0, self.tool] + math_utils.quat_apply(d.body_quat_w[0, self.tool], off) - root
        c = self.cube.data
        out = dict(
            joint_pos=d.joint_pos[0, self.arm_ids], joint_vel=d.joint_vel[0, self.arm_ids],
            finger_pos=d.joint_pos[0, self.finger_ids],
            tcp_pos=tcp, tcp_quat=d.body_quat_w[0, self.tool],  # tool0 orientation = TCP orientation
            cube_pos=c.root_pos_w[0] - root, cube_quat=c.root_quat_w[0],
        )
        out = {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}
        out.update({f"rgb_{c}": self.env.scene[f"cam_{c}"].data.output["rgb"][0, :, :, :3].cpu().numpy()
                    for c in ("ext1", "ext2", "wrist")})
        return out


def run_episode(sim: Sim, kind: str, seed: int):
    sampled, _ = build_program(kind, np.random.default_rng(seed))
    sim.reset(sampled["cube_xy"], sampled["cube_yaw"])

    # warm-up (not recorded): rise from the ready pose to HOME, cube settles on the table
    cur_pos, cur_yaw, cur_grip = np.array(E.READY_TCP), 0.0, 0.0
    for i in range(1, 41):
        s = i / 40
        sim.step(cur_pos + (np.array(HOME) - cur_pos) * (s * s * (3 - 2 * s)), 0.0, 0.0)
    cur_pos = np.array(HOME)

    # the cube has settled (it slides a few mm when it lands): plan from where it really is
    s0 = sim.state()
    qw, qx, qy, qz = s0["cube_quat"]
    settled = (float(s0["cube_pos"][0]), float(s0["cube_pos"][1]), float(Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2]))
    info, segs = build_program(kind, np.random.default_rng(seed), cube=settled)
    info["cube_xy_sampled"] = sampled["cube_xy"]  # info["cube_xy"] is the settled position the plan uses

    obs = [s0]
    acts, grips, phases = [], [], []
    for sg in segs:
        tgt = np.array(sg["pos"])
        p0, y0, g0 = cur_pos.copy(), cur_yaw, cur_grip
        for i in range(1, sg["steps"] + 1):
            s = i / sg["steps"]
            e = s * s * (3 - 2 * s)
            pos, yaw = p0 + (tgt - p0) * e, y0 + (sg["yaw"] - y0) * e
            grip = g0 + (sg["grip"] - g0) * s
            sim.step(pos, yaw, grip)
            obs.append(sim.state())
            acts.append(np.concatenate([pos, quat_wxyz(yaw)]).astype(np.float32))
            grips.append(grip)
            phases.append(PHASES.index(sg["phase"]))
        cur_pos, cur_yaw, cur_grip = tgt, sg["yaw"], sg["grip"]

    data = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
    data.update(action_tcp=np.stack(acts), action_grip=np.array(grips, np.float32), phase=np.array(phases, np.int16))

    cube = data["cube_pos"]
    fell = np.where(cube[:, 2] < -0.04)[0]
    end = cube[-1]
    on_table = bool(end[2] > 0.0 and abs(end[1]) < E.TABLE_HALF_Y and E.TABLE_X[0] < end[0] < E.TABLE_X[1])
    lifted = float(cube[:, 2].max() - E.CUBE_Z)
    out = dict(info, steps=len(acts), cube_end=[float(v) for v in end], on_table=on_table,
               fell_step=int(fell[0]) if len(fell) else None, max_cube_lift=lifted)
    if "target_xy" in info:
        out["dist_to_target"] = float(np.hypot(end[0] - info["target_xy"][0], end[1] - info["target_xy"][1]))
    if kind == "success":
        out["outcome"] = "placed" if on_table and out["dist_to_target"] < 0.04 and end[2] < 0.04 else "unexpected"
    elif kind == "drop_on_table":
        out["outcome"] = "dropped_on_table" if on_table and lifted > 0.02 else "unexpected"
    elif kind in ("release_off_edge", "push_off_edge"):
        out["outcome"] = "fell_off_table" if not on_table else "unexpected"
    else:  # miss_grasp: expect the cube NOT to be carried away
        out["outcome"] = "missed" if lifted < 0.03 else "unexpected"
    return data, out


def save_episode(folder: Path, data: dict, meta: dict):
    folder.mkdir(parents=True, exist_ok=True)
    for c in ("ext1", "ext2", "wrist"):
        imageio.mimwrite(folder / f"{c}.mp4", list(data[f"rgb_{c}"]), fps=10, macro_block_size=1)
    np.savez_compressed(folder / "data.npz", meta=json.dumps(meta), **data)


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in args.plan.split(","):
        kind, count = item.split(":")
        assert kind in KINDS, kind
        jobs += [kind] * int(count)
    sim = Sim()
    rows = []
    for i, kind in enumerate(jobs, start=args.first_index):
        data, meta = run_episode(sim, kind, args.seed * 100003 + i)
        meta["index"] = i
        save_episode(out / f"ep_{i:03d}_{kind}", data, meta)
        rows.append(meta)
        print(f"ep {i:03d} {kind:17s} steps={meta['steps']:3d} outcome={meta['outcome']:17s} on_table={meta['on_table']} "
              f"lift={meta['max_cube_lift']:.3f} fell_step={meta['fell_step']} cube_end={np.round(meta['cube_end'], 3).tolist()}")
    (out / f"summary_{args.first_index:03d}.json").write_text(json.dumps(rows, indent=1))
    sim.env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
