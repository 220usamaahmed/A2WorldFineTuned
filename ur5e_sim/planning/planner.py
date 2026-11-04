"""cuRobo motion planning for the UR5e + Robotiq 2F-85 (thin wrapper around curobo's MotionGen).

Frames: everything here is in the ARM-BASE frame (``ur5e_base_link``, the articulation root): world frame shifted down by
ur5e_env.TABLE_Z. Poses are (x, y, z, qw, qx, qy, qz) with the TCP (``grasp_frame``, between the finger pads) as the
end-effector. Trajectories come back as numpy arrays [T, 6] of arm joint positions (robot_defs.ARM_JOINTS order) spaced
by ``control_dt`` = the env's control step, ready to be fed one row per env.step to the JointTrajectoryAction.

Import after the Isaac app is launched if used inside Isaac Lab (torch / CUDA are shared with the simulator).
"""
from __future__ import annotations

import itertools
import os
import sys
from contextlib import contextmanager

# switching between goal types (single pose / goal set / joint goal) re-captures cuRobo's CUDA graphs; it refuses to
# unless this is set (needs torch built with CUDA >= 12, true here: 12.8)
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import robot_defs as n  # noqa: E402

ROBOT_YML = HERE / "ur5e_2f85.yml"
HOME_TCP = (0.35, 0.0, 0.35)  # arm-base frame, tool pointing down
GRIPPER_LINKS = [
    "robotiq_arg2f_base_link", "left_outer_knuckle", "right_outer_knuckle", "left_outer_finger", "right_outer_finger",
    "left_inner_knuckle", "right_inner_knuckle", "left_inner_finger", "right_inner_finger",
    "left_inner_finger_pad", "right_inner_finger_pad",
]
FINGER_LINKS = GRIPPER_LINKS[1:]


def pose7(pos, quat) -> np.ndarray:
    return np.r_[np.asarray(pos, float), np.asarray(quat, float)]


@dataclass
class Plan:
    q: np.ndarray  # [T, 6] joint positions at dt spacing (row 0 = start)
    dt: float
    status: str = ""

    @property
    def duration(self) -> float:
        return (len(self.q) - 1) * self.dt

    def __add__(self, other: "Plan") -> "Plan":
        return Plan(np.concatenate([self.q, other.q[1:]]), self.dt, f"{self.status}+{other.status}")


class ArmPlanner:
    def __init__(self, world: WorldConfig, control_dt: float = 0.1, time_dilation: float = 0.5,
                 device: str = "cuda:0", obb_cache: int = 80, mesh_cache: int = 12, warmup: bool = True,
                 collision_activation_distance: float = 0.01, log_level: str = "warning"):
        setup_curobo_logger(log_level)
        self.tdt = TensorDeviceType(device=torch.device(device))
        self.dt = control_dt
        # cuRobo's shortest allowed trajectory step is its interpolation dt (x 32 optimisation steps = the shortest
        # motion), so it interpolates finely and _to_plan resamples to the control step
        interpolation_dt = 0.02
        self.time_dilation = time_dilation
        robot_cfg = load_yaml(str(ROBOT_YML))["robot_cfg"]
        cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg, world, tensor_args=self.tdt,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": obb_cache, "mesh": mesh_cache},
            interpolation_dt=interpolation_dt,
            collision_activation_distance=collision_activation_distance,
            num_trajopt_seeds=12, num_graph_seeds=12, num_ik_seeds=64,
            use_cuda_graph=True,
        )
        self.mg = MotionGen(cfg)
        if warmup:
            self.mg.warmup(enable_graph=True, warmup_js_trajopt=False)
        # separate IK solvers for batched reachability checks and for Cartesian chains (one pose at a time), sharing the
        # motion generator's collision world: one solver switching between batch sizes re-captures its CUDA graphs
        # every time (seconds per grasp plan)
        self.ik_batch_size = 64
        ik_kw = dict(world_coll_checker=self.mg.world_coll_checker, tensor_args=self.tdt, use_cuda_graph=True,
                     collision_activation_distance=collision_activation_distance, self_collision_check=True)
        self.ik_many = IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg, None, num_seeds=32, **ik_kw))
        self.ik_one = IKSolver(IKSolverConfig.load_from_robot_config(robot_cfg, None, num_seeds=16, **ik_kw))
        self.joint_names = list(n.ARM_JOINTS)
        assert self.mg.joint_names == self.joint_names, f"cuRobo joint order {self.mg.joint_names}"
        self.attached: str | None = None
        self.verbose = False
        lim = self.mg.kinematics.get_joint_limits().position.cpu().numpy()  # [2, dof]
        self.joint_limits = (lim[0][:6], lim[1][:6])
        self.vel_limit = self.mg.kinematics.get_joint_limits().velocity.cpu().numpy()[1][:6]
        # home: tool down, high above the table centre. The via-point for reconfigurations (e.g. a wrist flip, which at
        # robot_defs.ARM_READY, only 3 cm above the table, would sweep the gripper through the table)
        ok, sol = self.ik_reachable(np.r_[HOME_TCP, 0.0, 1.0, 0.0, 0.0][None], ref=n.ARM_READY, n_solutions=8)
        cands = sol[0][ok[0]]
        assert len(cands), "no IK for the home pose"
        self.home = cands[np.argmin(np.abs(cands - np.asarray(n.ARM_READY)).sum(1))]

    # ---------------------------------------------------------------- helpers
    def _js(self, q) -> JointState:
        q = self.tdt.to_device(torch.as_tensor(np.asarray(q, np.float32)).view(1, -1))
        return JointState.from_position(q, joint_names=self.joint_names)

    def _pose(self, p) -> Pose:
        p = self.tdt.to_device(torch.as_tensor(np.asarray(p, np.float32)).view(-1, 7))
        return Pose(position=p[:, :3], quaternion=p[:, 3:])

    def _cfg(self, **kw) -> MotionGenPlanConfig:
        kw.setdefault("max_attempts", 8)
        kw.setdefault("enable_graph_attempt", 2)
        kw.setdefault("timeout", 5.0)
        kw.setdefault("time_dilation_factor", self.time_dilation)
        return MotionGenPlanConfig(**kw)

    def _to_plan(self, res, tag, quiet=False) -> Plan | None:
        if not bool(res.success.view(-1)[0]):
            if not quiet:
                print(f"   [planner] {tag}: FAILED ({res.status})", flush=True)
            return None
        q = res.get_interpolated_plan().position.view(-1, len(self.joint_names)).cpu().numpy()
        return Plan(resample(q, float(res.interpolation_dt), self.dt), self.dt, tag)

    # ---------------------------------------------------------------- world
    def update_world(self, world: WorldConfig):
        self.mg.update_world(world)

    def set_obstacle_pose(self, name: str, p7):
        self.mg.world_coll_checker.update_obstacle_pose(name, self._pose(p7))

    def enable_obstacle(self, name: str, on: bool = True):
        self.mg.world_coll_checker.enable_obstacle(name, on)

    # ---------------------------------------------------------------- queries
    def fk(self, q) -> np.ndarray:
        p = self.mg.compute_kinematics(self._js(q)).ee_pose
        return pose7(p.position[0].cpu().numpy(), p.quaternion[0].cpu().numpy())

    def wrap_to(self, q, ref) -> np.ndarray:
        """2*pi-equivalent of joint configuration(s) q [..., 6] closest to ``ref``, within the joint limits. Most UR5e
        joints turn +-2*pi, so IK returns any of up to 2^5 equivalent solutions; an unwrapped one can mean a full wrist
        turn that no planner finds a way for."""
        q = np.array(q, float)
        ref = np.broadcast_to(np.asarray(ref, float), q.shape)
        lo, hi = self.joint_limits
        span = hi - lo
        wrap = span > 2 * np.pi + 1e-3
        k = np.round((ref - q) / (2 * np.pi))
        qw = q + 2 * np.pi * k
        qw = np.where(qw > hi, qw - 2 * np.pi, qw)
        qw = np.where(qw < lo, qw + 2 * np.pi, qw)
        return np.where(wrap, qw, q)

    def equivalents(self, q, ref, slack: float = 0.5) -> list[np.ndarray]:
        """2*pi-equivalents of q within the joint limits that are about as near to ``ref`` as the nearest one: where a
        joint is ~pi from ref, turning either way is equally short, but only one way may be collision-free."""
        q = self.wrap_to(q, ref)
        lo, hi = self.joint_limits
        opts = []
        for j in range(len(q)):
            o = [q[j]]
            if hi[j] - lo[j] > 2 * np.pi + 1e-3:
                for alt in (q[j] - 2 * np.pi, q[j] + 2 * np.pi):
                    if lo[j] <= alt <= hi[j] and abs(alt - ref[j]) <= np.pi + slack:
                        o.append(alt)
            opts.append(o)
        out = [np.array(c) for c in itertools.product(*opts)]
        return sorted(out, key=lambda c: np.abs(c - ref).sum())[:4]

    def ik_reachable(self, poses: np.ndarray, ref=None, n_solutions: int = 1):
        """Collision-aware IK for many TCP poses [N, 7]. Returns (success, solutions) as [N] / [N, 6] or, with
        ``n_solutions`` > 1, [N, k] / [N, k, 6] (distinct solutions from different seeds, e.g. the two wrist branches),
        each wrapped to the 2*pi-equivalent nearest ``ref`` (default: the ready configuration). Chunks are padded to a
        fixed batch size so cuRobo's CUDA graph is captured once."""
        ok, sol = [], []
        batch = self.ik_batch_size
        k = n_solutions
        poses = np.asarray(poses, np.float32).reshape(-1, 7)
        for i in range(0, len(poses), batch):
            chunk = poses[i:i + batch]
            m = len(chunk)
            if m < batch:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], batch - m, 0)])
            r = self.ik_many.solve_batch(self._pose(chunk), return_seeds=k)
            ok.append(r.success.view(batch, k).cpu().numpy()[:m])
            sol.append(r.solution.view(batch, k, -1).cpu().numpy()[:m, :, :6])
        ok, sol = np.concatenate(ok), self.wrap_to(np.concatenate(sol), n.ARM_READY if ref is None else ref)
        return (ok[:, 0], sol[:, 0]) if k == 1 else (ok, sol)

    # ---------------------------------------------------------------- planning
    def plan_pose(self, q, p7, tag="to pose", quiet=False, **kw) -> Plan | None:
        res = self.mg.plan_single(self._js(q), self._pose(p7), self._cfg(**kw))
        return self._to_plan(res, tag, quiet)

    def plan_joints(self, q, q_goal, tag="to joints", quiet=False, **kw) -> Plan | None:
        """Joint-space goal: cuRobo trajopt, else the straight joint-space line if it is collision-free (cuRobo's
        joint-space trajopt has no graph search and fails on large reconfigurations even when the straight line is free)."""
        res = self.mg.plan_single_js(self._js(q), self._js(q_goal), self._cfg(**kw))
        p = self._to_plan(res, tag, quiet=True)
        if p is None:
            p = self.joint_line(q, q_goal, tag=f"{tag} (joint line)")
        if p is None and not quiet:
            print(f"   [planner] {tag}: FAILED ({res.status}; straight joint line in collision)", flush=True)
        return p

    def joint_line(self, q, q_goal, tag="joint line", max_step=0.01) -> Plan | None:
        """Straight line in joint space if every sample (``max_step`` rad apart) is collision-free and within limits;
        smoothstep timing with the peak joint speed at ``time_dilation`` x the joint velocity limit."""
        q, q_goal = np.asarray(q, float), np.asarray(q_goal, float)
        d = np.abs(q_goal - q).max()
        m = max(2, int(np.ceil(d / max_step)))
        qs = np.linspace(q, q_goal, m + 1)
        js = JointState.from_position(self.tdt.to_device(torch.as_tensor(qs, dtype=torch.float32)), joint_names=self.joint_names)
        if not bool(self.mg.check_constraints(js).feasible.all()):
            return None
        vmax = self.vel_limit * self.time_dilation
        T = max(1.5 * np.max(np.abs(q_goal - q) / vmax), 2 * self.dt)
        t = np.r_[np.arange(0, T, self.dt), T]
        u = np.clip(t / T, 0, 1)
        prog = u * u * (3 - 2 * u)
        return Plan(q + prog[:, None] * (q_goal - q), self.dt, tag)

    def plan_to_config(self, q, q_goal, tag="to config", via=None) -> Plan | None:
        """Joint-space plan; if the direct one fails, via ``via`` (default: self.home, high above the table)."""
        p = self.plan_joints(q, q_goal, tag=tag, quiet=True)
        if p is not None:
            return p
        via = np.asarray(self.home if via is None else via, float)
        if np.abs(np.asarray(q) - via).max() < 1e-3 or np.abs(np.asarray(q_goal) - via).max() < 1e-3:
            print(f"   [planner] {tag}: FAILED (joint-space trajopt)", flush=True)
            return None
        a = self.plan_joints(q, via, tag=f"{tag} (to home)", quiet=True)
        b = self.plan_joints(via, q_goal, tag=f"{tag} (from home)", quiet=True) if a is not None else None
        if b is None:
            print(f"   [planner] {tag}: FAILED (joint-space trajopt, also via home)", flush=True)
            return None
        return a + b

    def plan_goalset(self, q, poses: np.ndarray, tag="to goal set", **kw) -> tuple[Plan | None, int]:
        """Plan to whichever of the TCP poses [N, 7] is best reachable; returns (plan, index into poses)."""
        p = self._pose(poses)
        goal = Pose(position=p.position.unsqueeze(0), quaternion=p.quaternion.unsqueeze(0))  # [1, N, ...]
        res = self.mg.plan_goalset(self._js(q), goal, self._cfg(**kw))
        plan = self._to_plan(res, tag)
        return plan, (int(res.goalset_index.view(-1)[0]) if plan is not None else -1)

    def plan_linear(self, q, p7, tag="linear", frame="goal", free_axis=2, ignore: list[str] = (),
                    disable_links: list[str] = (), hold_weight: float = 0.1, max_speed: float = 0.06, **kw) -> Plan | None:
        """Straight-line TCP motion from FK(q) to pose p7 with the orientation held (the start must differ from the goal
        only along ``free_axis`` (0/1/2 = x/y/z) of ``frame``: "goal" = the target TCP frame, "base" = arm-base frame).

        First cuRobo trajectory optimisation with a hold-pose cost; if that fails (it does when an arm link is within a
        few cm of an obstacle, e.g. the upper arm over a crate's front wall), a Cartesian IK chain (plan_cartesian),
        which is straight by construction. ``ignore``: world obstacles switched off for this motion (the object being
        grasped, a crate being hooked); ``disable_links``: robot links whose world collision is switched off."""
        w = [hold_weight] * 6  # (rx ry rz x y z); cuRobo's plan_grasp uses 0.1
        w[3 + free_axis] = 0.0
        metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=self.tdt.to_device(w),
                                project_to_goal_frame=(frame == "goal"))
        with self.relaxed(ignore, disable_links):
            res = self.mg.plan_single(self._js(q), self._pose(p7), self._cfg(pose_cost_metric=metric, enable_graph_attempt=None, **kw))
            if bool(res.success.view(-1)[0]):
                return self._to_plan(res, tag)
            return self.plan_cartesian(q, p7, tag=f"{tag} (IK chain)", max_speed=max_speed)

    def plan_cartesian(self, q, p7, tag="cartesian", step=0.005, max_speed=0.06, max_joint_step=0.08,
                       quiet=False) -> Plan | None:
        """Exactly straight TCP path from FK(q) to p7 (orientation slerped): collision-aware IK every ``step`` m, each
        seeded with and regularised towards the previous solution; fails if any IK fails or a joint jumps by more than
        ``max_joint_step`` rad between waypoints (a branch switch). Timed with a smoothstep profile whose peak TCP speed is
        ``max_speed`` m/s (zero velocity at both ends)."""
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        p0 = self.fk(q)
        dist = float(np.linalg.norm(np.asarray(p7[:3]) - p0[:3]))
        rots = R.from_quat(np.stack([p0[3:7], np.asarray(p7[3:7])])[:, [1, 2, 3, 0]])
        ang = (rots[1] * rots[0].inv()).magnitude()
        m = max(2, int(np.ceil(max(dist / step, ang / 0.02))))
        s = np.linspace(0, 1, m + 1)
        slerp = Slerp([0, 1], rots)
        qs = [np.asarray(q, float)]
        for k in range(1, m + 1):
            xyzw = slerp([s[k]]).as_quat()[0]
            wp = np.r_[(1 - s[k]) * p0[:3] + s[k] * np.asarray(p7[:3]), xyzw[3], xyzw[:3]]
            prev = self.tdt.to_device(torch.as_tensor(qs[-1], dtype=torch.float32))
            r = self.ik_one.solve_single(self._pose(wp), retract_config=prev.view(1, -1), seed_config=prev.view(1, 1, -1))
            if not bool(r.success.view(-1)[0]):
                if not quiet:
                    print(f"   [planner] {tag}: FAILED (IK at {s[k]:.0%} of the path)", flush=True)
                return None
            qk = self.wrap_to(r.solution.view(-1)[: len(self.joint_names)].cpu().numpy(), qs[-1])
            if np.abs(qk - qs[-1]).max() > max_joint_step:
                if not quiet:
                    print(f"   [planner] {tag}: FAILED (joint jump {np.abs(qk - qs[-1]).max():.2f} rad at {s[k]:.0%})", flush=True)
                return None
            qs.append(qk)
        qs = np.array(qs)
        T = max(1.5 * dist / max_speed, 1.5 * ang / 0.5, 2 * self.dt)  # smoothstep peak speed = 1.5 x mean speed
        t = np.r_[np.arange(0, T, self.dt), T]
        u = np.clip(t / T, 0, 1)
        prog = u * u * (3 - 2 * u)  # path parameter in [0, 1]
        out = np.stack([np.interp(prog, s, qs[:, j]) for j in range(qs.shape[1])], axis=1)
        return Plan(out, self.dt, tag)

    def plan_grasp(self, q, grasps: np.ndarray, approach: float = 0.10, target: str | None = None,
                   max_tries: int = 8, n_branches: int = 4, transit_z: float = 0.35, tag="grasp"):
        """Plan reaching one of ``grasps`` [N, 7] (TCP poses): a free-space joint motion to the pregrasp (``approach`` m
        back along the grasp's z axis), then a straight approach.

        Built backwards from the grasp, so that the pregrasp is on the same IK branch as the grasp: (1) collision-aware IK
        at every grasp pose (``n_branches`` distinct solutions each; only the ``target`` obstacle, the object, is
        ignored); (2) (grasp, solution) pairs are tried nearest-first in joint space from q (candidates earlier in
        ``grasps`` win ties: pass them best first); (3) per pair, a Cartesian IK chain from the grasp configuration back to
        the pregrasp (straight, collision-checked, no branch switch) and a free-space joint plan to its end; failing
        that, via home; failing that, to a point ``transit_z`` above the table straight over the pregrasp (same branch)
        and straight down from there (room for wrist reconfigurations high up, as the old scripts' fixed transit height).
        (Planning to the pregrasp *pose* lets cuRobo pick any branch; from some, the descent is impossible, e.g. with the
        wrist ending under a drawer housing's roof. And the nearest branch matters: reaching a wrist-flipped one means a
        full wrist reconfiguration among obstacles.)
        Returns (plan_to_pregrasp, plan_approach, index into grasps) or (None, None, -1)."""
        grasps = np.asarray(grasps, float).reshape(-1, 7)
        ign = [target] if target else []
        with self.relaxed(ign):
            ok, sol = self.ik_reachable(grasps, ref=q, n_solutions=n_branches)
        pairs = []
        for gi in range(len(grasps)):
            seen = []
            for b in np.where(ok[gi])[0]:
                for qe in self.equivalents(sol[gi, b], np.asarray(q, float)):
                    if any(np.abs(qe - s_).max() < 0.2 for s_ in seen):
                        continue  # same solution found from another seed
                    seen.append(qe)
                    pairs.append((np.abs(qe - np.asarray(q)).sum() + 0.05 * gi, gi, qe))
        pairs.sort(key=lambda t: t[0])
        if not pairs:
            print(f"   [planner] {tag}: no collision-free IK for any of {len(grasps)} grasps", flush=True)
            return None, None, -1
        lines = []
        for _, gi, qg in pairs[:max_tries]:
            with self.relaxed(ign):
                back = self.plan_cartesian(qg, pregrasp(grasps[gi], approach), tag=f"{tag} #{gi}: retreat line", quiet=not self.verbose)
            if back is None:
                continue
            lines.append((gi, back))
            to_pre = self.plan_joints(q, back.q[-1], tag=f"{tag} #{gi}: to pregrasp", quiet=True)
            if to_pre is not None:
                return to_pre, Plan(back.q[::-1].copy(), back.dt, f"{tag} #{gi}: approach"), int(gi)
        for gi, back in lines[:3]:  # second chance: via home
            to_pre = self.plan_to_config(q, back.q[-1], tag=f"{tag} #{gi}: to pregrasp")
            if to_pre is not None:
                return to_pre, Plan(back.q[::-1].copy(), back.dt, f"{tag} #{gi}: approach"), int(gi)
        for gi, back in lines[:3]:  # third: straight down from transit height, on the pregrasp's IK branch
            pre7 = pregrasp(grasps[gi], approach)
            if pre7[2] >= transit_z - 0.02:
                continue
            top7 = np.r_[pre7[:2], transit_z, pre7[3:]]
            up = self.plan_cartesian(back.q[-1], top7, tag=f"{tag} #{gi}: lift line", quiet=not self.verbose)
            if up is None:
                continue
            to_top = self.plan_to_config(q, up.q[-1], tag=f"{tag} #{gi}: to transit")
            if to_top is not None:
                down = Plan(up.q[::-1].copy(), up.dt, "down")
                return to_top + down, Plan(back.q[::-1].copy(), back.dt, f"{tag} #{gi}: approach"), int(gi)
        print(f"   [planner] {tag}: no feasible approach ({len(pairs)} IK solutions, {len(lines)} straight lines)", flush=True)
        return None, None, -1

    @contextmanager
    def relaxed(self, ignore=(), disable_links=()):
        """Temporarily switch off world obstacles and/or robot links' world collision."""
        for nm in ignore:
            self.enable_obstacle(nm, False)
        if disable_links:
            self.mg.toggle_link_collision(list(disable_links), False)
        try:
            yield
        finally:
            for nm in ignore:
                if nm != self.attached:
                    self.enable_obstacle(nm, True)
            if disable_links:
                self.mg.toggle_link_collision(list(disable_links), True)

    # ---------------------------------------------------------------- attached objects
    def attach(self, q, obstacle: str, surface_radius: float = 0.004) -> bool:
        ok = self.mg.attach_objects_to_robot(self._js(q), [obstacle], surface_sphere_radius=surface_radius)
        if ok:
            self.attached = obstacle
        return bool(ok)

    def detach(self):
        if self.attached is not None:
            self.mg.detach_object_from_robot()
            self.enable_obstacle(self.attached, True)
            self.attached = None


def resample(q: np.ndarray, dt_in: float, dt_out: float) -> np.ndarray:
    """Joint trajectory [T, n] at dt_in -> at dt_out (linear interpolation; the last sample is always the end point)."""
    t_end = (len(q) - 1) * dt_in
    t = np.arange(0.0, t_end + 1e-9, dt_out)
    if t[-1] < t_end - 1e-9:
        t = np.r_[t, t_end]
    ti = np.arange(len(q)) * dt_in
    return np.stack([np.interp(t, ti, q[:, j]) for j in range(q.shape[1])], axis=1)


def pregrasp(g7, d) -> np.ndarray:
    """Pose d metres back along the TCP's own z axis (the approach direction) from g7."""
    from scipy.spatial.transform import Rotation as R

    z = R.from_quat(np.asarray(g7[3:7])[[1, 2, 3, 0]]).as_matrix()[:, 2]
    return np.r_[np.asarray(g7[:3]) - d * z, g7[3:7]]


def cuboid(name, center, dims, quat=(1, 0, 0, 0)) -> Cuboid:
    return Cuboid(name=name, pose=[*map(float, center), *map(float, quat)], dims=[*map(float, dims)])


def mesh(name, obj_file, p7) -> Mesh:
    return Mesh(name=name, file_path=str(obj_file), pose=[*map(float, p7)])
