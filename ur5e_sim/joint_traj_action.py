"""Arm action term for following planned joint trajectories (cuRobo) at the 10 Hz control rate.

The action is the arm's joint configuration to reach at the end of the control step (6 values, rad). The term spreads
that step over the physics substeps: at substep k of K the position target is q_prev + (k+1)/K * (q_cmd - q_prev) and the
velocity target is (q_cmd - q_prev) / control_dt. The velocity feed-forward matters: the arm actuators are implicit PD
drives, and with a zero velocity target the damping term pulls against every motion, so the arm lags its targets by about
velocity * damping / stiffness (0.1 s here; this caused the ~3 cm tracking error of the old IK-pose action at speed).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class JointTrajectoryAction(ActionTerm):
    cfg: JointTrajectoryActionCfg
    _asset: Articulation

    def __init__(self, cfg: JointTrajectoryActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        ids, found = self._asset.find_joints(list(cfg.joint_names), preserve_order=True)
        if list(found) != list(cfg.joint_names):
            raise ValueError(f"expected joints {cfg.joint_names}, found {found}")
        self._joint_ids = ids
        self._n_sub = env.cfg.decimation
        self._dt = env.cfg.decimation * env.cfg.sim.dt
        nj = len(ids)
        self._raw = torch.zeros(self.num_envs, nj, device=self.device)
        self._cmd = torch.zeros(self.num_envs, nj, device=self.device)
        self._start = torch.zeros(self.num_envs, nj, device=self.device)
        self._vel = torch.zeros(self.num_envs, nj, device=self.device)
        self._k = 0
        self._initialised = False

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd

    def process_actions(self, actions: torch.Tensor):
        if not self._initialised:  # first command: start from where the arm is
            self._cmd[:] = self._asset.data.joint_pos[:, self._joint_ids]
            self._initialised = True
        self._raw[:] = actions
        self._start[:] = self._cmd  # the previous command (not the measured state): targets stay continuous
        self._cmd[:] = actions
        self._vel[:] = (self._cmd - self._start) / self._dt if self.cfg.feed_forward else 0.0
        self._k = 0

    def apply_actions(self):
        self._k = min(self._k + 1, self._n_sub)
        a = self._k / self._n_sub if self.cfg.interpolate else 1.0
        self._asset.set_joint_position_target(self._start + a * (self._cmd - self._start), joint_ids=self._joint_ids)
        vel = self._vel if self._k < self._n_sub or not self.cfg.stop_at_end else torch.zeros_like(self._vel)
        self._asset.set_joint_velocity_target(vel, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._initialised = False
        self._vel[env_ids] = 0.0
        self._k = 0


@configclass
class JointTrajectoryActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = JointTrajectoryAction

    joint_names: list[str] = MISSING
    interpolate: bool = True
    """Ramp the position target linearly over the substeps (otherwise jump to the command at once)."""
    feed_forward: bool = True
    """Set the velocity target to the planned joint velocity (otherwise 0, i.e. pure position control)."""
    stop_at_end: bool = False
    """Zero the velocity target on the last substep. Off: the next command usually continues the motion."""
