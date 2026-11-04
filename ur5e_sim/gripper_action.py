"""Gripper action term for the Robotiq 2F-85 (Isaac Lab): finger angle command -> position targets for all six joints.

Copied unchanged from steve_il/mdp/actions.py (the Steve robot project, commit e7592bb), where it was tuned and verified
for grasping a 5 cm cube. See the class docstring for why each finger is driven separately.
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


# ----------------------------------------------------------------------------------------------
# Gripper: both fingers driven, the rest of each linkage follows its own finger
# ----------------------------------------------------------------------------------------------
class MimicGripperPositionAction(ActionTerm):
    """Finger angle command -> position targets for all six Robotiq joints, each finger independent.

    In the imported model only ``finger_joint`` is driven; the other five joints stand in for the real
    four-bar linkage and follow it through PhysX mimic constraints whose stiffness scales with the (tiny)
    link inertia. Under load the linkage gives way: the pads pitch, the finger over-closes and a grasped
    object tilts and slides out. Making every follower chase the *left* finger fails differently: when the
    object blocks one side, the stiff followers on that side fight their own mismatch.

    Here each finger is handled on its own. The two outer knuckles (``driven_joints``) both get the
    commanded angle, with a low actuator gain that sets the squeeze force of that finger (gain x remaining
    error; a stiff gain squeezes a 0.1 kg cube with >100 N and the linkage gives way); at
    equilibrium the object is centred and squeezed equally from both sides. Every other joint is a
    follower: it is position-driven to ``multiplier * measured angle`` of the outer knuckle of *its own
    side* (``followers``: joint -> (leader, multiplier)). With a high gain on the inner-finger joints each
    pad stays parallel to the palm under contact loads.

    The action is the commanded finger angle [rad], 0 (open) ... 0.8 (closed).
    """

    cfg: MimicGripperPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: MimicGripperPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        names = [*cfg.driven_joints, *cfg.followers]
        ids, found = self._asset.find_joints(names, preserve_order=True)
        if list(found) != names:
            raise ValueError(f"expected joints {names}, found {found}")
        self._joint_ids = ids
        n_driven = len(cfg.driven_joints)
        self._n_driven = n_driven
        leader_index = [cfg.driven_joints.index(leader) for leader, _ in cfg.followers.values()]
        self._leader_ids = [ids[i] for i in leader_index]
        self._multipliers = torch.tensor([m for _, m in cfg.followers.values()], device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 1, device=self.device)  # commanded angle
        self._targets = torch.zeros(self.num_envs, len(ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, self.cfg.min_angle, self.cfg.max_angle)

    def apply_actions(self):
        self._targets[:, : self._n_driven] = self._processed_actions
        leader_angle = self._asset.data.joint_pos[:, self._leader_ids]
        self._targets[:, self._n_driven :] = leader_angle * self._multipliers
        self._asset.set_joint_position_target(self._targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class MimicGripperPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MimicGripperPositionAction

    driven_joints: list[str] = MISSING
    """Joints that all receive the commanded angle (the two outer knuckles)."""
    followers: dict[str, tuple[str, float]] = MISSING
    """Follower joint -> (driven joint it follows, multiplier)."""
    min_angle: float = 0.0
    max_angle: float = 0.8


# ----------------------------------------------------------------------------------------------
# Simplified physics (assets/ur5e_gripper_simple.urdf, make_simple_gripper.py): two sliding pads
# ----------------------------------------------------------------------------------------------
class SimpleGripperAction(ActionTerm):
    """Finger angle command [rad] (0 open ... 0.8 closed, same meaning as MimicGripperPositionAction) for the gripper
    with simplified physics.

    Physics: the two prismatic pad slides get the slide position that puts the pads at the inner gap the real linkage
    would have at that angle. Their actuators are stiff but force-limited (``effort_limit_sim`` = the squeeze force):
    once the pads touch an object they press with that force, whatever the object's width.
    Visual: the six Robotiq linkage joints (no colliders) are position-driven, per side, to the angle at which the real
    linkage would put that side's pad where the physics pad actually is. So the rendered fingers follow the pads.
    """

    cfg: SimpleGripperActionCfg
    _asset: Articulation

    def __init__(self, cfg: SimpleGripperActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        import json

        info = json.loads(open(cfg.info_path).read())
        self._ang = torch.tensor(info["angle"], device=self.device)
        self._gap = torch.tensor(info["gap"], device=self.device)  # decreasing with the angle
        s = info["slides"]["left"]
        self._ratio = abs(s["open_pos"][1] - s["closed_pos"][1]) / s["travel"]  # sideways pad motion per m of slide (0.95)
        self._travel = s["travel"]
        ids, found = self._asset.find_joints(list(cfg.slide_joints), preserve_order=True)
        assert list(found) == list(cfg.slide_joints), found
        self._slide_ids = ids
        self._vis = {}
        for side, joints in (("left", cfg.left_linkage), ("right", cfg.right_linkage)):
            ids, found = self._asset.find_joints(list(joints), preserve_order=True)
            assert list(found) == list(joints), found
            self._vis[side] = (ids, torch.tensor(list(joints.values()), device=self.device))
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd = torch.zeros(self.num_envs, 1, device=self.device)

    @staticmethod
    def _interp(x, xp, fp):
        """Piecewise-linear interpolation on increasing xp (torch, batched x)."""
        x = x.clamp(xp[0], xp[-1])
        i = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
        x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
        return f0 + (f1 - f0) * (x - x0) / (x1 - x0)

    def gap_for_angle(self, a):
        return self._interp(a, self._ang, self._gap)

    def angle_for_gap(self, g):
        return self._interp(g, self._gap.flip(0), self._ang.flip(0))

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._cmd[:] = torch.clamp(actions, self.cfg.min_angle, self.cfg.max_angle)

    def apply_actions(self):
        # physics: both slides to the position of the commanded gap
        g = self.gap_for_angle(self._cmd[:, 0])
        s = ((self._gap[0] - g) / (2.0 * self._ratio)).clamp(0.0, self._travel)  # both pads move in by ratio * s
        # coupling, as the real gripper (one actuator drives both fingers): neither pad's target runs more than
        # ``coupling`` ahead of the other pad. Two independent force-limited pads are a neutral tie: one wins, shoves the
        # object 27 mm and pushes the other pad back to its stop. ``coupling`` = force limit / stiffness keeps the full
        # squeeze force while the object ends up centred within a few mm.
        meas = self._asset.data.joint_pos[:, self._slide_ids]
        tgt = torch.stack([torch.minimum(s, meas[:, 1] + self.cfg.coupling), torch.minimum(s, meas[:, 0] + self.cfg.coupling)], 1)
        self._asset.set_joint_position_target(tgt, joint_ids=self._slide_ids)
        # visual: each side's linkage to the angle that matches its measured pad position
        for k, side in enumerate(("left", "right")):
            g_side = self._gap[0] - 2.0 * self._ratio * meas[:, k]  # the gap if both pads were where this one is
            a = self.angle_for_gap(g_side)
            ids, mult = self._vis[side]
            self._asset.set_joint_position_target(a[:, None] * mult[None], joint_ids=ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw[env_ids] = 0.0
        self._cmd[env_ids] = 0.0


@configclass
class SimpleGripperActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SimpleGripperAction

    info_path: str = MISSING
    """assets/simple_gripper.json (written by make_simple_gripper.py)."""
    slide_joints: tuple[str, str] = ("left_pad_slide", "right_pad_slide")
    left_linkage: dict[str, float] = MISSING
    """Visual linkage joint -> multiplier on the left finger angle."""
    right_linkage: dict[str, float] = MISSING
    coupling: float = 0.005
    """Max lead of one pad's target over the other pad's position [m] (set to force limit / pad stiffness)."""
    min_angle: float = 0.0
    max_angle: float = 0.8
