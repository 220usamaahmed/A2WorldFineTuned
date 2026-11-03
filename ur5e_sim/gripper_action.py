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
