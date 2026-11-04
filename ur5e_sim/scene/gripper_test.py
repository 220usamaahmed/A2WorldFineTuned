"""Simplified gripper, in free air: commanded vs reached pad gap, and how well the visual linkage follows the physics pads.

    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 python gripper_test.py --headless
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import robot_defs as n  # noqa: E402
import ur5e_env as E  # noqa: E402

HOME = [-0.391, -1.497, 1.492, 1.576, 1.571, 1.18]  # planner.home: tool down, TCP 0.35 m above the table
import os  # noqa: E402

cfg = E.use_simple_gripper(E.Ur5eTableJointEnvCfg(), grip_force=float(os.environ.get("GRIP_FORCE", "40")))
cfg.scene.cube.spawn.rigid_props.disable_gravity = True  # floats between the pads for the contact test
cfg.scene.robot.init_state.joint_pos.update(dict(zip(n.ARM_JOINTS, HOME)))
env = ManagerBasedRLEnv(cfg)
env.reset()
robot = env.scene["robot"]
phys = robot.find_bodies(["left_pad_phys", "right_pad_phys"], preserve_order=True)[0]
slides = robot.find_joints(["left_pad_slide", "right_pad_slide"], preserve_order=True)[0]
print("slide joint ids", slides, [robot.joint_names[j] for j in slides], "limits", robot.data.joint_pos_limits[0, slides].cpu().numpy().tolist())
print("slide drive: stiffness", robot.data.joint_stiffness[0, slides].cpu().numpy(), "damping", robot.data.joint_damping[0, slides].cpu().numpy(),
      "effort limit", robot.data.joint_effort_limits[0, slides].cpu().numpy(), "vel limit", robot.data.joint_vel_limits[0, slides].cpu().numpy(),
      "physx max force", robot.root_physx_view.get_dof_max_forces()[0, slides].cpu().numpy(), "physx stiffness", robot.root_physx_view.get_dof_stiffnesses()[0, slides].cpu().numpy())
vis = robot.find_bodies(["left_inner_finger_pad", "right_inner_finger_pad"], preserve_order=True)[0]
term = env.action_manager.get_term("gripper")
for a in (0.0, 0.2, 0.4, 0.6, 0.75, 0.3, 0.0):
    for _ in range(12):
        env.step(torch.tensor([[*HOME, a]], device=env.device))
    pp = robot.data.body_pos_w[0, phys].cpu().numpy()
    pv = robot.data.body_pos_w[0, vis].cpu().numpy()
    gap = np.linalg.norm(pp[0] - pp[1]) - 0.00635
    want = float(term.gap_for_angle(torch.tensor([a], device=env.device))[0])
    off = np.linalg.norm(pp - pv, axis=1)
    sj = robot.data.joint_pos[0, slides].cpu().numpy()
    st = robot.data.joint_pos_target[0, slides].cpu().numpy()
    print(f"   slides {np.round(sj * 1000, 1)} mm (targets {np.round(st * 1000, 1)})")
    print(f"command {a:.2f} rad: pad gap {gap * 1000:5.1f} mm (want {want * 1000:5.1f}); visual pads {np.round(off * 1000, 1)} mm from the physics pads")
# ---- in contact: a 5 cm cube floating (no gravity) between the open pads, offset towards the left pad by each OFFSET ----
OFFSETS = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else ["0.008"])]
for off in OFFSETS:
    print(f'--- cube offset {off * 1000:+.0f} mm towards the left pad', flush=True)
    for _ in range(15):
        env.step(torch.tensor([[*HOME, 0.0]], device=env.device))
    cube = env.scene["cube"]
    tool = robot.find_bodies(["ur5e_tool0"])[0][0]
    from scipy.spatial.transform import Rotation as R  # noqa: E402

    q = robot.data.body_quat_w[0, tool].cpu().numpy()
    Rt = R.from_quat(q[[1, 2, 3, 0]])
    tcp = robot.data.body_pos_w[0, tool].cpu().numpy() + Rt.apply([0, 0, n.TCP_OFFSET_Z])
    left_dir = robot.data.body_pos_w[0, phys[0]].cpu().numpy() - tcp
    left_dir /= np.linalg.norm(left_dir)
    c = tcp + off * left_dir
    cube.write_root_pose_to_sim(torch.tensor([[*c, *q]], dtype=torch.float32, device=env.device))
    cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for k in range(30):
        a = min(0.75, 0.05 * k)
        env.step(torch.tensor([[*HOME, a]], device=env.device))
        if k % 3 == 2:
            sj = robot.data.joint_pos[0, slides].cpu().numpy()
            cc = cube.data.root_pos_w[0].cpu().numpy()
            pp = robot.data.body_pos_w[0, phys].cpu().numpy()
            d_tool = Rt.inv().apply(cc - c) * 1000  # tool frame: x/y one is the closing axis, z the approach axis
            cq = cube.data.root_quat_w[0].cpu().numpy()
            rot = np.degrees((R.from_quat(cq[[1, 2, 3, 0]]) * R.from_quat(q[[1, 2, 3, 0]]).inv()).magnitude())
            print(f"   contact step {k:2d} cmd {a:.2f}: slides {np.round(sj * 1000, 1)} mm, cube moved (tool frame) {np.round(d_tool, 1)} mm, rotated {rot:.0f} deg, "
                  f"pad-pad {np.linalg.norm(pp[0] - pp[1]) * 1000 - 6.35:.1f} mm")
env.close()
app.close()
