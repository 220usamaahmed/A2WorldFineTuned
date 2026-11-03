"""Build the scene, settle at the ready pose, print robot state and save the three camera views to preview.png."""
import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="preview.png")
parser.add_argument("--steps", type=int, default=15)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import ur5e_env as E  # noqa: E402
import robot_defs as n  # noqa: E402

env = ManagerBasedRLEnv(E.Ur5eTableEnvCfg())
env.reset()
robot = env.scene["robot"]
print("joint names:", robot.joint_names)
print("body names:", robot.body_names)
act = torch.tensor([[*E.READY_TCP, *E.TOOL_DOWN_QUAT, 0.0]], dtype=torch.float32, device=env.device)
for _ in range(args.steps):
    env.step(act)
tool = robot.find_bodies(n.EE_LINK)[0][0]
print("tool0 pos (root frame):", (robot.data.body_pos_w[0, tool] - robot.data.root_pos_w[0]).cpu().numpy().round(4))
print("arm q:", robot.data.joint_pos[0, robot.find_joints(n.ARM_JOINTS, preserve_order=True)[0]].cpu().numpy().round(3))
imgs = []
for name in ("cam_ext1", "cam_ext2", "cam_wrist"):
    rgb = env.scene[name].data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    print(name, rgb.shape, rgb.mean().round(1))
    imgs.append(rgb)
cv2.imwrite(args.out, cv2.cvtColor(np.concatenate(imgs, 1), cv2.COLOR_RGB2BGR))
print("saved", args.out)
env.close()
simulation_app.close()
