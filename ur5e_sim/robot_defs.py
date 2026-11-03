"""Joint / link names and reference values for the UR5e + Robotiq 2F-85 arm.

Copied from steve_il/robots/steve_names.py (the Steve robot project, commit e7592bb), reduced to what the arm-only
environment needs. Always address joints by name: Isaac orders DOFs by USD traversal, not URDF order.
"""

EE_LINK = "ur5e_tool0"  # tool frame of the UR5e (the gripper mounts here, z = approach axis)

ARM_JOINTS = (
    "ur5e_shoulder_pan_joint",
    "ur5e_shoulder_lift_joint",
    "ur5e_elbow_joint",
    "ur5e_wrist_1_joint",
    "ur5e_wrist_2_joint",
    "ur5e_wrist_3_joint",
)

GRIPPER_DRIVEN_JOINT = "finger_joint"  # 0 (open) .. 0.8 rad (closed)
# joint -> multiplier w.r.t. finger_joint (URDF <mimic>, offset 0)
GRIPPER_MIMIC_JOINTS = {
    "left_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
}

# Joint configuration with the tool pointing straight down (found with the IK controller in the Steve project;
# wrist_3 is the 2*pi-equivalent solution closest to 0).
ARM_READY = (-0.344, -1.112, 1.876, 0.807, 1.571, 1.226)

# Tool centre point: midpoint between the two finger pads, along the tool z axis (hand-derived from the URDF chain
# 0.0085 + 0.054904 - 0.0041 + 0.0471 + 0.03242, checked against the simulated pads in the Steve project).
TCP_OFFSET_Z = 0.138824
