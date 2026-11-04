"""Generate assets/ur5e_gripper_simple.urdf: the UR5e + Robotiq 2F-85 with simplified gripper *physics*.

    python make_simple_gripper.py          (then: OMNI_KIT_ACCEPT_EULA=YES python convert_arm.py --headless --simple)

Why: in the original model the pads sit at the end of the four-bar linkage (outer knuckle -> outer finger -> inner finger,
closed by mimic constraints). In contact the linkage gives way instead of squeezing (finger joint at 0.55 rad with the pad
where 0.12 rad would put it), and sampled grasps lifted 5-7 of 20 objects on the table although the arm reached every
grasp (docs/stage2_motion_planning.md).

What changes (everything else, incl. all visual meshes, is copied unchanged):
  * the Robotiq finger links (knuckles, fingers, the original pads) lose their colliders and <mimic> tags: they become
    a pure visual, position-driven to follow the physics pads (gripper_action.SimpleGripperAction);
  * two new links, left_pad_phys / right_pad_phys, carry a pad collider (the real pad's 22 x 37.5 mm gripping face, but
    15 mm thick, extending outwards: thin pads tunnel through objects at high squeeze), no visual,
    on prismatic joints left_pad_slide / right_pad_slide from robotiq_arg2f_base_link. Each slides along the straight
    line from the real pad's open position to its closed position (finger_joint 0 -> 0.8 rad), keeping the pad's
    orientation. The real pad moves on an arc (13.9 mm towards the fingertips in total); the line deviates from it by
    <= ~5 mm, so the visual pads stay close to the physics pads;
  * the palm (robotiq_arg2f_base_link) keeps its collider.
The squeeze becomes a force-limited prismatic drive: a set force whatever the object's width.
Writes the slide geometry to assets/simple_gripper.json (used by gripper_action.py and planning/grasps.py).
"""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
SRC = HERE / "assets" / "ur5e_gripper.urdf"
DST = HERE / "assets" / "ur5e_gripper_simple.urdf"
INFO = HERE / "assets" / "simple_gripper.json"
BASE = "robotiq_arg2f_base_link"
FINGER_LINKS = ["left_outer_knuckle", "left_outer_finger", "left_inner_knuckle", "left_inner_finger", "left_inner_finger_pad",
                "right_outer_knuckle", "right_outer_finger", "right_inner_knuckle", "right_inner_finger", "right_inner_finger_pad"]
PAD_BOX = (0.022, 0.00635, 0.0375)  # the real pad, in the original pad link's frame (thickness along y)
# The physics pad keeps the real pad's inner (gripping) face but is 15 mm thick, extending outwards: a 6.35 mm box
# squeezing at tens of N lets a light object get pushed more than half-way through and tunnel out (seen at 40 N).
PHYS_THICK = 0.015
PAD_MASS = 0.02


def pad_poses(robot, angle):
    """4x4 poses of the two original pads in the gripper base frame at finger_joint = angle (mimic joints applied)."""
    robot.update_cfg({"finger_joint": angle})
    Tb = np.linalg.inv(robot.get_transform(BASE))
    return {s: Tb @ robot.get_transform(f"{s}_inner_finger_pad") for s in ("left", "right")}


def angles_table(robot, n=17):
    """finger angle -> inner gap between the pads (the visual linkage's angle for a given pad position)."""
    a = np.linspace(0.0, 0.8, n)
    gap = []
    for x in a:
        P = pad_poses(robot, x)
        gap.append(float(np.linalg.norm(P["left"][:3, 3] - P["right"][:3, 3]) - PAD_BOX[1]))
    return a.tolist(), gap


def main():
    robot = yourdfpy.URDF.load(str(SRC), build_scene_graph=True, load_meshes=False)
    open_, closed = pad_poses(robot, 0.0), pad_poses(robot, 0.8)
    tree = ET.parse(SRC)
    root = tree.getroot()

    for link in root.findall("link"):
        if link.get("name") in FINGER_LINKS:
            for c in link.findall("collision"):
                link.remove(c)
    for joint in root.findall("joint"):
        for m in joint.findall("mimic"):
            joint.remove(m)

    info = {"slides": {}, "pad_box": PAD_BOX}
    for side in ("left", "right"):
        T0, T1 = open_[side], closed[side]
        d = T1[:3, 3] - T0[:3, 3]
        travel = float(np.linalg.norm(d))
        axis_child = T0[:3, :3].T @ (d / travel)  # URDF joint axes are given in the joint (child) frame
        rpy = R.from_matrix(T0[:3, :3]).as_euler("xyz")
        link = ET.SubElement(root, "link", name=f"{side}_pad_phys")
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "mass", value=str(PAD_MASS))
        ET.SubElement(inert, "inertia", ixx="2e-06", ixy="0", ixz="0", iyy="2e-06", iyz="0", izz="2e-06")
        # outward = away from the other pad, expressed in the pad frame (its y axis is the thickness axis)
        y_in_base = T0[:3, 1]
        outward_sign = 1.0 if np.sign(y_in_base[1]) == np.sign(T0[1, 3]) else -1.0
        shift = outward_sign * (PHYS_THICK - PAD_BOX[1]) / 2
        col = ET.SubElement(link, "collision")
        ET.SubElement(col, "origin", rpy="0 0 0", xyz=f"0 {shift:.6f} 0")
        ET.SubElement(ET.SubElement(col, "geometry"), "box", size=f"{PAD_BOX[0]} {PHYS_THICK} {PAD_BOX[2]}")
        j = ET.SubElement(root, "joint", name=f"{side}_pad_slide", type="prismatic")
        ET.SubElement(j, "origin", rpy=" ".join(f"{v:.9f}" for v in rpy), xyz=" ".join(f"{v:.9f}" for v in T0[:3, 3]))
        ET.SubElement(j, "parent", link=BASE)
        ET.SubElement(j, "child", link=f"{side}_pad_phys")
        ET.SubElement(j, "axis", xyz=" ".join(f"{v:.9f}" for v in axis_child))
        ET.SubElement(j, "limit", effort="1000", lower="0", upper=f"{travel:.6f}", velocity="0.5")
        info["slides"][side] = dict(open_pos=T0[:3, 3].tolist(), closed_pos=T1[:3, 3].tolist(), travel=travel)

    ET.indent(tree, space="  ")
    tree.write(DST, xml_declaration=True, encoding="utf-8")

    # the visual linkage: finger angle <-> inner gap; and how far the line deviates from the real pad arc
    a, gap = angles_table(robot)
    info["angle"], info["gap"] = a, gap
    dev = []
    for x in a:
        P = pad_poses(robot, x)
        for side in ("left", "right"):
            s = info["slides"][side]
            p0, p1 = np.array(s["open_pos"]), np.array(s["closed_pos"])
            u = (p1 - p0) / s["travel"]
            t = np.clip((P[side][:3, 3] - p0) @ u, 0, s["travel"])
            dev.append(np.linalg.norm(P[side][:3, 3] - (p0 + t * u)))
    info["max_line_deviation_m"] = float(max(dev))
    INFO.write_text(json.dumps(info, indent=1))
    print(f"wrote {DST} and {INFO}")
    for side, s in info["slides"].items():
        print(f"  {side}: travel {s['travel'] * 1000:.1f} mm from {np.round(np.array(s['open_pos']) * 1000, 1)} to {np.round(np.array(s['closed_pos']) * 1000, 1)} mm (gripper base frame)")
    print(f"  inner gap {gap[0] * 1000:.1f} mm open -> {gap[-1] * 1000:.1f} mm closed; straight slide vs real pad arc: <= {info['max_line_deviation_m'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
