"""Recorded UR5e episodes (data.npz) -> A2World's LIBERO training layout (docs/DATA.md of the A2World repo).

    python convert_to_train.py --episodes $A2W_DATA/train --out $A2W_DATA/train_processed [--also-ext2]

Output
    clips/agentview_<ep>_clip_<k>.mp4      ext1 (the fixed exterior camera), upright as rendered, 256x256, 10 fps label
    clips/eye_in_hand_<ep>_clip_<k>.mp4    wrist camera
    metadata/agentview_<ep>_clip_<k>.npz   actions [64,2,7], end_position [64,2,3], end_orientation [64,2,4] (xyzw),
                                            effector_position [64,2,2], start_frame, clip_len, fps
    dataset.json                           [{"caption", "media_path"}] for the agentview clips
    action_stats.json                      percentiles of the raw actions (compare with LIBERO's)

Action convention: the training loader multiplies raw action[:6] by 0.05 and
maps the gripper (1 - g) / 2. Raw values are normalised base-frame deltas of the MEASURED TCP pose between consecutive frames:
    raw_pos = clip(delta_pos / POS_SCALE, -1, 1);  raw_rot = clip(delta_rotvec / ROT_SCALE, -1, 1)
    raw_grip = +1 (close) / -1 (open) from the commanded finger angle
action[t] is what moves the robot from frame t to frame t+1; the last frame of an episode has no action (zeros).
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

POS_SCALE = 0.018  # m per unit raw action: our smoothstep motion peaks at ~1.8 cm per step, so nothing clips (LIBERO: ~14 mm at its p95)
ROT_SCALE = 0.1  # rad per unit raw action: puts our (yaw-only) rotations at LIBERO-like magnitudes
GRIP_THRESHOLD = 0.375  # rad: commanded finger angle above this = closing command
CLIP, STRIDE, FPS = 64, 16, 10
CAPTION = "move the red cube on the table with the robot arm"


def raw_actions(d) -> np.ndarray:
    """[T+1, 7] raw servo-style actions (last row zero)."""
    pos = d["tcp_pos"].astype(np.float64)
    rot = Rotation.from_quat(d["tcp_quat"][:, [1, 2, 3, 0]].astype(np.float64))  # wxyz -> xyzw
    dpos = pos[1:] - pos[:-1]
    drot = (rot[1:] * rot[:-1].inv()).as_rotvec()  # rotation between frames, expressed in the base frame
    a = np.zeros((len(pos), 7), np.float32)
    a[:-1, :3] = np.clip(dpos / POS_SCALE, -1, 1)
    a[:-1, 3:6] = np.clip(drot / ROT_SCALE, -1, 1)
    a[:-1, 6] = np.where(d["action_grip"] > GRIP_THRESHOLD, 1.0, -1.0)
    a[-1, 6] = a[-2, 6]
    return a


def clip_metadata(d, a, start):
    sl = slice(start, start + CLIP)
    n = CLIP
    actions = np.zeros((n, 2, 7), np.float32)
    actions[:, 0] = a[sl]
    pos = np.zeros((n, 2, 3), np.float32)
    pos[:, 0] = d["tcp_pos"][sl]
    ori = np.zeros((n, 2, 4), np.float32)
    ori[:, 0] = d["tcp_quat"][sl][:, [1, 2, 3, 0]]  # xyzw
    ori[:, 1, 3] = 1.0
    width = 0.04 * (1.0 - np.clip(d["finger_pos"][sl].mean(1) / 0.8, 0, 1))  # LIBERO-like finger positions [m]
    grip = np.zeros((n, 2, 2), np.float32)
    grip[:, 0, 0], grip[:, 0, 1] = width, -width
    return dict(end_position=pos, end_orientation=ori, effector_position=grip, actions=actions,
                start_frame=np.asarray(start), clip_len=np.asarray(CLIP), fps=np.asarray(FPS))


def write_clip_set(out: Path, stem: str, d, a, agent_key: str, entries: list):
    n_frames = len(d["tcp_pos"])
    for k, start in enumerate(range(0, n_frames - CLIP + 1, STRIDE)):
        name = f"{stem}_clip_{k:05d}"
        sl = slice(start, start + CLIP)
        imageio.mimwrite(out / "clips" / f"agentview_{name}.mp4", list(d[agent_key][sl]), fps=FPS, macro_block_size=1)
        imageio.mimwrite(out / "clips" / f"eye_in_hand_{name}.mp4", list(d["rgb_wrist"][sl]), fps=FPS, macro_block_size=1)
        np.savez_compressed(out / "metadata" / f"agentview_{name}.npz", **clip_metadata(d, a, start))
        entries.append({"caption": CAPTION, "media_path": f"clips/agentview_{name}.mp4"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-ext2", action="store_true", help="also write each episode with ext2 as the agent view")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    (a.out / "metadata").mkdir(parents=True, exist_ok=True)
    entries, all_raw = [], []
    folders = sorted(f for f in a.episodes.glob("ep_*") if (f / "data.npz").exists())
    if a.limit:
        folders = folders[: a.limit]
    for i, folder in enumerate(folders):
        d = dict(np.load(folder / "data.npz"))
        if len(d["tcp_pos"]) < CLIP:
            print("skip (too short):", folder.name)
            continue
        raw = raw_actions(d)
        all_raw.append(raw[:-1])
        write_clip_set(a.out, folder.name, d, raw, "rgb_ext1", entries)
        if a.also_ext2:
            write_clip_set(a.out, folder.name + "_cam2", d, raw, "rgb_ext2", entries)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(folders)} episodes, {len(entries)} clips", flush=True)
    (a.out / "dataset.json").write_text(json.dumps(entries, indent=2) + "\n")
    raw = np.concatenate(all_raw)
    stats = {"n_steps": int(len(raw)),
             "abs_percentiles_50_95_100": np.percentile(np.abs(raw), [50, 95, 100], axis=0).round(3).tolist(),
             "fraction_clipped_pos": float((np.abs(raw[:, :3]) >= 1).mean()),
             "fraction_clipped_rot": float((np.abs(raw[:, 3:6]) >= 1).mean()),
             "libero_reference_abs_percentiles_50_95_100": [[0.22, 0.15, 0.42, 0.03, 0.03, 0.01, 1.0],
                                                           [0.94, 0.78, 0.94, 0.10, 0.12, 0.08, 1.0],
                                                           [0.94, 0.92, 0.94, 0.17, 0.22, 0.21, 1.0]]}
    (a.out / "action_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"{len(folders)} episodes -> {len(entries)} clips in {a.out}")


if __name__ == "__main__":
    main()
