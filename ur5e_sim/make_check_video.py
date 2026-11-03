"""Check video for recorded episodes: the 3 camera views on top, state plots with a moving cursor below.

    python make_check_video.py data/pilot/ep_000_success [more episode folders ...] --out check/

Per frame t: the three images (observation t), the phase and the command that is applied after it, the TCP pose,
finger angle and cube position. Plots: arm joints, TCP position, finger angle (measured and commanded), cube position.
Also writes <out>/overview.mp4 with every episode back to back (one title card each).
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

PHASES = ["warmup", "move", "descend", "close", "hold", "lift", "carry", "release", "retreat", "push", "idle"]
W = 1152
CAM = W // 3  # 384 px per camera
PLOT_H = 300
INFO_H = 96
FPS = 10
TITLES = {"ext1": "exterior camera 1", "ext2": "exterior camera 2", "wrist": "wrist camera"}


def plot_panel(d, T):
    """Pre-render the 4 plot panels once; returns (image, x pixel of step 0, x pixel of step T) per panel."""
    t = np.arange(T + 1)
    fig, axs = plt.subplots(1, 4, figsize=(W / 100, PLOT_H / 100), dpi=100)
    jp = np.degrees(d["joint_pos"])
    for i in range(6):
        axs[0].plot(t, jp[:, i], lw=1, label=f"q{i + 1}")
    axs[0].set_title("arm joints [deg]", fontsize=8)
    axs[0].legend(fontsize=5, ncol=3, loc="best")
    for i, c in enumerate("xyz"):
        axs[1].plot(t, d["tcp_pos"][:, i], lw=1.2, label=f"tcp {c}")
    axs[1].set_title("TCP position, arm-base frame [m]", fontsize=8)
    axs[1].legend(fontsize=6, ncol=3)
    axs[2].plot(t, d["finger_pos"].mean(1), lw=1.2, label="finger angle (measured)")
    axs[2].plot(t[:-1], d["action_grip"], lw=1, ls="--", label="commanded")
    axs[2].set_title("gripper [rad] (0 open, 0.75 closed)", fontsize=8)
    axs[2].legend(fontsize=6)
    for i, c in enumerate("xyz"):
        axs[3].plot(t, d["cube_pos"][:, i], lw=1.2, label=f"cube {c}")
    axs[3].axhline(0.0, color="gray", lw=0.6, ls=":")
    axs[3].set_title("cube position [m] (z<0: below table top)", fontsize=8)
    axs[3].legend(fontsize=6, ncol=3)
    for ax in axs:
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, T)
        ax.set_xlabel("control step (10 Hz)", fontsize=6)
        ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    spans = []
    for ax in axs:
        bb = ax.get_window_extent()
        x0 = bb.x0 + (0 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) * bb.width
        spans.append((bb.x0, bb.width, PLOT_H - bb.y1, PLOT_H - bb.y0))  # x0, width, y_top, y_bottom (image coords)
    plt.close(fig)
    return img, spans


def put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def episode_frames(folder: Path):
    d = dict(np.load(folder / "data.npz"))
    meta = json.loads(str(d["meta"]))
    T = len(d["action_tcp"])
    plots, spans = plot_panel(d, T)
    frames = []
    for t in range(T + 1):
        cams = []
        for c in ("ext1", "ext2", "wrist"):
            im = cv2.resize(d[f"rgb_{c}"][t], (CAM, CAM), interpolation=cv2.INTER_CUBIC)
            put(im, TITLES[c], (6, 18), 0.5, (255, 255, 0))
            cams.append(im)
        top = np.concatenate(cams, axis=1)

        plot = plots.copy()
        for x0, width, y0, y1 in spans:
            x = int(round(x0 + width * t / T))
            cv2.line(plot, (x, int(y0)), (x, int(y1)), (220, 30, 30), 1, cv2.LINE_AA)

        info = np.zeros((INFO_H, W, 3), np.uint8)
        phase = PHASES[int(d["phase"][min(t, T - 1)])]
        eul = np.degrees(Rotation.from_quat(d["tcp_quat"][t][[1, 2, 3, 0]]).as_euler("xyz"))
        cmd = d["action_tcp"][min(t, T - 1)]
        put(info, f"{folder.name}   step {t:3d}/{T}   t = {t / FPS:5.1f} s   phase: {phase}", (8, 20), 0.55, (255, 255, 0), 1)
        put(info, f"TCP pos [m]   x {d['tcp_pos'][t][0]:+.3f}  y {d['tcp_pos'][t][1]:+.3f}  z {d['tcp_pos'][t][2]:+.3f}"
                  f"      TCP rpy [deg]  {eul[0]:+7.1f} {eul[1]:+7.1f} {eul[2]:+7.1f}", (8, 44), 0.5)
        put(info, f"command (applied next) pos {cmd[0]:+.3f} {cmd[1]:+.3f} {cmd[2]:+.3f}   finger cmd "
                  f"{d['action_grip'][min(t, T - 1)]:.2f} rad   finger measured {d['finger_pos'][t].mean():.2f} rad", (8, 66), 0.5)
        fell = d["cube_pos"][t][2] < -0.04
        put(info, f"cube  x {d['cube_pos'][t][0]:+.3f}  y {d['cube_pos'][t][1]:+.3f}  z {d['cube_pos'][t][2]:+.3f}"
                  f"   {'FELL OFF THE TABLE' if fell else ''}      episode outcome: {meta['outcome']}",
            (8, 88), 0.5, (80, 80, 255) if fell else (255, 255, 255))
        frames.append(np.concatenate([top, plot, info], axis=0))
    return frames, meta


def title_card(text_lines, shape):
    img = np.zeros(shape, np.uint8)
    for i, line in enumerate(text_lines):
        put(img, line, (40, 80 + 40 * i), 0.9 if i == 0 else 0.6, (255, 255, 0) if i == 0 else (255, 255, 255), 2 if i == 0 else 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("check"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    allf = []
    for folder in a.episodes:
        frames, meta = episode_frames(folder)
        imageio.mimwrite(a.out / f"{folder.name}.mp4", frames, fps=FPS, macro_block_size=1)
        card = title_card(
            [folder.name, f"intended: {meta['kind']}   outcome: {meta['outcome']}",
             f"steps: {meta['steps']}   cube start (x, y) = ({meta['cube_xy'][0]:.3f}, {meta['cube_xy'][1]:.3f})   max lift {meta['max_cube_lift']:.3f} m",
             f"fell at step: {meta['fell_step']}   cube end = {np.round(meta['cube_end'], 3).tolist()}"],
            frames[0].shape,
        )
        allf += [card] * (2 * FPS) + frames
        print("wrote", a.out / f"{folder.name}.mp4", f"({len(frames)} frames)", flush=True)
    imageio.mimwrite(a.out / "overview.mp4", allf, fps=FPS, macro_block_size=1)
    print("wrote", a.out / "overview.mp4")


if __name__ == "__main__":
    main()
