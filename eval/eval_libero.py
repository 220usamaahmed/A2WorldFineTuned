"""Open-loop A2World-sim rollouts on real LIBERO demos: error vs ground truth, timing, comparison videos.

Run from A2World/world_model with its .venv active and CUDA_VISIBLE_DEVICES=0.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from skimage.metrics import structural_similarity

from a2world import rollout

CHUNK = 20
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
WM = ROOT / "A2World" / "world_model"


def upright(frames):  # raw LIBERO obs are rotated 180 degrees
    return np.ascontiguousarray(frames[:, ::-1, ::-1])


def resize(frames, size=256):
    return np.stack([rollout.resize_frame(f, size, size) for f in frames])


class TimedPipeline:
    def __init__(self, pipe):
        self.pipe, self.times = pipe, []

    def __call__(self, **kw):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.pipe(**kw)
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - t0)
        return out


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--checkpoint", default=str(WM / "a2world_ckpt/a2world-libero.pt"))
    p.add_argument("--demos", type=int, default=10)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "results" / "libero_eval"))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = a.chunks * CHUNK

    sys.argv = ["rollout", "--checkpoint", a.checkpoint, "--input", "x", "y", "--actions", "x", "--autoregressive",
                "--action-format", "libero-servo", "--variant", "libero"]
    args = rollout.parse_args()
    pipe = TimedPipeline(rollout.setup_pipeline(args))

    captured = {}
    rollout.save_image_or_video = lambda vid, path, fps=10: captured.__setitem__("vid", vid)

    per_demo = []
    with h5py.File(a.hdf5) as h:
        names = sorted(h["data"], key=lambda n: int(n.split("_")[-1]))[: a.demos]
        for name in names:
            d = h["data"][name]
            gt = [resize(upright(np.asarray(d[f"obs/{v}"][: steps + 1]))) for v in ("agentview_rgb", "eye_in_hand_rgb")]
            act = np.zeros((steps, 2, 7), np.float32)
            act[:, 0] = np.asarray(d["actions"][:steps])
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for i, g in enumerate(gt):
                    paths.append(f"{tmp}/v{i}.png")
                    cv2.imwrite(paths[-1], cv2.cvtColor(g[0], cv2.COLOR_RGB2BGR))
                np.savez(f"{tmp}/a.npz", actions=act)
                args.input, args.actions, args.output = paths, f"{tmp}/a.npz", f"{tmp}/o.mp4"
                n0 = len(pipe.times)
                rollout.generate_rollout(args, pipe)
            v = captured["vid"][0].float().cpu()  # [C,T,H,2W] in [-1,1]
            pred = ((v.permute(1, 2, 3, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)  # [T,H,2W,C]
            pred = [pred[..., :256, :], pred[..., 256:, :]]
            m = {k: np.zeros((2, steps)) for k in ("mse", "mae", "ssim", "static_mse")}
            for vi in range(2):
                for t in range(1, steps + 1):
                    g, pr, s = gt[vi][t].astype(np.float64), pred[vi][t].astype(np.float64), gt[vi][0].astype(np.float64)
                    m["mse"][vi, t - 1] = ((g - pr) ** 2).mean()
                    m["mae"][vi, t - 1] = np.abs(g - pr).mean()
                    m["static_mse"][vi, t - 1] = ((g - s) ** 2).mean()
                    m["ssim"][vi, t - 1] = structural_similarity(g, pr, channel_axis=2, data_range=255)
            per_demo.append({"demo": name, "chunk_times": pipe.times[n0:], **{k: x.tolist() for k, x in m.items()}})
            print(name, "PSNR(mean over steps) agent/wrist:",
                  [round(float(np.mean(10 * np.log10(255**2 / m["mse"][vi]))), 2) for vi in range(2)],
                  "chunk s:", [round(t, 1) for t in pipe.times[n0:]], flush=True)
            if len(per_demo) <= 3:
                rows = []
                for t in range(steps + 1):
                    top = np.concatenate([label(gt[0][t], f"real agentview t={t}"), label(gt[1][t], "real wrist")], 1)
                    bot = np.concatenate([label(pred[0][t], f"A2World agentview t={t}"), label(pred[1][t], "A2World wrist")], 1)
                    rows.append(np.concatenate([top, bot], 0))
                imageio.mimwrite(out / f"compare_{name}.mp4", rows, fps=10, macro_block_size=1)
                idx = [0, 20, 40, 60, steps]
                cols = [np.concatenate([gt[0][t], pred[0][t], gt[1][t], pred[1][t]], 0) for t in idx]
                cv2.imwrite(str(out / f"frames_{name}.png"),
                            cv2.cvtColor(np.concatenate(cols, 1), cv2.COLOR_RGB2BGR))

    json.dump(per_demo, open(out / "metrics.json", "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
