"""Fine-tuned A2World-sim (libero variant, 2 views, history) on the held-out UR5e episodes: open-loop rollout vs real frames.

Run from A2World/world_model with its .venv active, one visible GPU:
    CUDA_VISIBLE_DEVICES=0 python ../../eval/eval_ur5e_ft.py --checkpoint <exported .pt> --episodes ../../ur5e_sim/data/pilot --out ../../results/ur5e_finetuned/iter3600

Same protocol as eval_ur5e.py (frame 0 of each view + recorded actions -> autoregressive 20-step chunks, compared with the real frames),
but with the training-time action format: raw servo-style actions from ur5e_sim/convert_to_train.py (loaded with --action-format libero-servo).
Views: ext1 = agent view (camera embedding 2), wrist = eye-in-hand (embedding 0).
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from skimage.metrics import structural_similarity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root
sys.path.insert(0, str(ROOT / "ur5e_sim"))
import convert_to_train as conv  # noqa: E402  (raw_actions: same conversion as the training data)
from a2world import rollout  # noqa: E402

CHUNK = 20
VIEWS = ("ext1", "wrist")


def red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h < 8) | (h > 170)) & (s > 120) & (v > 70)


def red_stats(img):
    m = red_mask(img)
    n = int(m.sum())
    if n < 3:
        return n, None
    ys, xs = np.nonzero(m)
    return n, (float(xs.mean()), float(ys.mean()))


def label(img, text, color=(255, 255, 0)):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--max-chunks", type=int, default=100)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = ["rollout", "--checkpoint", a.checkpoint, "--variant", "libero", "--input", "x", "y", "--actions", "x",
                "--action-format", "libero-servo", "--autoregressive"]
    args = rollout.parse_args()
    pipe = rollout.setup_pipeline(args)
    times = []

    class Timed:
        def __call__(self, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = pipe(**kw)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            return r

    captured = {}
    rollout.save_image_or_video = lambda vid, path, fps=10: captured.__setitem__("vid", vid)

    results = []
    folders = sorted(f for f in Path(a.episodes).glob("ep_*") if (f / "data.npz").exists())
    if a.only:
        folders = [f for f in folders if f.name in a.only]
    for folder in folders:
        d = dict(np.load(folder / "data.npz"))
        meta = json.loads(str(d["meta"]))
        n_chunks = min(len(d["action_tcp"]) // CHUNK, a.max_chunks)
        steps = n_chunks * CHUNK
        gt = {v: d[f"rgb_{v}"][: steps + 1] for v in VIEWS}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            paths = []
            for v in VIEWS:
                paths.append(f"{tmp}/{v}.png")
                cv2.imwrite(paths[-1], cv2.cvtColor(gt[v][0], cv2.COLOR_RGB2BGR))
            np.savez(f"{tmp}/actions.npz", actions=conv.raw_actions(d)[:steps])
            args.input, args.actions, args.output = paths, f"{tmp}/actions.npz", f"{tmp}/o.mp4"
            n0 = len(times)
            rollout.generate_rollout(args, Timed())
        v = captured["vid"][0].float().cpu()  # [C, T, H, 2W] in [-1, 1]
        pred_all = ((v.permute(1, 2, 3, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        pred = {name: pred_all[:, :, i * 256:(i + 1) * 256] for i, name in enumerate(VIEWS)}
        assert pred["ext1"].shape[0] == steps + 1, pred["ext1"].shape

        m = {k: {v_: np.zeros(steps) for v_ in VIEWS} for k in ("mse", "mae", "ssim", "static_mse")}
        red = {v_: {"real_area": [], "pred_area": [], "real_c": [], "pred_c": []} for v_ in VIEWS}
        for v_ in VIEWS:
            for t in range(1, steps + 1):
                g, pr, s0 = (x.astype(np.float64) for x in (gt[v_][t], pred[v_][t], gt[v_][0]))
                m["mse"][v_][t - 1] = ((g - pr) ** 2).mean()
                m["mae"][v_][t - 1] = np.abs(g - pr).mean()
                m["static_mse"][v_][t - 1] = ((g - s0) ** 2).mean()
                m["ssim"][v_][t - 1] = structural_similarity(g, pr, channel_axis=2, data_range=255)
            for t in range(steps + 1):
                for key, img in (("real", gt[v_][t]), ("pred", pred[v_][t])):
                    n, c = red_stats(img)
                    red[v_][f"{key}_area"].append(n)
                    red[v_][f"{key}_c"].append(c)
        results.append({"episode": folder.name, "kind": meta["kind"], "outcome": meta["outcome"], "steps": steps,
                        "fell_step": meta["fell_step"], "chunk_times": times[n0:],
                        **{k: {v_: x.tolist() for v_, x in dd.items()} for k, dd in m.items()}, "red": red})
        psnr = {v_: float(np.mean(10 * np.log10(255 ** 2 / m["mse"][v_]))) for v_ in VIEWS}
        spsnr = {v_: float(np.mean(10 * np.log10(255 ** 2 / np.maximum(m["static_mse"][v_], 1e-6)))) for v_ in VIEWS}
        print(f"{folder.name:26s} steps={steps:3d} PSNR " + " ".join(f"{v_}={psnr[v_]:.1f}(static {spsnr[v_]:.1f})" for v_ in VIEWS)
              + f" | chunk s={np.mean(times[n0:]):.1f}", flush=True)

        rows = []
        for t in range(steps + 1):
            top = np.concatenate([label(gt[v_][t], f"real {v_} t={t}") for v_ in VIEWS], 1)
            bot = np.concatenate([label(pred[v_][t], f"A2World fine-tuned {v_}", (255, 160, 0)) for v_ in VIEWS], 1)
            band = np.zeros((22, top.shape[1], 3), np.uint8)
            tag = f"{folder.name}  outcome: {meta['outcome']}  (fell at step {meta['fell_step']})  chunks of {CHUNK}"
            cv2.putText(band, tag, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(np.concatenate([band, top, bot], 0))
        imageio.mimwrite(out / f"compare_{folder.name}.mp4", rows, fps=10, macro_block_size=1)
        idx = sorted({0, steps // 4, steps // 2, 3 * steps // 4, steps})
        cols = [np.concatenate([np.concatenate([gt[v_][t] for v_ in VIEWS], 0), np.concatenate([pred[v_][t] for v_ in VIEWS], 0)], 0) for t in idx]
        cv2.imwrite(str(out / f"frames_{folder.name}.png"), cv2.cvtColor(np.concatenate(cols, 1), cv2.COLOR_RGB2BGR))
        json.dump(results, open(out / "metrics.json", "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
