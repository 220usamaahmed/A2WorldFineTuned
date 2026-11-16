"""Mirror USD assets from NVIDIA's public Isaac asset bucket to local disk, together with everything they reference.

    python fetch_assets.py [--dest $A2W_DATA/assets] [--list assets.txt | PATH ...]

PATH is relative to the Isaac root (e.g. Props/YCB/Axis_Aligned/003_cracker_box.usd). A folder (trailing "/") fetches every
usd file below it. The dependency walk (sublayers, references, payloads, textures, MDL files) uses pxr, so run this under
Isaac Lab's python, e.g.  python fetch_assets.py ... --headless.  Files are stored under <dest>/Isaac/5.1/Isaac/ with the
same relative layout as the bucket, so the relative references inside the USD files keep working.
"""
import argparse
import posixpath
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ls_isaac_assets import BUCKET, ROOT, list_keys  # noqa: E402
import paths  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="*")
ap.add_argument("--list", type=Path, help="text file with one path per line (# comments allowed)")
ap.add_argument("--dest", type=Path, default=paths.ASSETS)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

from pxr import Sdf, UsdUtils  # noqa: E402

LOCAL = args.dest / "Isaac/5.1/Isaac"
USD_EXT = {".usd", ".usda", ".usdc", ".usdz"}


def download(rel: str) -> bool:
    """Fetch one file (skip if present). Returns False if the bucket has no such file."""
    dst = LOCAL / rel
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(f"{BUCKET}/{ROOT}{rel}", timeout=60).read()
            dst.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return False
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {rel}")


def dependencies(rel: str) -> list[str]:
    """Bucket-relative paths referenced by a local usd file."""
    layer_path = str(LOCAL / rel)
    sub, refs, payloads = UsdUtils.ExtractExternalReferences(layer_path)
    out = []
    for p in list(sub) + list(refs) + list(payloads):
        p = p.split("[")[0]  # strip "file.usdz[inner/path]"
        if not p or "://" in p or p.startswith("/") or p.startswith("op:"):
            if p:
                print("   (skipping non-relative reference)", p)
            continue
        out.append(posixpath.normpath(posixpath.join(posixpath.dirname(rel), p)))
    return out


_listed_dirs: dict[str, list[str]] = {}


def mdl_dependencies(rel: str) -> list[str]:
    """MDL files import sibling modules and name textures in their source text, which the USD walk cannot see."""
    here = posixpath.dirname(rel)
    if here not in _listed_dirs:  # modules are imported by name, so take every .mdl next to this one
        _listed_dirs[here] = [k for k, _ in list_keys(here + "/") if k.endswith(".mdl") and posixpath.dirname(k) == here]
    out = list(_listed_dirs[here])
    text = (LOCAL / rel).read_text(errors="ignore")
    for t in re.findall(r'"([^"\n]+\.(?:png|jpg|jpeg|exr|tga|bmp|hdr))"', text, re.I):
        out.append(posixpath.normpath(posixpath.join(here, t.lstrip("/"))))
    return out


def fetch_tree(roots: list[str]):
    seen, missing, todo = set(), [], list(roots)
    pool = ThreadPoolExecutor(8)
    while todo:
        batch = [p for p in dict.fromkeys(todo) if p not in seen]
        todo = []
        seen.update(batch)
        for rel, ok in zip(batch, pool.map(download, batch)):
            if not ok:
                missing.append(rel)
            elif rel.endswith(".mdl"):
                todo += mdl_dependencies(rel)
            elif Path(rel).suffix in USD_EXT:
                try:
                    todo += dependencies(rel)
                except Exception as e:  # noqa: BLE001
                    print("   dependency scan failed:", rel, e)
        print(f"  {len(seen)} files so far, {len(todo)} queued", flush=True)
    return seen, missing


roots = list(args.paths)
if args.list:
    roots += [l.split("#")[0].strip() for l in args.list.read_text().splitlines() if l.split("#")[0].strip()]
expanded = []
for r in roots:
    if r.endswith("/"):
        expanded += [k for k, _ in list_keys(r) if Path(k).suffix in USD_EXT and "/.thumbs/" not in k]
    else:
        expanded.append(r)
print(f"{len(expanded)} root assets -> {LOCAL}")
seen, missing = fetch_tree(expanded)
size = sum(f.stat().st_size for f in LOCAL.rglob("*") if f.is_file())
print(f"done: {len(seen)} files referenced, {len(missing)} not in the bucket, {size / 2**20:.0f} MB under {LOCAL}")
for m in missing[:40]:
    print("  missing:", m)
app.close()
