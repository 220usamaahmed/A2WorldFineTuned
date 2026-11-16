"""List files under a folder of NVIDIA's public Isaac asset bucket (no Isaac Sim needed).

    python ls_isaac_assets.py Props/Food [--all]     # hides .thumbs, textures and materials unless --all
"""
import argparse
import re
import urllib.parse
import urllib.request

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ROOT = "Assets/Isaac/5.1/Isaac/"
HIDE = re.compile(r"/\.thumbs/|/(Materials|materials|textures|Textures)/|\.(png|jpg|jpeg|mdl|exr|last_generated)$")


def list_keys(prefix: str):
    """Yield (key relative to the Isaac root, size in bytes) for every object under prefix."""
    token = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={ROOT}{prefix}" + (f"&continuation-token={urllib.parse.quote(token, safe='')}" if token else "")
        xml = urllib.request.urlopen(url).read().decode()
        for key, size in re.findall(r"<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>", xml):
            yield key[len(ROOT):], int(size)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            return
        token = m.group(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    prefix = a.prefix.rstrip("/") + "/"
    for key, size in list_keys(prefix):
        if a.all or not HIDE.search("/" + key):
            print(f"{size / 1024:10.0f} KB  {key}")
