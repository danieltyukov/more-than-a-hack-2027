"""Download the Sportec Open DFL dataset (7 Bundesliga matches, CC-BY 4.0) into external/dfl.

    python scripts/get_dfl.py

Bassek, Rein, Weber and Memmert, "An integrated dataset of spatiotemporal and event data in elite
soccer", Scientific Data 12:195 (2025), doi:10.6084/m9.figshare.28196177.

It is the training data for the per-frame models in possession/learned.py: unlike the competition
match it comes with a per-frame ball status and owning team from the provider. About 2.6 GB.
Files already present with the right checksum are skipped.
"""

import hashlib
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "external", "dfl")
ARTICLE = "https://api.figshare.com/v2/articles/28196177/files?page_size=100"
# figshare's springernature host answers scripts with a bot challenge; this one does not
DOWNLOAD = "https://ndownloader.figshare.com/files/{}"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(f):
    path = os.path.join(OUT, f["name"])
    if os.path.isfile(path) and md5(path) == f["computed_md5"]:
        return None
    urllib.request.urlretrieve(DOWNLOAD.format(f["id"]), path + ".part")
    if md5(path + ".part") != f["computed_md5"]:
        return "checksum mismatch for {}".format(f["name"])
    os.replace(path + ".part", path)
    print("{:>5} MB  {}".format(f["size"] // 1_000_000, f["name"]), flush=True)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    with urllib.request.urlopen(ARTICLE) as response:
        files = json.load(response)
    # one connection is throttled to about 1 MB/s, so fetch the big files side by side
    with ThreadPoolExecutor(max_workers=8) as pool:
        errors = [e for e in pool.map(fetch, sorted(files, key=lambda f: -f["size"])) if e]
    if errors:
        sys.exit("\n".join(errors))
    print("ok, {} files in {}".format(len(files), OUT))


if __name__ == "__main__":
    main()
