#!/usr/bin/env python3
"""Fetch a pinned HF repo revision with aria2c, then verify every LFS file by sha256.
usage: hf_fetch.py REPO REVISION DEST [glob ...]"""
import fnmatch, hashlib, json, os, subprocess, sys, urllib.request

repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
globs = sys.argv[4:] or ["*"]
os.makedirs(dest, exist_ok=True)
tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"))
files = [f for f in tree if f["type"] == "file" and any(fnmatch.fnmatch(f["path"], g) for g in globs)]
lst = os.path.join(dest, ".aria2-input.txt")
with open(lst, "w") as fh:
    for f in files:
        out = os.path.join(dest, f["path"])
        if os.path.exists(out) and os.path.getsize(out) == f["size"]:
            continue
        fh.write(f"https://huggingface.co/{repo}/resolve/{rev}/{f['path']}\n  out={f['path']}\n  dir={dest}\n")
if os.path.getsize(lst):
    subprocess.run(["aria2c", "-i", lst, "-x", "16", "-s", "16", "-j", "3", "-k", "8M", "--file-allocation=none",
                    "--max-tries=20", "--retry-wait=5", "--console-log-level=warn", "--summary-interval=60"], check=True)
manifest = []
for f in files:
    out = os.path.join(dest, f["path"])
    size = os.path.getsize(out)
    assert size == f["size"], (out, size, f["size"])
    oid = (f.get("lfs") or {}).get("oid")
    if oid:
        h = hashlib.sha256()
        with open(out, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == oid, (out, h.hexdigest(), oid)
    manifest.append({"path": f["path"], "size": size, "sha256": oid})
json.dump({"repo": repo, "revision": rev, "verified": True, "files": manifest}, open(os.path.join(dest, "READY.json"), "w"), indent=1)
print("READY", repo, rev, len(manifest), "files")
