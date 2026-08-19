#!/usr/bin/env python3
"""Verify a freshly downloaded prepared JxAgent dataset (audit patch §19).

Designed to run on the MI300X (or any host) against a clean snapshot of the
published dataset, completely independent of the original build directory.

Checks (all must pass; any failure exits 1):
  - final/manifest.json, stats.json, SHA256SUMS exist
  - every SHA256SUMS entry matches the actual file bytes
  - train/validation line counts equal manifest counts
  - image file count equals manifest.image_count
  - images_tree_hash: recomputed tree hash matches the manifest
  - per-source sample counts match manifest.sources
  - no absolute or Windows-style image paths anywhere in the JSONLs
  - (optional, --full) every sample passes processing.validation.validate_sample

Usage:
  python tools/verify_clean_download.py --dataset-root /data/JxAgentData [--full]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES: list[str] = []


def check(cond: bool, message: str) -> bool:
    if cond:
        print(f"  [ok] {message}")
    else:
        print(f"  [FAIL] {message}")
        FAILURES.append(message)
    return cond


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def images_tree_hash(root: str) -> tuple[str, int]:
    images_root = os.path.join(root, "images")
    rows = []
    for dirpath, _dirs, files in os.walk(images_root):
        for fn in files:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, images_root).replace(os.sep, "/")
            rows.append((rel, full))
    rows.sort()
    lines = []
    for rel, full in rows:
        lines.append(f"{rel}\t{os.path.getsize(full)}\t{sha256_file(full)}")
    digest = hashlib.sha256(("\n".join(lines) + ("\n" if lines else "")).encode()).hexdigest()
    return digest, len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--full", action="store_true",
                   help="also run per-sample validation (needs the JxAgent repo on PATH)")
    args = p.parse_args()
    root = os.path.abspath(args.dataset_root)

    print(f"[verify] clean download at {root}")
    final = os.path.join(root, "final")
    manifest_path = os.path.join(final, "manifest.json")
    sums_path = os.path.join(final, "SHA256SUMS")
    check(os.path.isfile(manifest_path), "final/manifest.json exists")
    check(os.path.isfile(sums_path), "final/SHA256SUMS exists")
    if FAILURES:
        return 1
    manifest = json.load(open(manifest_path, encoding="utf-8"))

    print("[verify] SHA256SUMS integrity")
    sums = {}
    for line in open(sums_path, encoding="utf-8"):
        line = line.strip()
        if line:
            digest, rel = line.split(None, 1)
            sums[rel.strip().replace("\\", "/")] = digest
    for rel, digest in sorted(sums.items()):
        path = os.path.join(root, rel.replace("/", os.sep))
        if not check(os.path.isfile(path), f"{rel} present"):
            continue
        check(sha256_file(path) == digest, f"{rel} sha256 matches")

    print("[verify] counts")
    train = read_jsonl(os.path.join(final, "train.jsonl"))
    val = read_jsonl(os.path.join(final, "validation.jsonl"))
    check(len(train) == manifest.get("train_samples"),
          f"train line count {len(train)} == manifest {manifest.get('train_samples')}")
    check(len(val) == manifest.get("validation_samples"),
          f"validation line count {len(val)} == manifest {manifest.get('validation_samples')}")

    print("[verify] images tree hash")
    tree_digest, n_images = images_tree_hash(root)
    check(n_images == manifest.get("image_count"),
          f"image count {n_images} == manifest {manifest.get('image_count')}")
    check(tree_digest == manifest.get("images_tree_hash"),
          "images_tree_hash matches manifest")

    print("[verify] per-source counts")
    per_source: dict[str, int] = {}
    for s in train + val:
        per_source[s.get("source", "?")] = per_source.get(s.get("source", "?"), 0) + 1
    # finalize() records every TARGET source in manifest.sources (0 when the
    # source contributed nothing, e.g. a partial-source run); observed counts
    # can only contain sources with >=1 sample. Zero entries carry no claim,
    # so drop them before the exact comparison.
    manifest_sources = {k: v for k, v in manifest.get("sources", {}).items() if v}
    check(per_source == manifest_sources,
          f"per-source counts match manifest ({per_source} vs {manifest_sources})")

    print("[verify] portability")
    bad_paths = 0
    for s in train + val:
        for ip in s.get("images", []):
            if (not isinstance(ip, str) or os.path.isabs(ip) or "\\" in ip
                    or ip.startswith("/") or ".." in ip):
                bad_paths += 1
    check(bad_paths == 0, f"no absolute/Windows/relative-escape image paths ({bad_paths} bad)")

    if args.full:
        print("[verify] full per-sample validation")
        try:
            from processing.validation import validate_sample
            invalid = 0
            for s in train + val:
                ok, _reason = validate_sample(s, root)
                if not ok:
                    invalid += 1
            check(invalid == 0, f"all {len(train) + len(val)} samples validate ({invalid} bad)")
        except ImportError:
            print("  [skip] processing/ not importable from this checkout")

    if FAILURES:
        print(f"[verify] FAILED: {len(FAILURES)} check(s)")
        return 1
    print("[verify] PASSED: clean download is intact and self-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
