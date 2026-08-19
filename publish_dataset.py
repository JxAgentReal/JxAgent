#!/usr/bin/env python3
"""Publish the prepared JxAgent dataset to a PRIVATE Hugging Face dataset repo.

The MI300X instance downloads this prepared dataset (mi300x/download_dataset.sh)
instead of rebuilding from raw sources.

UPLOAD POLICY (audit patch 2026-08-16): explicit ALLOW-LIST, not an ignore
list. Only intended production artifacts are ever uploaded:

    final/**        train/validation JSONL + all metadata + SHA256SUMS
    images/**       the dataset images themselves
    manifests/**    per-source build manifests (intentional release artifacts)

Everything else — state/, .tmp/, *.part, .venv/, __pycache__/, smoke dirs,
audit scratch, logs, caches — is excluded BY CONSTRUCTION (never selected).

The exact file count and total bytes are printed BEFORE anything is sent.
--dry-run lists what would be uploaded and exits without uploading.

Usage:
  python publish_dataset.py --dataset-root ./JxAgentData --repo <user>/JxAgentData
  python publish_dataset.py --dataset-root ./JxAgentData --repo <user>/JxAgentData --dry-run
"""
from __future__ import annotations

import argparse
import os

# Only these top-level dataset-root entries may ever be published.
ALLOW_ROOTS = ("final", "images", "manifests")
# Defense in depth: even inside allowed roots, never ship these.
EXCLUDE_NAMES = {"__pycache__", ".tmp", ".venv", "state", ".git", ".cache",
                 ".audit", "rlvr"}
EXCLUDE_SUFFIXES = (".part", ".tmp", ".lock", ".pyc")


def select_publication_files(root: str) -> list[str]:
    """Deterministic (sorted) list of files allowed to be published."""
    selected = []
    for entry in sorted(os.listdir(root)):
        if entry not in ALLOW_ROOTS:
            continue
        top = os.path.join(root, entry)
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames
                           if d not in EXCLUDE_NAMES and not d.endswith("__pycache__")]
            for fn in sorted(filenames):
                if fn.endswith(EXCLUDE_SUFFIXES) or fn == "build.lock":
                    continue
                selected.append(os.path.relpath(os.path.join(dirpath, fn), root)
                                .replace(os.sep, "/"))
    return sorted(selected)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo", required=True, help="private HF dataset repo id")
    p.add_argument("--dry-run", action="store_true",
                   help="list what WOULD be uploaded, upload nothing")
    args = p.parse_args()

    root = os.path.abspath(args.dataset_root)
    for required in ("final/train.jsonl", "final/validation.jsonl",
                     "final/manifest.json", "final/SHA256SUMS"):
        if not os.path.exists(os.path.join(root, required)):
            raise SystemExit(f"missing {required}; dataset not finalized "
                             f"(or built with a pre-hash builder)")

    files = select_publication_files(root)
    total_bytes = sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"[publish] selected {len(files)} files, "
          f"{total_bytes / (1 << 30):.2f} GiB from {root}")
    if len(files) > 20:
        preview = files[:10] + ["..."] + files[-5:]
    else:
        preview = files
    for f in preview:
        print(f"  {f}")
    if args.dry_run:
        print("[publish] dry run — nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    repo_url = api.create_repo(args.repo, repo_type="dataset", private=True,
                               exist_ok=True)
    print(f"[publish] {root} -> {repo_url}")

    # allow_patterns implements the allow-list server-side; the file list
    # printed above is exactly what matches it (verified by tests)
    api.upload_large_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=root,
        allow_patterns=[f"{r}/**" for r in ALLOW_ROOTS],
        ignore_patterns=[f"**/{n}/**" for n in EXCLUDE_NAMES]
        + [f"**{s}" for s in EXCLUDE_SUFFIXES],
    )
    print("[publish] done. On the MI300X: JXAGENT_DATASET_REPO=" + args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
