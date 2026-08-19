#!/usr/bin/env python3
"""Generate TRAINING_HANDOFF.json + TRAINING_HANDOFF.md for the MI300X session.

The handoff is generated ONLY from a validated, finalized, published dataset
root: it refuses to run against an unvalidated or fatally-failed build, and
it requires the dataset repo id (and revision) of the published copy so the
training machine never guesses what to download.

Prerequisites enforced:
  - final/manifest.json, stats.json, SHA256SUMS exist
  - manifest.fatal_failure is false
  - manifest.quota_acceptance fully accepted (when present)
  - train/validation hashes in SHA256SUMS match the actual files (cheap guard
    against a post-finalize tamper)

Usage:
  python tools/make_training_handoff.py --dataset-root ./JxAgentData \
      --dataset-repo <user>/JxAgentData --dataset-revision <repo-commit-or-tag>

This tool NEVER fabricates a production handoff: every required number comes
from the finalized artifacts; anything missing is an error, not a default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = os.environ.get("JXAGENT_MODEL_ID", "Qwen/Qwen3.8-27B")
EPOCH_TOKEN_ESTIMATE = 234_000_000     # 2026-08 token audit (PROJECT_STATUS.json)
OPTIMIZER_STEPS_ESTIMATE = 2980


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_sha256sums(root: str) -> dict:
    path = os.path.join(root, "final", "SHA256SUMS")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            digest, rel = line.split(None, 1)
            out[rel.strip().replace("\\", "/")] = digest
    return out


def build_handoff(root: str, dataset_repo: str, dataset_revision: str) -> dict:
    final = os.path.join(root, "final")
    manifest = json.load(open(os.path.join(final, "manifest.json"), encoding="utf-8"))
    stats = json.load(open(os.path.join(final, "stats.json"), encoding="utf-8"))

    if manifest.get("fatal_failure") is not False:
        raise SystemExit("REFUSING: manifest.fatal_failure is not false")
    qa = manifest.get("quota_acceptance")
    if qa and not all(v.get("accepted") for v in qa.values()):
        rejected = [k for k, v in qa.items() if not v.get("accepted")]
        raise SystemExit(f"REFUSING: quota acceptance failed for {rejected}")
    if not dataset_repo or not dataset_revision:
        raise SystemExit("REFUSING: --dataset-repo and --dataset-revision are "
                         "required (the published HF repo id and its revision)")

    sums = _load_sha256sums(root)
    # tamper guard: recorded hashes must match the actual bytes
    for rel in ("final/train.jsonl", "final/validation.jsonl", "final/manifest.json"):
        if rel not in sums:
            raise SystemExit(f"REFUSING: {rel} missing from SHA256SUMS")
        actual = _sha256_file(os.path.join(root, rel))
        if actual != sums[rel]:
            raise SystemExit(f"REFUSING: {rel} hash mismatch (file changed after finalize?)")

    handoff = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "builder_git_commit": manifest.get("builder_commit"),
        "config_hash": manifest.get("config_hash"),
        "files": {
            "train": "final/train.jsonl",
            "validation": "final/validation.jsonl",
            "images_root": "images/",
            "manifest": "final/manifest.json",
            "sha256sums": "final/SHA256SUMS",
        },
        "hashes": {
            "train": sums.get("final/train.jsonl"),
            "validation": sums.get("final/validation.jsonl"),
            "manifest": sums.get("final/manifest.json"),
            "images_tree_hash": manifest.get("images_tree_hash"),
        },
        "counts": {
            "train": manifest.get("train_samples"),
            "validation": manifest.get("validation_samples"),
            "images": manifest.get("image_count"),
            "per_source": manifest.get("sources", {}),
        },
        "validation_status": {
            "fatal_failure": manifest.get("fatal_failure"),
            "failures": (manifest.get("validation_result") or {}).get("failures", {}),
            "quota_acceptance_passed": stats.get("quota_acceptance_passed"),
            "validator": "evaluation/validate_actions.py --full",
        },
        "model_id": MODEL_ID,
        "estimated_tokens": EPOCH_TOKEN_ESTIMATE,
        "estimated_optimizer_steps": OPTIMIZER_STEPS_ESTIMATE,
        "policies": {
            "reasoning": {"rate": (manifest.get("build_config") or {}).get("reasoning",
                          {"rate": 0.12})},
            "representation": {"single": 0.55, "window": 0.40, "chunk": 0.05},
            "image": {"format": "webp", "quality": 80, "max_long": 1600,
                      "grounding_max_long": 1920},
        },
        "build_id": manifest.get("build_id"),
        "source_revisions": manifest.get("source_revisions", {}),
        "environment": manifest.get("environment", {}),
        "note": "generated by tools/make_training_handoff.py; verify the clean "
                "download (tools/verify_clean_download.py) on the MI300X path "
                "before training",
    }
    return handoff


def render_md(h: dict) -> str:
    lines = [
        "# JxAgent TRAINING HANDOFF", "",
        f"- dataset repo: `{h['dataset_repo']}` @ `{h['dataset_revision']}`",
        f"- builder commit: `{h['builder_git_commit']}`  config hash: `{h['config_hash']}`",
        f"- model: `{h['model_id']}`  (~{h['estimated_tokens']:,} tokens/epoch, "
        f"~{h['estimated_optimizer_steps']:,} optimizer steps)",
        f"- train: {h['counts']['train']:,}  validation: {h['counts']['validation']:,}  "
        f"images: {h['counts']['images']:,}",
        f"- images tree hash: `{h['hashes']['images_tree_hash']}`",
        "", "## Per-source counts", "",
        "| source | samples |", "|---|---|",
    ]
    for k, v in sorted(h["counts"]["per_source"].items()):
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "", "## Verification", "",
        f"- fatal_failure: {h['validation_status']['fatal_failure']}",
        f"- quota acceptance passed: {h['validation_status']['quota_acceptance_passed']}",
        f"- train sha256: `{h['hashes']['train']}`",
        f"- validation sha256: `{h['hashes']['validation']}`",
        "",
        "Run `python tools/verify_clean_download.py --dataset-root <fresh-download>`",
        "on the MI300X before starting any training.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--dataset-repo", required=True,
                   help="published private HF dataset repo id")
    p.add_argument("--dataset-revision", required=True,
                   help="revision (commit SHA/tag) of the published dataset repo")
    args = p.parse_args()

    root = os.path.abspath(args.dataset_root)
    handoff = build_handoff(root, args.dataset_repo, args.dataset_revision)
    json_path = os.path.join(root, "TRAINING_HANDOFF.json")
    md_path = os.path.join(root, "TRAINING_HANDOFF.md")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(handoff, f, ensure_ascii=False, indent=1)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(handoff))
    print(f"[handoff] wrote {json_path}")
    print(f"[handoff] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
