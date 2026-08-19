#!/usr/bin/env python3
"""Fail closed on unresolved Qwen interface state or any interface drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    root = Path(a.model_dir).resolve()
    mp = Path(a.manifest or root / "jxagent_interface_manifest.json").resolve()
    if not mp.is_file():
        raise SystemExit(f"FATAL: interface manifest missing: {mp}")
    m = json.loads(mp.read_text(encoding="utf-8"))
    errors = []
    if m.get("status") != "verified":
        errors.append("manifest status is not verified: " + ",".join(m.get("unresolved") or []))
    expected_model = os.environ.get("JXAGENT_MODEL_ID", "Qwen/Qwen3.8-27B")
    if m.get("model_id") != expected_model:
        errors.append(f"unexpected model_id: {m.get('model_id')} != {expected_model}")
    if int(m.get("schema_version") or 0) < 3:
        errors.append("interface manifest schema too old")

    for rel, dig in (m.get("tracked_files") or {}).items():
        p = root / rel
        if not p.is_file():
            errors.append(f"missing tracked file {rel}")
        elif sha(p) != dig:
            errors.append(f"interface drift: {rel}")

    native = m.get("native_cua") or {}
    cp = native.get("contract_path")
    cd = native.get("contract_sha256")
    if not cp or not cd:
        errors.append("native contract missing from manifest")
    else:
        p = Path(cp)
        if not p.is_file():
            errors.append(f"native contract missing: {cp}")
        elif sha(p) != cd:
            errors.append("native contract drift")

    for row in native.get("scaffold_evidence") or []:
        p = Path(row["path"])
        p = p if p.is_absolute() else root / p
        if not p.is_file() or sha(p) != row.get("sha256"):
            errors.append(f"CUA scaffold drift/missing: {row['path']}")

    for row in native.get("source_evidence") or []:
        p = Path(row.get("path") or "")
        if not p.is_file() or sha(p) != row.get("sha256"):
            errors.append(f"native source evidence drift/missing: {row.get('path')}")

    if errors:
        print("\n".join("[interface][FAIL] " + x for x in errors))
        return 2
    print("[interface] VERIFIED and unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
