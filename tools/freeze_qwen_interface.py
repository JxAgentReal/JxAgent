#!/usr/bin/env python3
"""Freeze Qwen3.8's native computer-use interface before JxAgent preproduction.

The tool is intentionally fail closed. It records the exact local model files,
chat template, processor configuration, official scaffold evidence, and a
machine-readable JxAgent native-interface contract derived from that evidence.
It never guesses coordinate space, action serialization, message layout, or
history policy.

A verified manifest therefore requires BOTH:
  1. locally present model/tokenizer/processor interface files, and
  2. --native-contract pointing to a JSON contract whose source_evidence files
     are present and hash-pinned.

If any item is missing the manifest is still written, but status is
"unresolved" and every preproduction/training gate must reject it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

MODEL_ID = os.environ.get("JXAGENT_MODEL_ID", "Qwen/Qwen3.8-27B")
SUPPORTED_ADAPTER_FAMILIES = {"jxagent_text_action_v1"}
SUPPORTED_COORDINATE_SPACES = {"processed_image_pixels", "normalized_0_1000"}
SUPPORTED_HISTORY_MODES = {"text_actions", "visual_recent_rounds"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_obj(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def candidate_scaffolds(root: Path) -> List[Path]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.stat().st_size > 5_000_000:
            continue
        low = p.name.lower()
        if (any(k in low for k in ("cua", "computer", "agent", "tool_schema", "tool-schema"))
                and p.suffix.lower() in {".json", ".py", ".md", ".txt", ".jinja", ".j2"}):
            hits.append(p)
    return sorted(hits)


def has_cua_evidence(paths: List[Path]) -> bool:
    """Weak discovery signal only. It can never verify the native contract."""
    action_terms = ("click", "drag", "scroll", "coordinate", "computer", "screen")
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()[:2_000_000]
        except Exception:
            continue
        if sum(term in text for term in action_terms) >= 3 and ("tool" in text or "action" in text):
            return True
    return False


def _resolve_evidence_path(contract_path: Path, model_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    # Contract-relative first, model-root second. Both are deterministic and
    # explicit. Never search the filesystem by basename.
    cp = (contract_path.parent / p).resolve()
    if cp.exists():
        return cp
    return (model_root / p).resolve()


def validate_native_contract(contract: object, contract_path: Path,
                             model_root: Path) -> Tuple[List[str], List[dict]]:
    errors: List[str] = []
    evidence_rows: List[dict] = []
    if not isinstance(contract, dict):
        return ["native_contract_not_json_object"], evidence_rows
    if contract.get("model_id") != MODEL_ID:
        errors.append("native_contract_model_id")
    if int(contract.get("schema_version") or 0) < 1:
        errors.append("native_contract_schema_version")

    adapter = contract.get("adapter") or {}
    if not isinstance(adapter, dict) or adapter.get("family") not in SUPPORTED_ADAPTER_FAMILIES:
        errors.append("unsupported_or_missing_adapter_family")

    coord = contract.get("coordinate_space") or {}
    coord_type = coord.get("type") if isinstance(coord, dict) else coord
    if coord_type not in SUPPORTED_COORDINATE_SPACES:
        errors.append("unsupported_or_missing_coordinate_space")

    layout = contract.get("message_layout") or {}
    required_layout = ("system_prompt", "image_placeholder", "assistant_action_template")
    if not isinstance(layout, dict):
        errors.append("missing_message_layout")
    else:
        for key in required_layout:
            if not isinstance(layout.get(key), str) or not layout.get(key):
                errors.append(f"missing_message_layout_{key}")
        if (isinstance(layout.get("assistant_action_template"), str)
                and "{action}" not in layout["assistant_action_template"]):
            errors.append("assistant_action_template_missing_action_placeholder")

    history = contract.get("history_policy") or {}
    if not isinstance(history, dict) or history.get("mode") not in SUPPORTED_HISTORY_MODES:
        errors.append("unsupported_or_missing_history_policy")
    elif history.get("mode") == "text_actions":
        template = layout.get("text_user_template") if isinstance(layout, dict) else None
        if not isinstance(template, str) or any(x not in template for x in ("{image}", "{task}", "{history}")):
            errors.append("invalid_text_user_template")
        item = layout.get("history_item_template") if isinstance(layout, dict) else None
        if not isinstance(item, str) or "{action}" not in item:
            errors.append("invalid_history_item_template")
    elif history.get("mode") == "visual_recent_rounds":
        try:
            n = int(history.get("recent_visual_rounds"))
        except (TypeError, ValueError):
            n = -1
        if not 1 <= n <= 4:
            errors.append("recent_visual_rounds_must_be_1_to_4")
        if history.get("task_location") not in {"first_user", "current_user", "all_user"}:
            errors.append("invalid_task_location")
        if history.get("older_actions") not in {"coordinate_free", "full", "omit"}:
            errors.append("invalid_older_actions_policy")
        with_task = layout.get("visual_user_with_task_template") if isinstance(layout, dict) else None
        no_task = layout.get("visual_user_without_task_template") if isinstance(layout, dict) else None
        if not isinstance(with_task, str) or "{image}" not in with_task or "{task}" not in with_task:
            errors.append("invalid_visual_user_with_task_template")
        if not isinstance(no_task, str) or "{image}" not in no_task:
            errors.append("invalid_visual_user_without_task_template")
        if history.get("older_actions") != "omit":
            old = layout.get("older_history_template") if isinstance(layout, dict) else None
            item = layout.get("history_item_template") if isinstance(layout, dict) else None
            if not isinstance(old, str) or "{history}" not in old:
                errors.append("invalid_older_history_template")
            if not isinstance(item, str) or "{action}" not in item:
                errors.append("invalid_history_item_template")
            if history.get("older_summary_location") not in {"first_user", "current_user"}:
                errors.append("invalid_older_summary_location")

    evidence = contract.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("missing_source_evidence")
    else:
        for i, row in enumerate(evidence):
            if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
                errors.append(f"invalid_source_evidence_{i}")
                continue
            p = _resolve_evidence_path(contract_path, model_root, str(row["path"]))
            if not p.is_file():
                errors.append(f"source_evidence_missing_{i}")
                continue
            actual = sha(p)
            if actual != str(row["sha256"]).lower():
                errors.append(f"source_evidence_hash_mismatch_{i}")
                continue
            evidence_rows.append({"path": str(p), "sha256": actual,
                                  "kind": row.get("kind") or "official_interface_evidence"})
    return errors, evidence_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--output", default=None)
    ap.add_argument("--native-scaffold", action="append", default=[],
                    help="official local CUA scaffold/tool file; discovery evidence only")
    ap.add_argument("--native-contract", default=None,
                    help="machine-readable JxAgent contract derived from official local evidence; required for verified status")
    args = ap.parse_args()
    root = Path(args.model_dir).resolve()
    if not (root / "config.json").is_file():
        raise SystemExit(f"missing config.json in {root}")

    tracked_names = ["config.json", "tokenizer_config.json", "special_tokens_map.json",
                     "preprocessor_config.json", "processor_config.json",
                     "chat_template.json", "generation_config.json", "REVISION.txt"]
    files: Dict[str, str] = {}
    for name in tracked_names:
        p = root / name
        if p.is_file():
            files[name] = sha(p)

    tok_cfg = read_json(root / "tokenizer_config.json") or {}
    chat_template = tok_cfg.get("chat_template")
    template_source = "tokenizer_config.json:chat_template" if chat_template else None
    if not chat_template:
        p = root / "chat_template.json"
        obj = read_json(p) if p.is_file() else None
        if isinstance(obj, dict):
            chat_template = obj.get("chat_template") or obj.get("template")
            template_source = "chat_template.json" if chat_template else None
    tokenizer_error = None
    if not chat_template:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True,
                                                local_files_only=True)
            chat_template = getattr(tok, "chat_template", None)
            template_source = "AutoTokenizer.chat_template" if chat_template else None
            tokenizer_class = tok.__class__.__name__
            special_tokens = {k: getattr(tok, k, None) for k in
                              ("bos_token", "eos_token", "pad_token", "unk_token")}
            special_token_ids = {k: getattr(tok, k + "_id", None) for k in
                                 ("bos_token", "eos_token", "pad_token", "unk_token")}
        except Exception as e:
            tokenizer_class = None
            special_tokens = {}
            special_token_ids = {}
            tokenizer_error = str(e)
    else:
        tokenizer_class = tok_cfg.get("tokenizer_class")
        special_tokens = {k: tok_cfg.get(k) for k in
                          ("bos_token", "eos_token", "pad_token", "unk_token")}
        special_token_ids = {}

    cfg = read_json(root / "config.json") or {}
    proc_cfg = read_json(root / "preprocessor_config.json") or read_json(root / "processor_config.json") or {}

    explicit = [Path(x).resolve() for x in args.native_scaffold]
    for p in explicit:
        if not p.is_file():
            raise SystemExit(f"native scaffold not found: {p}")
    scaffolds = explicit or candidate_scaffolds(root)
    weak_cua_discovery = bool(scaffolds and has_cua_evidence(scaffolds))
    scaffold_rows = []
    for p in scaffolds:
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        scaffold_rows.append({"path": rel, "sha256": sha(p), "explicit": p in explicit})

    contract = None
    contract_path = None
    contract_errors: List[str] = []
    contract_evidence: List[dict] = []
    if args.native_contract:
        contract_path = Path(args.native_contract).resolve()
        if not contract_path.is_file():
            contract_errors.append("native_contract_file_missing")
        else:
            contract = read_json(contract_path)
            contract_errors, contract_evidence = validate_native_contract(
                contract, contract_path, root)
    else:
        contract_errors.append("native_contract_missing")

    template_hash = hashlib.sha256(str(chat_template or "").encode("utf-8")).hexdigest()
    unresolved = []
    if not chat_template:
        unresolved.append("chat_template")
    if not proc_cfg:
        unresolved.append("processor_config")
    unresolved.extend(contract_errors)

    manifest = {
        "schema_version": 3,
        "model_dir": str(root),
        "model_id": MODEL_ID,
        "revision": (root / "REVISION.txt").read_text().strip() if (root / "REVISION.txt").is_file() else None,
        "status": "verified" if not unresolved else "unresolved",
        "unresolved": sorted(set(unresolved)),
        "tracked_files": files,
        "model": {"model_type": cfg.get("model_type"), "architectures": cfg.get("architectures")},
        "tokenizer": {"class": tokenizer_class, "chat_template_source": template_source,
                      "chat_template_sha256": template_hash,
                      "special_tokens": special_tokens, "special_token_ids": special_token_ids,
                      "load_error": tokenizer_error},
        "processor": {"class_hint": proc_cfg.get("processor_class"),
                      "image_processor_type": proc_cfg.get("image_processor_type"),
                      "config_sha256": sha_obj(proc_cfg) if proc_cfg else None},
        "native_cua": {
            "heuristic_scaffold_discovery": weak_cua_discovery,
            "scaffold_evidence": scaffold_rows,
            "contract_path": str(contract_path) if contract_path else None,
            "contract_sha256": sha(contract_path) if contract_path and contract_path.is_file() else None,
            "contract": contract if isinstance(contract, dict) else None,
            "source_evidence": contract_evidence,
            "note": "Heuristic CUA discovery is never sufficient for verification. Coordinate space, action grammar, history policy and message layout must come from the explicit hash-pinned native contract.",
        },
        "jxagent_training_contract": {"assistant_loss": "assistant messages only",
                                   "synthetic_reasoning": False,
                                   "interface_must_not_drift": True},
    }
    out = Path(args.output or root / "jxagent_interface_manifest.json")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "status": manifest["status"],
                      "unresolved": manifest["unresolved"]}, indent=2))
    return 0 if manifest["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
