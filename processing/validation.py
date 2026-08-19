"""Sample validation and dataset finalization (spec sections 22, 26).

Per-sample checks:
  schema, non-empty assistant target, valid action syntax, coordinates within
  final image bounds, POSIX-only relative image paths, images exist and decode.

Finalization writes final/{train,validation}.jsonl, manifest.json, stats.json,
source_stats.json and enforces the fatal-failure conditions.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .coordinates import parse_rendered_action, action_in_bounds
from .quality import UNDERSTANDING_MAX_CONTROLS
from .motifs import coverage as motif_coverage
from .token_budget import estimate_loss_token_report

VALID_SOURCES = {"procua", "gui360", "videocua", "groundcua", "pcagente", "replay"}
WINDOWS_PATH_RE = re.compile(r"\\\\|\\")
_ACTION_LINE_RE = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)


def last_assistant_content(sample: dict) -> str:
    content = ""
    for m in reversed(sample.get("messages", [])):
        if m.get("role") == "assistant":
            content = m.get("content", "")
            break
    return content if isinstance(content, str) else json.dumps(content)


def extract_action_text(sample: dict) -> str:
    """The final action of the last assistant turn ('Action: x' line or the
    whole content when it is a bare action)."""
    content = last_assistant_content(sample).strip()
    m = None
    for m in _ACTION_LINE_RE.finditer(content):
        pass
    if m:
        return m.group(1).strip()
    lines = [l for l in content.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def validate_sample(sample: dict, dataset_root: str, *, text_only_ok: bool = True) -> Tuple[bool, str]:
    """Validate one final sample dict. Returns ``(ok, failure_reason)``.

    Hardening rule: action supervision is accepted only when it has exactly one
    assistant target conditioned on the sample's real visual state.  This
    prevents stale legacy window/chunk examples from hiding invalid earlier
    assistant turns behind a valid final turn.
    """
    msgs = sample.get("messages")
    if not msgs or not isinstance(msgs, list):
        return False, "invalid_messages"
    if any(not isinstance(m, dict) for m in msgs):
        return False, "invalid_messages"
    roles = [m.get("role") for m in msgs]
    if roles[0] not in ("system", "user") or "assistant" not in roles:
        return False, "invalid_roles"
    assistants = [m for m in msgs if m.get("role") == "assistant"]
    if any(not isinstance(m.get("content"), str) or not m["content"].strip()
           for m in assistants):
        return False, "empty_assistant_target"
    for m in msgs:
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False, "empty_message_content"
    if sample.get("source") not in VALID_SOURCES:
        return False, "unknown_source"
    if not sample.get("trajectory_id"):
        return False, "missing_trajectory_id"

    images = sample.get("images", [])
    if not isinstance(images, list):
        return False, "invalid_images"
    meta_pre = sample.get("metadata", {}) or {}
    image_placeholder = str(meta_pre.get("interface_image_placeholder") or "<image>")
    n_placeholders = sum(str(m.get("content", "")).count(image_placeholder) for m in msgs)
    if n_placeholders != len(images):
        return False, "image_placeholder_mismatch"
    decoded_sizes = []
    for p in images:
        if not isinstance(p, str) or WINDOWS_PATH_RE.search(p) or os.path.isabs(p) or p.startswith("/"):
            return False, "non_portable_path"
        if ".." in p:
            return False, "non_portable_path"
        full = os.path.join(dataset_root, p.replace("/", os.sep))
        if not os.path.exists(full):
            return False, "missing_image"
        try:
            from PIL import Image
            with Image.open(full) as im:
                im.load()
                decoded_sizes.append((int(im.width), int(im.height)))
        except Exception:
            return False, "image_undecodable"

    task_type = str(sample.get("task_type", ""))
    meta = sample.get("metadata", {}) or {}

    if task_type in {"action", "grounding"}:
        # Exactly one assistant turn may contribute supervised loss. Native
        # visual-history contracts may retain prior real screenshot/assistant
        # rounds, but those assistant turns MUST be loss=False and each visual
        # state must be present. Legacy/text-history samples remain one-image.
        native_visual = (
            bool(meta.get("native_interface"))
            and meta.get("interface_history_mode") == "visual_recent_rounds"
        )
        # Keep the legacy diagnostic stable for ordinary action samples. Native
        # visual history is the sole supported multi-assistant exception.
        if not native_visual and len(assistants) != 1:
            return False, "multi_assistant_action_without_state_alignment"
        supervised_assistants = [m for m in assistants if m.get("loss", True) is not False]
        if len(supervised_assistants) != 1:
            return False, "action_requires_exactly_one_supervised_assistant"
        if meta.get("native_interface"):
            expected_hash = meta.get("assistant_target_sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                return False, "native_target_hash_missing"
            actual_hash = hashlib.sha256(
                str(supervised_assistants[0].get("content") or "").encode("utf-8")
            ).hexdigest()
            if actual_hash != expected_hash:
                return False, "native_target_hash_mismatch"
        if native_visual:
            if len(images) < 1:
                return False, "action_requires_visual_state"
            if any(m.get("loss", True) is not False for m in assistants[:-1]):
                return False, "native_history_prior_assistant_has_loss"
            if assistants[-1] is not supervised_assistants[0]:
                return False, "native_history_supervised_assistant_not_last"
        else:
            if len(assistants) != 1:
                return False, "multi_assistant_action_without_state_alignment"
            if len(images) != 1:
                return False, "action_requires_exactly_one_image"
        action_text = extract_action_text(sample)
        canonical_action = meta.get("action_canonical_final_pixels")
        # Legacy JxAgent targets are directly parseable and must be validated from
        # the actual assistant text so post-assembly corruption cannot be hidden
        # behind stale metadata. Native targets may wrap or normalize the action,
        # so only those use the canonical final-pixel action stored at assembly.
        parse_text = canonical_action if meta.get("native_interface") else action_text
        action = parse_rendered_action(str(parse_text or ""))
        if action is None:
            return False, "invalid_action_syntax"
        final_size = meta.get("final_image_size") or meta.get("image_size")
        if not final_size or len(final_size) < 2:
            return False, "missing_final_image_size"
        try:
            final_wh = (int(final_size[0]), int(final_size[1]))
        except Exception:
            return False, "missing_final_image_size"
        if decoded_sizes and final_wh != decoded_sizes[-1]:
            return False, "final_image_size_mismatch"
        if action.points and not action_in_bounds(action, final_wh[0], final_wh[1]):
            return False, "coordinate_out_of_bounds"
        if task_type == "grounding" and not action.points:
            return False, "grounding_without_point"
        if action.verb == "finish" and meta.get("finish_evidence") != "yes":
            return False, "finish_without_evidence"

    elif task_type == "screen_understanding":
        # Understanding targets are JSON, not GUI actions.  Validate the exact
        # observable schema and normalized coordinate contract.
        if len(assistants) != 1 or len(images) != 1:
            return False, "invalid_understanding_turn_structure"
        final_size = meta.get("final_image_size") or meta.get("image_size")
        if final_size:
            try:
                final_wh = (int(final_size[0]), int(final_size[1]))
            except Exception:
                return False, "missing_final_image_size"
            if decoded_sizes and final_wh != decoded_sizes[-1]:
                return False, "final_image_size_mismatch"
        try:
            controls = json.loads(assistants[0]["content"])
        except Exception:
            return False, "malformed_understanding_json"
        if not isinstance(controls, list) or not controls or len(controls) > UNDERSTANDING_MAX_CONTROLS:
            return False, "malformed_understanding_json"
        last_key = None
        for c in controls:
            if not isinstance(c, dict) or not str(c.get("control_text", "")).strip():
                return False, "malformed_understanding_control"
            rect = c.get("control_rect")
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                return False, "malformed_understanding_control"
            try:
                x1, y1, x2, y2 = map(float, rect)
            except Exception:
                return False, "malformed_understanding_control"
            if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
                return False, "understanding_rect_out_of_bounds"
            # The target contract is spatially deterministic, so enforce it.
            key = (round(y1, 6), round(x1, 6), round(y2, 6), round(x2, 6))
            if last_key is not None and key < last_key:
                return False, "understanding_not_spatially_sorted"
            last_key = key

    elif sample.get("source") == "replay":
        # Replay is intentionally free-form text/tool/VQA supervision. Its
        # source adapter owns category-specific structural validation.
        if not text_only_ok and not images:
            return False, "text_only_not_allowed"

    return True, ""


# --------------------------------------------------------------------- stats

def compute_stats(samples: List[dict], dataset_root: str) -> Dict:
    per_source = Counter(s["source"] for s in samples)
    per_type = Counter(s.get("task_type", "?") for s in samples)
    repr_counts = Counter(s.get("metadata", {}).get("representation", "n/a") for s in samples)
    reasoning = [s for s in samples if "Plan:" in last_assistant_content(s)]
    reasoning_cats = Counter(s.get("metadata", {}).get("reasoning_category", "?") for s in reasoning)
    verbs = Counter()
    seq_est = []
    img_sizes = []
    img_bytes = 0
    img_count = 0
    buckets = Counter()
    apps = Counter()
    for s in samples:
        q = (s.get("metadata", {}) or {}).get("quality", {})
        if q.get("bucket"):
            buckets[q["bucket"]] += 1
        if (s.get("metadata", {}) or {}).get("app"):
            apps[s["metadata"]["app"]] += 1
    for s in samples:
        canonical = (s.get("metadata", {}) or {}).get("action_canonical_final_pixels")
        a = parse_rendered_action(str(canonical or extract_action_text(s)))
        if a:
            verbs[a.verb] += 1
        meta = s.get("metadata", {})
        if meta.get("estimated_tokens"):
            seq_est.append(meta["estimated_tokens"])
        if meta.get("final_image_size"):
            img_sizes.append(tuple(meta["final_image_size"]))
        for p in s.get("images", []):
            full = os.path.join(dataset_root, p.replace("/", os.sep))
            try:
                img_bytes += os.path.getsize(full)
                img_count += 1
            except OSError:
                pass

    def avg(xs):
        xs = [x for x in xs if x]
        return sum(xs) / len(xs) if xs else 0

    total_quality = sum(buckets.values())
    # ---- extended mixture report (pre-build patch 2026-08-16) ----
    replay_categories = Counter(s.get("task_type", "?") for s in samples
                                if s.get("source") == "replay")
    repr_by_source: Dict[str, Counter] = defaultdict(Counter)
    finish_evidence = Counter()
    finish_count = 0
    grounding_buckets = defaultdict(Counter)
    dim_mismatch = 0
    epoch_tokens = 0
    msg_counts = []
    for s in samples:
        meta = s.get("metadata", {}) or {}
        repr_by_source[s.get("source", "?")][meta.get("representation", "?")] += 1
        if meta.get("finish_evidence"):
            finish_evidence[meta["finish_evidence"]] += 1
        if meta.get("coordinate_dimension_mismatch"):
            dim_mismatch += 1
        epoch_tokens += int(meta.get("estimated_tokens") or 0)
        msg_counts.append(len(s.get("messages", [])))
        tw = meta.get("target_width_px")
        th = meta.get("target_height_px")
        if tw is not None:
            from .quality import grounding_bucket
            grounding_buckets[s.get("source", "?")][grounding_bucket(
                int(tw), int(th) if th is not None else None)] += 1
    for s in samples:
        canonical = (s.get("metadata", {}) or {}).get("action_canonical_final_pixels")
        a = parse_rendered_action(str(canonical or extract_action_text(s)))
        if a and a.verb == "finish":
            finish_count += 1
    return {
        "total_samples": len(samples),
        "quality_buckets": dict(buckets),
        "quality_bucket_shares": {k: round(v / max(1, total_quality), 4)
                                  for k, v in buckets.items()},
        "top_applications": dict(apps.most_common(20)),
        "samples_per_source": dict(per_source),
        "task_type_distribution": dict(per_type),
        "replay_category_counts": dict(replay_categories),
        "representation_counts_by_source": {k: dict(v) for k, v in repr_by_source.items()},
        "single_step_count": repr_counts.get("single", 0),
        "short_window_count": repr_counts.get("window", 0),
        "long_chunk_count": repr_counts.get("chunk", 0),
        "finish_sample_count": finish_count,
        "finish_evidence": {"yes": finish_evidence.get("yes", 0),
                            "no": finish_evidence.get("no", 0),
                            "unknown": max(0, finish_count - sum(finish_evidence.values()))},
        "grounding_size_distribution_by_source": {k: dict(v) for k, v in grounding_buckets.items()},
        "coordinate_dimension_mismatches": dim_mismatch,
        "reasoning_count": len(reasoning),
        "reasoning_percentage": round(100.0 * len(reasoning) / max(1, len(samples)), 2),
        "reasoning_category_distribution": dict(reasoning_cats),
        "action_verb_distribution": dict(verbs),
        "average_estimated_tokens": round(avg(seq_est), 1),
        "estimated_epoch_tokens": epoch_tokens,
        "average_messages_per_sample": round(avg(msg_counts), 2),
        "average_image_width": round(avg([w for w, h in img_sizes]), 1),
        "average_image_height": round(avg([h for w, h in img_sizes]), 1),
        "average_image_file_bytes": round(img_bytes / max(1, img_count), 1),
        "total_image_bytes": img_bytes,
        "total_images": img_count,
    }


FATAL_CONDITIONS = [
    "missing_image", "invalid_json", "empty_assistant_target",
    "coordinate_out_of_bounds", "image_undecodable", "non_portable_path",
    "duplicate_sample_id",
]

# Schema-integrity failures: ZERO tolerance in production by default. A build
# that silently drops malformed rows and still reports success is a red-team
# hole. An exceptional class may only be tolerated via an explicit configured
# threshold (fraction of total); tolerated counts/examples are recorded.
SCHEMA_FATAL_REASONS = {
    "invalid_action_syntax", "invalid_roles", "invalid_messages",
    "empty_message_content", "unknown_source", "missing_trajectory_id",
    "image_placeholder_mismatch", "grounding_without_point", "invalid_images",
    "multi_assistant_action_without_state_alignment", "action_without_image",
    "action_requires_exactly_one_image", "action_requires_exactly_one_supervised_assistant",
    "action_requires_visual_state", "native_history_prior_assistant_has_loss",
    "native_history_supervised_assistant_not_last", "native_target_hash_missing",
    "native_target_hash_mismatch", "missing_final_image_size",
    "final_image_size_mismatch", "invalid_understanding_turn_structure",
    "malformed_understanding_json", "malformed_understanding_control",
    "understanding_rect_out_of_bounds", "understanding_not_spatially_sorted",
    "finish_without_evidence",
}
ALL_FATAL_REASONS = set(FATAL_CONDITIONS) | SCHEMA_FATAL_REASONS | {
    "train_validation_group_overlap", "motif_coverage_floor_unmet",
    "auxiliary_loss_share_exceeded"}


# Per-source quota acceptance thresholds (fraction of target that must be
# realized). PC-Agent-E has a documented lower expectation: its quality gate
# realistically yields ~4.3k of the 4,503 quota.
DEFAULT_SOURCE_THRESHOLD = 0.99
SOURCE_THRESHOLDS = {
    "pcagente": {"threshold": 0.90,
                 "reason": "quality gate historically realizes ~4.3k of 4503 "
                           "(documented exception; do NOT backfill)"},
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_images_tree(dataset_root: str, state_dir: Optional[str] = None) -> Dict:
    """Hash every image under images/ in sorted path order.

    Writes state/image_hashes.jsonl with one {path,bytes,sha256} row per file
    and returns the aggregate `images_tree_hash`: sha256 over the sorted
    "path\\tbytes\\tsha256\\n" lines. Deterministic on any OS (POSIX relpaths,
    forward slashes)."""
    images_root = os.path.join(dataset_root, "images")
    rows = []
    if os.path.isdir(images_root):
        for dirpath, _dirnames, filenames in os.walk(images_root):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, images_root).replace(os.sep, "/")
                rows.append((rel, full))
    rows.sort(key=lambda r: r[0])
    lines = []
    out_path = os.path.join(state_dir or os.path.join(dataset_root, "state"),
                            "image_hashes.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    total_bytes = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for rel, full in rows:
            size = os.path.getsize(full)
            digest = _sha256_file(full)
            total_bytes += size
            lines.append(f"{rel}\t{size}\t{digest}")
            f.write(json.dumps({"path": rel, "bytes": size, "sha256": digest},
                               ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    tree_hash = hashlib.sha256(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")).hexdigest()
    return {"images_tree_hash": tree_hash, "image_files": len(rows),
            "image_bytes": total_bytes}


def compute_quota_acceptance(selected_per_source: Dict[str, int],
                             targets: Dict[str, int],
                             source_errors: Optional[Dict[str, str]] = None) -> Dict:
    """Per-source acceptance verdict. A source that raised an exception is
    never accepted, even if enough samples were selected before the error."""
    out = {}
    for name, target in targets.items():
        selected = selected_per_source.get(name, 0)
        spec = SOURCE_THRESHOLDS.get(name, {})
        threshold = spec.get("threshold", DEFAULT_SOURCE_THRESHOLD)
        reason = spec.get("reason", "default >=99% of target")
        realization = round(100.0 * selected / target, 2) if target else 100.0
        err = (source_errors or {}).get(name)
        if err:
            accepted = False
            reason = f"source raised an exception during the build: {err[:200]}"
        else:
            accepted = (selected >= target * threshold) if target else True
        out[name] = {
            "target": target, "selected": selected,
            "realization_pct": realization,
            "acceptance_threshold_pct": round(threshold * 100, 2),
            "accepted": accepted, "reason": reason,
            "source_error": bool(err),
        }
    return out


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def config_hash_of(config_snapshot: Dict) -> str:
    """Stable hash of the effective build config: same config -> same hash."""
    return hashlib.sha256(_canonical_json(config_snapshot).encode("utf-8")).hexdigest()


def environment_snapshot() -> Dict[str, str]:
    """Versions of artifact-affecting libraries (written to final/environment.json)."""
    import platform
    snap = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        from importlib.metadata import version, PackageNotFoundError
        for pkg in ("pillow", "numpy", "datasets", "huggingface_hub", "zstandard",
                    "av", "requests", "PyYAML", "tqdm"):
            try:
                snap[pkg] = version(pkg)
            except PackageNotFoundError:
                snap[pkg] = "not-installed"
    except Exception:  # noqa: BLE001 - environment.json is best-effort
        pass
    return snap


def summarize_failures(rows: List[dict], quota_acceptance: Optional[Dict]) -> Dict:
    """Safe summary of state/failures.jsonl for publication (final/build_failures_summary.json).
    Credentials are redacted; raw tracebacks are dropped (class/source/counts)."""
    from .state import redact_secrets
    by_class: Dict[str, Dict] = {}
    for r in rows:
        cls = redact_secrets(str(r.get("error", "unknown"))[:120])
        entry = by_class.setdefault(cls, {"count": 0, "sources": {}})
        entry["count"] += 1
        entry["sources"][r.get("source", "?")] = entry["sources"].get(r.get("source", "?"), 0) + 1
    acceptance = quota_acceptance or {}
    return {
        "failure_classes": [
            {"class": cls, "count": v["count"], "sources": v["sources"],
             "affected_quota": sorted(s for s in v["sources"]
                                      if not acceptance.get(s, {}).get("accepted", True))}
            for cls, v in sorted(by_class.items())
        ],
        "unresolved_quota_impact": sorted(
            s for s, v in acceptance.items() if not v.get("accepted", True)),
        "production_acceptance_passed": (all(v.get("accepted", True)
                                             for v in acceptance.values())
                                         if acceptance else None),
        "note": "tracebacks and request details omitted; secrets redacted",
    }


def cleanup_orphan_images(dataset_root: str, samples: List[dict]) -> Dict:
    """Delete generated images referenced by NO final sample.

    Runs AFTER train/validation are written, so references are final. Only
    builder-generated .webp files under images/ are ever removed — raw/source
    files elsewhere are untouched. Safe to rerun (idempotent)."""
    referenced = set()
    for s in samples:
        for p in s.get("images", []):
            if isinstance(p, str):
                referenced.add(p.replace("\\", "/"))
    img_root = os.path.join(dataset_root, "images")
    deleted = kept = errors = 0
    if os.path.isdir(img_root):
        for dirpath, _dirnames, filenames in os.walk(img_root):
            for fn in filenames:
                if not fn.lower().endswith(".webp"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, dataset_root).replace(os.sep, "/")
                if rel in referenced:
                    kept += 1
                else:
                    try:
                        os.remove(full)
                        deleted += 1
                    except OSError:
                        errors += 1
    return {"referenced": len(referenced), "kept": kept,
            "deleted_orphans": deleted, "errors": errors}


def finalize(dataset_root: str, samples: List[dict], *, validation_pct_marker: float = 3.0,
             decontamination_report: Optional[dict] = None,
             dedup_stats: Optional[dict] = None,
             targets: Optional[Dict[str, int]] = None,
             source_errors: Optional[Dict[str, str]] = None,
             tolerances: Optional[Dict[str, float]] = None,
             build_identity: Optional[dict] = None,
             state_dir: Optional[str] = None,
             release_gates: Optional[dict] = None) -> Dict:
    """Write final/ outputs and enforce fatal conditions. Returns stats.

    Deterministic artifact order (documented; avoids circular hashes):
      1. final/train.jsonl, final/validation.jsonl
      2. orphan cleanup + image tree hashing -> state/image_hashes.jsonl
      3. final/stats.json, final/source_stats.json (incl. image tree hash)
      4. final/build_config.json, final/environment.json,
         final/build_failures_summary.json (when build_identity provided)
      5. final/manifest.json (contains config hash, tree hash, metadata hashes)
      6. final/SHA256SUMS (hashes every final/ metadata file incl. manifest)
    """
    final_dir = os.path.join(dataset_root, "final")
    os.makedirs(final_dir, exist_ok=True)

    # belt-and-braces identity dedup (main() already merges by id): duplicates
    # would double-train steps and must fail the build, not ship silently
    seen_ids = set()
    deduped: List[dict] = []
    dup_count = 0
    for s in samples:
        key = (s.get("source"), s.get("trajectory_id"), s.get("step_id"))
        if key in seen_ids:
            dup_count += 1
            continue
        seen_ids.add(key)
        deduped.append(s)
    samples = deduped

    failures = Counter()
    if dup_count:
        failures["duplicate_sample_id"] = dup_count
    valid_samples = []
    schema_examples: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        ok, reason = validate_sample(s, dataset_root)
        if ok:
            valid_samples.append(s)
        else:
            failures[reason] += 1
            if reason in SCHEMA_FATAL_REASONS and len(schema_examples[reason]) < 3:
                schema_examples[reason].append(str(s.get("step_id", "?")))

    train = [s for s in valid_samples if s.get("split") == "train"]
    val = [s for s in valid_samples if s.get("split") == "validation"]

    from .splitting import trajectory_overlap, group_overlap
    overlap = trajectory_overlap(train, val)
    if overlap:
        failures["train_validation_trajectory_overlap"] = len(overlap)
    grp_overlap = group_overlap(train, val)
    if grp_overlap:
        failures["train_validation_group_overlap"] = len(grp_overlap)

    # Second-stage release gates are measured on the selected final rows.
    release_gates = release_gates or {}
    loss_report = estimate_loss_token_report(valid_samples)
    max_aux = float((release_gates.get("loss_token_gate") or {}).get(
        "max_auxiliary_task_share", 1.0))
    aux_violations = {k: v["assistant_loss_share"] for k, v in loss_report.get("task_type", {}).items()
                      if k not in {"action", "grounding"} and v["assistant_loss_share"] > max_aux}
    if aux_violations:
        failures["auxiliary_loss_share_exceeded"] = len(aux_violations)
    motifs = motif_coverage(valid_samples)
    cu_n = sum(1 for s in valid_samples if s.get("source") != "replay")
    floor_cfg = release_gates.get("coverage_floors") or {}
    import math as _math
    motif_required = {m: (int(_math.ceil(float(v) * cu_n)) if float(v) < 1.0 else int(v))
                      for m, v in floor_cfg.items()}
    motif_unmet = {m: {"required": n, "selected": motifs.get(m, 0)}
                   for m, n in motif_required.items() if motifs.get(m, 0) < n}
    if motif_unmet:
        failures["motif_coverage_floor_unmet"] = len(motif_unmet)

    # schema failures: zero tolerance by default; explicit thresholds only
    tolerances = tolerances or {}
    total_n = max(1, len(samples))
    tolerated: Dict[str, dict] = {}
    for reason in SCHEMA_FATAL_REASONS:
        n = failures.get(reason, 0)
        if not n:
            continue
        allowed = int(float(tolerances.get(reason, 0.0)) * total_n)
        if n <= allowed:
            tolerated[reason] = {
                "count": n, "percent": round(100.0 * n / total_n, 4),
                "configured_max_percent": round(float(tolerances.get(reason, 0.0)) * 100, 4),
                "examples": schema_examples.get(reason, []),
            }

    def write_jsonl(path: str, rows: List[dict]):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    write_jsonl(os.path.join(final_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(final_dir, "validation.jsonl"), val)

    # orphan sweep AFTER the final references exist (assemble never deletes)
    cleanup = cleanup_orphan_images(dataset_root, train + val)

    stats = compute_stats(valid_samples, dataset_root)
    stats.update({
        "train_samples": len(train),
        "validation_samples": len(val),
        "failures": dict(failures),
        "fatal": {k: v for k, v in failures.items() if k in ALL_FATAL_REASONS},
        "fatal_failure": bool(any(v > 0 for k, v in failures.items()
                                  if k in ALL_FATAL_REASONS and k not in tolerated)
                              or overlap or grp_overlap),
        "train_validation_trajectory_overlap": len(overlap),
        "train_validation_group_overlap": len(grp_overlap),
        "loss_token_budget_estimated": loss_report,
        "auxiliary_loss_share_violations": aux_violations,
        "motif_coverage": motifs,
        "motif_coverage_required": motif_required,
        "motif_coverage_unmet": motif_unmet,
        "orphan_cleanup": cleanup,
        "tolerated_failures": tolerated,
    })

    source_stats = defaultdict(lambda: {"selected": 0, "train": 0, "validation": 0,
                                        "task_types": Counter()})
    for s in valid_samples:
        st = source_stats[s["source"]]
        st["selected"] += 1
        st[s.get("split", "train")] += 1
        st["task_types"][s.get("task_type", "?")] += 1
    source_stats = {k: {**v, "task_types": dict(v["task_types"])} for k, v in source_stats.items()}
    quota_acceptance = None
    if targets:
        selected_per_source = {k: v.get("selected", 0) for k, v in source_stats.items()}
        quota_acceptance = compute_quota_acceptance(selected_per_source, targets,
                                                    source_errors)
        for k, t in targets.items():
            entry = source_stats.setdefault(k, {})
            entry["target"] = t
            entry["target_met"] = entry.get("selected", 0) == t
            entry.update(quota_acceptance[k])
    stats["quota_acceptance"] = quota_acceptance
    stats["quota_acceptance_passed"] = (all(v["accepted"] for v in quota_acceptance.values())
                                        if quota_acceptance else None)

    from .state import BuildState
    if decontamination_report is not None:
        BuildState.atomic_write_json(os.path.join(final_dir, "decontamination_report.json"),
                                     decontamination_report)
    if dedup_stats is not None:
        stats["dedup"] = dedup_stats

    # Hash the post-cleanup image tree BEFORE persisting stats.  Earlier code
    # mutated the in-memory stats after stats.json had already been written,
    # so callers saw images_tree_hash while the durable artifact did not.
    img_tree = hash_images_tree(dataset_root, state_dir)
    stats["images_tree_hash"] = img_tree["images_tree_hash"]
    stats["image_files_hashed"] = img_tree["image_files"]
    BuildState.atomic_write_json(os.path.join(final_dir, "loss_token_report_estimated.json"), loss_report)
    BuildState.atomic_write_json(os.path.join(final_dir, "stats.json"), stats)
    BuildState.atomic_write_json(os.path.join(final_dir, "source_stats.json"), source_stats)

    # build identity snapshots (config hash is computed by the orchestrator so
    # it can cover CLI flags; written here for hash-ordering simplicity)
    if build_identity:
        BuildState.atomic_write_json(os.path.join(final_dir, "build_config.json"),
                                     build_identity.get("config_snapshot", {}))
        BuildState.atomic_write_json(os.path.join(final_dir, "environment.json"),
                                     build_identity.get("environment", {}))
        if build_identity.get("failures_rows") is not None:
            BuildState.atomic_write_json(
                os.path.join(final_dir, "build_failures_summary.json"),
                summarize_failures(build_identity["failures_rows"], quota_acceptance))

    metadata_hashes = {}
    for name in ("train.jsonl", "validation.jsonl", "stats.json", "source_stats.json",
                 "decontamination_report.json", "build_config.json", "environment.json",
                 "build_failures_summary.json", "loss_token_report_estimated.json",
                 "selection_report.json"):
        p = os.path.join(final_dir, name)
        if os.path.exists(p):
            metadata_hashes[f"final/{name}"] = _sha256_file(p)

    manifest = {
        "dataset": "JxAgentData",
        "format": "ms-swift multimodal JSONL",
        "train_samples": len(train),
        "validation_samples": len(val),
        "total_samples": len(valid_samples),
        "targets": targets or {},
        "sources": {k: v.get("selected", 0) for k, v in source_stats.items()},
        "image_root": "images/",
        "image_format": "GUI WebP lossless; natural-photo replay WebP q80",
        "context_budget": 8192,
        "validation_split": f"group-aware semantic/provenance, {validation_pct_marker}% by md5(group_id)",
        "loss_token_report": "final/loss_token_report_estimated.json (exact tokenizer report required before training)",
        "motif_coverage_unmet": motif_unmet,
        "fatal_failure": stats["fatal_failure"],
        "validation_result": {
            "fatal_failure": stats["fatal_failure"],
            "failures": dict(failures),
            "tolerated_failures": tolerated,
        },
        "quota_acceptance": quota_acceptance,
        "hashes": metadata_hashes,
        "images_tree_hash": img_tree["images_tree_hash"],
        "image_count": img_tree["image_files"],
    }
    if build_identity:
        manifest.update({
            "build_id": build_identity.get("build_id"),
            "builder_commit": build_identity.get("builder_commit"),
            "config_hash": build_identity.get("config_hash"),
            "source_revisions": build_identity.get("source_revisions", {}),
            "build_timestamp": build_identity.get("finished_at"),
            "build_started_at": build_identity.get("started_at"),
            "selection_policy": build_identity.get("selection_policy"),
            "decontamination_reference": build_identity.get("decontamination"),
            "environment": build_identity.get("environment", {}),
            "state_corruption": build_identity.get("state_corruption"),
        })
    BuildState.atomic_write_json(os.path.join(final_dir, "manifest.json"), manifest)

    # step 6-7: SHA256SUMS over every final metadata file including manifest
    sums_path = os.path.join(final_dir, "SHA256SUMS")
    lines = []
    for name in ("train.jsonl", "validation.jsonl", "stats.json", "source_stats.json",
                 "decontamination_report.json", "build_config.json", "environment.json",
                 "build_failures_summary.json", "loss_token_report_estimated.json",
                 "selection_report.json", "manifest.json"):
        p = os.path.join(final_dir, name)
        if os.path.exists(p):
            lines.append(f"{_sha256_file(p)}  final/{name}")
    tmp = sums_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, sums_path)
    return stats


def quality_audit(samples: List[dict], dataset_root: str, n: int = 100,
                  seed: str = "audit") -> Optional[dict]:
    """Random inspection of n final computer-use examples classified
    good/questionable/bad via automated checks (a human pass is still
    recommended before production)."""
    cu = [s for s in samples if s.get("source") != "replay"]
    if not cu:
        return None
    import hashlib as _h
    ranked = sorted(cu, key=lambda s: _h.md5((seed + s.get("step_id", "")).encode()).hexdigest())
    picked = ranked[:n]
    counts = Counter()
    issues = Counter()
    for s in picked:
        ok, reason = validate_sample(s, dataset_root)
        if not ok:
            counts["bad"] += 1
            issues[reason] += 1
            continue
        # heuristics for questionable samples
        meta = s.get("metadata", {})
        suspicious = False
        text = last_assistant_content(s)
        if len(text) > 2000:
            suspicious = True
        if meta.get("estimated_tokens", 0) > 8192:
            suspicious = True
        if "Plan:" in text and len(text.split("Plan:", 1)[1]) > 400:
            suspicious = True
        counts["questionable" if suspicious else "good"] += 1
    return {
        "inspected": len(picked),
        "classification": dict(counts),
        "issue_reasons": dict(issues),
        "note": "automated audit; manual human review recommended before production",
    }
