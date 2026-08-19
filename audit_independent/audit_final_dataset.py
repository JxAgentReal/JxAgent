#!/usr/bin/env python3
"""INDEPENDENT read-only audit of the final JxAgent dataset.

Written by the independent data auditor (not the dataset author). Deliberately
re-implements action parsing / coordinate math instead of importing the
builder's code, so builder bugs cannot hide behind shared code.

Usage:
    python audit_final_dataset.py --dataset-root <path-to-JxAgentData> \
        [--decode-samples 500] [--reports-dir reports]

Never writes anywhere except its own --reports-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# independent action parsing (final rendered syntax)
# ----------------------------------------------------------------------------
_V = r"(click|double_click|right_click|middle_click|move|point|drag|scroll|type|press|hotkey|key_down|key_up|mouse_down|mouse_up|wait|finish)"
_ACTION_RE = re.compile(r"^" + _V + r"\((.*)\)$", re.DOTALL)
_NUM = r"-?\d+(?:\.\d+)?"


def _split_args(s: str):
    parts, depth, quote, cur = [], 0, "", ""
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_action(text: str):
    """Independent parse of a final action string -> (verb, points, args)."""
    if not text:
        return None
    text = text.strip()
    m = _ACTION_RE.match(text)
    if not m:
        return None
    verb, args_str = m.group(1), m.group(2)
    parts = _split_args(args_str)
    kv = {}
    for p in parts:
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", p, re.DOTALL)
        if mm:
            k, v = mm.group(1), mm.group(2).strip()
            if v[:1] in "\"'":
                kv[k] = v.strip("\"'")
            else:
                try:
                    kv[k] = int(v)
                except ValueError:
                    try:
                        kv[k] = float(v)
                    except ValueError:
                        kv[k] = v
    pts = []
    if verb in ("click", "double_click", "right_click", "middle_click", "move", "point"):
        if "x" in kv and "y" in kv:
            pts = [(kv["x"], kv["y"])]
    elif verb == "drag":
        if all(k in kv for k in ("x1", "y1", "x2", "y2")):
            pts = [(kv["x1"], kv["y1"]), (kv["x2"], kv["y2"])]
    elif verb == "scroll":
        if "x" in kv and "y" in kv:
            pts = [(kv["x"], kv["y"])]
    return verb, pts, kv


def last_assistant(sample):
    for m in reversed(sample.get("messages", [])):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


_ACTION_LINE = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)


def extract_action(sample):
    c = last_assistant(sample).strip()
    if not c:
        return ""
    m = None
    for m in _ACTION_LINE.finditer(c):
        pass
    if m:
        return m.group(1).strip()
    lines = [l for l in c.splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def webp_dims(path: str):
    """Read WebP dimensions from the header without a full decode (VP8X aware)."""
    with open(path, "rb") as f:
        head = f.read(64)
    if len(head) < 30 or head[12:16] != b"VP8X":
        return None
    w = 1 + int.from_bytes(head[24:27], "little")
    h = 1 + int.from_bytes(head[27:30], "little")
    return w, h


def pct(values, p):
    if not values:
        return 0
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(math.ceil(p / 100.0 * len(values))) - 1))
    return values[k]


class Report:
    def __init__(self):
        self.hard = []   # would make the dataset NOT READY
        self.soft = []   # needs explanation / small patch
        self.info = Counter()
        self.notes = defaultdict(list)

    def hard_add(self, code, detail, count=1):
        self.hard.append({"code": code, "detail": detail, "count": count})

    def soft_add(self, code, detail, count=1):
        self.soft.append({"code": code, "detail": detail, "count": count})


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                out.append({"__bad__": True, "line": i + 1, "error": str(e)})
    return out


# ----------------------------------------------------------------------------
def audit(root, args, rep: Report):
    train_p = os.path.join(root, "final", "train.jsonl")
    val_p = os.path.join(root, "final", "validation.jsonl")
    for p in (train_p, val_p, os.path.join(root, "final", "manifest.json")):
        if not os.path.exists(p):
            rep.hard_add("missing_artifact", p)
            return None
    train = load_jsonl(train_p)
    val = load_jsonl(val_p)
    for s in train + val:
        if s.get("__bad__"):
            rep.hard_add("invalid_json", f"line {s['line']}: {s['error']}")
    train = [s for s in train if not s.get("__bad__")]
    val = [s for s in val if not s.get("__bad__")]
    all_samples = train + val
    rep.info["train"] = len(train)
    rep.info["validation"] = len(val)
    rep.info["total"] = len(all_samples)

    manifest = json.load(open(os.path.join(root, "final", "manifest.json"), encoding="utf-8"))
    stats_p = os.path.join(root, "final", "stats.json")
    stats = json.load(open(stats_p, encoding="utf-8")) if os.path.exists(stats_p) else {}

    # ---------------- PHASE 1: counts vs targets ----------------
    targets = {"procua": 46000, "gui360": 16000, "videocua": 17500,
               "groundcua": 4000, "pcagente": 4503, "replay": 7500}
    by_src = Counter(s.get("source", "?") for s in all_samples)
    rep.info["per_source"] = dict(by_src)
    for src, t in targets.items():
        got = by_src.get(src, 0)
        if src == "pcagente" and got < 4000:
            rep.soft_add("source_underfill", f"{src}: {got} vs quota {t} (gate may explain)", got)
        elif abs(got - t) > max(200, 0.03 * t):
            rep.soft_add("source_count_drift", f"{src}: {got} vs target {t}")
    total = len(all_samples)
    if not (95000 * 0.97 <= total <= 95000 * 1.10):
        rep.soft_add("total_out_of_band", f"total {total} vs ~95.3k expected")
    if stats and stats.get("total_samples") not in (None, total):
        rep.hard_add("stats_count_mismatch",
                     f"stats.json total {stats.get('total_samples')} != jsonl count {total}")

    # duplicates by identity key (the defect class fixed at 19:04 — verify)
    id_key = lambda s: (s.get("source"), s.get("trajectory_id"), s.get("step_id"))
    idc = Counter(id_key(s) for s in all_samples)
    dups = {k: v for k, v in idc.items() if v > 1}
    if dups:
        rep.hard_add("duplicate_sample_ids",
                     f"{len(dups)} keys duplicated, e.g. {list(dups)[:3]}", sum(dups.values()))
    # exact whole-line duplicates
    lc = Counter(json.dumps(s, sort_keys=True) for s in all_samples)
    n_exact = sum(v - 1 for v in lc.values() if v > 1)
    if n_exact:
        rep.hard_add("exact_duplicate_samples", f"{n_exact} exact duplicate samples", n_exact)

    # train/val trajectory overlap
    t_ids = {s.get("trajectory_id") for s in train}
    v_ids = {s.get("trajectory_id") for s in val}
    ov = t_ids & v_ids
    if ov:
        rep.hard_add("train_val_trajectory_overlap", f"{len(ov)} overlapping trajectories", len(ov))

    # ---------------- image references (Phase 1/18) ----------------
    img_ref_count = Counter()
    img_sizes_bytes = []
    missing = 0
    bad_path = 0
    ph_mismatch = 0
    for s in all_samples:
        msgs = s.get("messages", [])
        n_ph = sum(m.get("content", "").count("<image>")
                   for m in msgs if isinstance(m.get("content"), str))
        imgs = s.get("images", [])
        if n_ph != len(imgs):
            ph_mismatch += 1
        for p in imgs:
            img_ref_count[p] += 1
            if not isinstance(p, str) or p.startswith(("/", "\\")) or ".." in p or re.search(r"\\\\|\\", p):
                bad_path += 1
                continue
            full = os.path.join(root, *p.split("/"))
            if not os.path.exists(full):
                missing += 1
                if missing <= 5:
                    rep.notes["missing_images"].append(p)
            else:
                img_sizes_bytes.append(os.path.getsize(full))
    if missing:
        rep.hard_add("missing_image", f"{missing} referenced images missing", missing)
    if bad_path:
        rep.hard_add("non_portable_path", f"{bad_path} non-portable paths", bad_path)
    if ph_mismatch:
        rep.hard_add("placeholder_image_mismatch", f"{ph_mismatch} samples", ph_mismatch)
    rep.info["unique_images"] = len(img_ref_count)
    rep.info["total_image_bytes_mb"] = round(sum(img_sizes_bytes) / 1e6, 1)
    multi_ref = {p: c for p, c in img_ref_count.items() if c > 1}
    rep.info["images_referenced_by_multiple_samples"] = len(multi_ref)
    # duplicate image CONTENT under different names (cheap: size+head hash)
    sig = Counter()
    for p in list(img_ref_count)[:20000]:
        full = os.path.join(root, *p.split("/"))
        try:
            with open(full, "rb") as f:
                head = f.read(4096)
            sig[(os.path.getsize(full), hashlib.md5(head).hexdigest())] += 1
        except OSError:
            pass
    dup_content = sum(v - 1 for v in sig.values() if v > 1)
    rep.info["duplicate_image_content_candidates"] = dup_content

    # ---------------- action / coordinate audit (Phases 4,5) ----------------
    verbs = Counter()
    verbs_by_src = defaultdict(Counter)
    coord_bad = 0
    aspect_mismatch = []
    conv_bad = 0
    finish_mid_traj = []
    wait_sequences = []
    decode_n = 0
    rng = _rng()
    decode_pool = [s for s in all_samples if s.get("images")]
    decode_pick = set()
    for _ in range(min(args.decode_samples, len(decode_pool))):
        decode_pick.add(int(rng() * len(decode_pool)) % len(decode_pool))

    by_traj_steps = defaultdict(list)
    for i, s in enumerate(all_samples):
        src = s.get("source", "?")
        if src == "replay" and not str(s.get("task_type", "")).startswith("action"):
            continue
        act_txt = extract_action(s)
        pa = parse_action(act_txt)
        if pa is None:
            rep.hard_add("invalid_action_syntax", f"{src} step {s.get('step_id')}: {act_txt[:80]!r}")
            continue
        verb, pts, kv = pa
        verbs[verb] += 1
        verbs_by_src[src][verb] += 1
        meta = s.get("metadata", {}) or {}
        claimed_final = meta.get("final_image_size")
        by_traj_steps[s.get("trajectory_id")].append((s.get("step_id"), verb, i))

        # coordinate bounds vs claimed final size AND (for decoded) actual size
        if pts and claimed_final:
            w, h = int(claimed_final[0]), int(claimed_final[1])
            for (x, y) in pts:
                if not (0 <= x < w and 0 <= y < h):
                    coord_bad += 1

        # independent conversion check: original -> final must be consistent
        conv_bad += _check_conversion(s, verb, pts, kv, meta, conv_bad)

        # aspect-ratio consistency (claimed original vs claimed final)
        orig = meta.get("original_image_size")
        if orig and claimed_final and orig[0] and orig[1]:
            a1, a2 = orig[0] / orig[1], claimed_final[0] / claimed_final[1]
            if abs(a1 - a2) / max(a1, a2) > 0.01:
                aspect_mismatch.append({"src": src, "id": s.get("step_id"),
                                        "orig": orig, "final": claimed_final})

        # decode a sample of images: actual dims vs claimed final + true bounds
        if i in decode_pick and s.get("images"):
            img = s["images"][-1]
            full = os.path.join(root, *img.split("/"))
            if os.path.exists(full):
                dims = webp_dims(full)
                decode_n += 1
                if dims and claimed_final and tuple(map(int, claimed_final)) != tuple(dims):
                    rep.hard_add("final_size_metadata_lie",
                                 f"{s.get('step_id')}: claimed {claimed_final} actual {dims}")
                if dims and pts:
                    for (x, y) in pts:
                        if not (0 <= x < dims[0] and 0 <= y < dims[1]):
                            rep.hard_add("coordinate_out_of_bounds_actual",
                                         f"{s.get('step_id')} point ({x},{y}) vs {dims}")
                if dims and orig:
                    a1, a2 = orig[0] / orig[1], dims[0] / dims[1]
                    if abs(a1 - a2) / max(a1, a2) > 0.01:
                        aspect_mismatch.append({"src": src, "id": s.get("step_id"),
                                                "orig": orig, "actual": dims})

    if coord_bad:
        rep.hard_add("coordinate_out_of_bounds_claimed", f"{coord_bad} samples", coord_bad)
    if aspect_mismatch:
        rep.soft_add("aspect_ratio_mismatch",
                     f"{len(aspect_mismatch)} samples where original vs final aspect differs >1% "
                     f"(coordinate-space mismatch suspicion; gui360/videocua)", len(aspect_mismatch))
        rep.notes["aspect_mismatch_examples"] = aspect_mismatch[:10]
    if conv_bad:
        rep.hard_add("coordinate_conversion_mismatch",
                     f"{conv_bad} samples where rendered point != original->final math", conv_bad)
    rep.info["decoded_images_checked"] = decode_n

    # ---------------- finish audit (Phase 6) ----------------
    finish_samples = []
    for s in all_samples:
        if s.get("source") == "replay":
            continue
        pa = parse_action(extract_action(s))
        if pa and pa[0] == "finish":
            finish_samples.append(s)
    fstatus = Counter()
    for s in finish_samples:
        pa = parse_action(extract_action(s))
        fstatus[pa[2].get("status", "?")] += 1
        tid = s.get("trajectory_id")
        # finish not at end of its own trajectory => premature candidate
        steps = by_traj_steps.get(tid, [])
        if steps and s.get("step_id") != steps[-1][0]:
            finish_mid_traj.append(s.get("step_id"))
    rep.info["finish_total"] = len(finish_samples)
    rep.info["finish_status_distribution"] = dict(fstatus)
    rep.info["finish_share_pct"] = round(100 * len(finish_samples) / max(1, len(all_samples)), 2)
    if fstatus.get("failure"):
        rep.soft_add("finish_failure_status_trained",
                     f"{fstatus['failure']} finish(status=\"failure\") targets trained", fstatus["failure"])
    rep.info["finish_mid_trajectory_candidates"] = len(finish_mid_traj)
    rep.notes["finish_mid_trajectory_examples"] = finish_mid_traj[:10]

    # ---------------- wait audit (Phase 7) ----------------
    waits = [s for s in all_samples
             if (pa := parse_action(extract_action(s))) and pa[0] == "wait"]
    rep.info["wait_total"] = len(waits)
    wait_zero = sum(1 for s in waits if (pa := parse_action(extract_action(s))) and "seconds" not in pa[2])
    rep.info["wait_without_seconds"] = wait_zero
    # consecutive waits inside one multi-turn sample
    consec = 0
    for s in all_samples:
        acts = []
        for m in s.get("messages", []):
            if m.get("role") == "assistant":
                pa = parse_action(m.get("content", "").strip().splitlines()[-1] if m.get("content") else "")
                if pa:
                    acts.append(pa[0])
        for a, b in zip(acts, acts[1:]):
            if a == "wait" and b == "wait":
                consec += 1
    rep.info["consecutive_wait_pairs_in_windows"] = consec
    if consec:
        rep.soft_add("consecutive_wait_pairs", f"{consec} wait->wait pairs in multi-turn samples", consec)

    # ---------------- reasoning audit (Phase 9) ----------------
    reasoning = [s for s in all_samples if last_assistant(s).lstrip().startswith("Plan:")]
    action_samples = [s for s in all_samples if s.get("task_type") == "action"]
    rep.info["reasoning_total"] = len(reasoning)
    rep.info["reasoning_rate_pct_of_action"] = round(
        100 * len([s for s in reasoning if s.get("task_type") == "action"]) / max(1, len(action_samples)), 2)
    rcats = Counter((s.get("metadata", {}) or {}).get("reasoning_category", "?") for s in reasoning)
    rep.info["reasoning_categories"] = dict(rcats)
    bad_reasoning = [s for s in reasoning if s.get("task_type") in ("grounding", "screen_understanding")
                     or s.get("source") == "replay"]
    if bad_reasoning:
        rep.hard_add("reasoning_on_non_action", f"{len(bad_reasoning)} samples", len(bad_reasoning))
    no_action_line = sum(1 for s in reasoning if "Action:" not in last_assistant(s))
    if no_action_line:
        rep.hard_add("reasoning_missing_action_line", f"{no_action_line} samples", no_action_line)
    long_reasoning = sum(1 for s in reasoning if len(last_assistant(s)) > 400)
    if long_reasoning:
        rep.soft_add("reasoning_too_long", f"{long_reasoning} > 400 chars", long_reasoning)
    dup_plan = sum(1 for s in reasoning if "Plan: Plan:" in last_assistant(s))
    if dup_plan:
        rep.hard_add("double_plan_prefix", f"{dup_plan} samples", dup_plan)

    # ---------------- multi-step / representation (Phase 10) ----------------
    reps = Counter((s.get("metadata", {}) or {}).get("representation", "?") for s in all_samples)
    rep.info["representation_counts"] = dict(reps)
    n = max(1, len(all_samples))
    rep.info["representation_shares_pct"] = {k: round(100 * v / n, 1) for k, v in reps.items()}
    drift = abs(reps.get("window", 0) / n - 0.40)
    rep.info["window_share_vs_design"] = {"design": 0.40, "actual": round(reps.get("window", 0) / n, 4)}
    win_len = Counter((s.get("metadata", {}) or {}).get("window_length") for s in all_samples
                      if (s.get("metadata", {}) or {}).get("representation") in ("window", "chunk"))
    rep.info["window_length_distribution"] = {str(k): v for k, v in sorted(win_len.items(), key=lambda x: (x[0] is None, x[0]))}

    # ---------------- apps (Phase 12) ----------------
    apps = Counter((s.get("metadata", {}) or {}).get("app") or "?" for s in all_samples)
    rep.info["top_apps"] = dict(apps.most_common(25))
    total_a = sum(apps.values())
    hhi = sum((v / total_a) ** 2 for v in apps.values())
    rep.info["app_hhi"] = round(hhi, 4)
    office = sum(v for k, v in apps.items() if k and any(x in str(k).lower() for x in
                ("excel", "word", "ppt", "powerpoint", "libreoffice", "office")))
    rep.info["office_share_pct"] = round(100 * office / total_a, 1)
    if office / total_a > 0.45:
        rep.soft_add("office_overrep", f"office-family share {100*office/total_a:.1f}%")
    for src in ("gui360", "videocua", "groundcua"):
        sub = Counter((s.get("metadata", {}) or {}).get("app") or "?" for s in all_samples
                      if s.get("source") == src)
        if sub:
            top, cnt = sub.most_common(1)[0]
            if cnt / sum(sub.values()) > 0.60 and sum(sub.values()) > 200:
                rep.soft_add("source_app_dominance", f"{src}: {top} = {100*cnt/sum(sub.values()):.1f}%")

    # ---------------- verb distribution (Phase 13) ----------------
    rep.info["verb_distribution"] = dict(verbs.most_common())
    rep.info["verbs_by_source"] = {k: dict(v.most_common(8)) for k, v in verbs_by_src.items()}
    if verbs.get("move", 0) / max(1, sum(verbs.values())) > 0.02:
        rep.hard_add("move_verb_anchored", "move actions must never anchor samples")

    # ---------------- replay (Phase 14) ----------------
    repsub = Counter((s.get("metadata", {}) or {}).get("replay_source", "?") for s in all_samples
                     if s.get("source") == "replay")
    rep.info["replay_sources"] = dict(repsub)
    rtypes = Counter(s.get("task_type") for s in all_samples if s.get("source") == "replay")
    rep.info["replay_task_types"] = dict(rtypes)
    vqa_no_img = sum(1 for s in all_samples if s.get("task_type") == "replay_vqa" and not s.get("images"))
    if vqa_no_img:
        rep.hard_add("vqa_without_image", f"{vqa_no_img} samples", vqa_no_img)
    no_license = sum(1 for s in all_samples if s.get("source") == "replay"
                     and not (s.get("metadata", {}) or {}).get("license"))
    if no_license:
        rep.soft_add("replay_license_missing", f"{no_license} samples", no_license)

    # ---------------- contamination (Phase 15) ----------------
    ref_path = args.osworld_cache
    if os.path.exists(ref_path):
        pairs = json.load(open(ref_path, encoding="utf-8"))
        word_re = re.compile(r"[a-z0-9]+")

        def norm(t):
            return " ".join(word_re.findall((t or "").lower()))

        def shingles(t, n=8):
            w = norm(t).split()
            if len(w) < n:
                return {" ".join(w)} if w else set()
            return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

        refs = [(tid, shingles(instr), set(norm(instr).split())) for tid, instr in pairs]
        exact_removed_still_present = 0
        high = []
        band = []
        sim_hist = Counter()
        for s in all_samples:
            task = ""
            for m in s.get("messages", []):
                if m.get("role") == "user":
                    for part in re.findall(r"Task:\s*(.*)", str(m.get("content", "")))[:1]:
                        task = part
                    if not task and "<image>" not in str(m.get("content", "")):
                        c = str(m.get("content", ""))
                        if len(c) < 600:
                            task = c
                    break
            if not task:
                continue
            cand = shingles(task)
            words = set(norm(task).split())
            short = len(words) < 8
            best, best_id = 0.0, ""
            for tid, rsh, rw in refs:
                if not cand or not rsh:
                    continue
                inter = len(cand & rsh)
                j = inter / (len(cand) + len(rsh) - inter) if inter else 0.0
                c = (len(words & rw) / max(1, len(words))) if short else \
                    (len(cand & rsh) / len(cand) if cand else 0.0)
                sc = max(j, c)
                if sc > best:
                    best, best_id = sc, tid
            b = round(best, 3)
            sim_hist[min(9, int(b * 10)) / 10] += 1
            if best >= 0.5:
                high.append({"id": s.get("step_id"), "src": s.get("source"),
                             "sim": b, "ref": best_id, "task": task[:120]})
            elif best >= 0.25:
                band.append({"id": s.get("step_id"), "src": s.get("source"),
                             "sim": b, "ref": best_id, "task": task[:120]})
        rep.info["contamination_similarity_histogram"] = {str(k): v for k, v in sorted(sim_hist.items())}
        rep.info["contamination_ge_0.5"] = len(high)
        rep.notes["contamination_high_examples"] = high[:20]
        rep.notes["contamination_band_0.25_0.5_examples"] = band[:30]
        if high:
            rep.hard_add("osworld_similarity_ge_threshold",
                         f"{len(high)} final samples with similarity >= 0.5 vs OSWorld refs "
                         f"(decontamination failed)", len(high))
    else:
        rep.soft_add("no_osworld_reference_cache", ref_path)

    # ---------------- tokens (Phase 16) ----------------
    def tok_stats(samples):
        v = [int((s.get("metadata", {}) or {}).get("estimated_tokens") or 0) for s in samples]
        v = [x for x in v if x]
        if not v:
            return {}
        return {"n": len(v), "mean": round(sum(v) / len(v)), "median": pct(v, 50),
                "p90": pct(v, 90), "p95": pct(v, 95), "p99": pct(v, 99), "max": max(v)}
    rep.info["token_stats_overall"] = tok_stats(all_samples)
    for src in sorted({s.get("source") for s in all_samples}):
        rep.info[f"token_stats_{src}"] = tok_stats([s for s in all_samples if s.get("source") == src])
    for r_ in ("single", "window", "chunk", "grounding", "understanding", "replay"):
        rep.info[f"token_stats_repr_{r_}"] = tok_stats(
            [s for s in all_samples if (s.get("metadata", {}) or {}).get("representation") == r_])
    est_epoch = sum(int((s.get("metadata", {}) or {}).get("estimated_tokens") or 0) for s in all_samples)
    rep.info["estimated_epoch_tokens"] = est_epoch
    over = [s for s in all_samples if int((s.get("metadata", {}) or {}).get("estimated_tokens") or 0) > 8192]
    if over:
        rep.hard_add("over_context_budget", f"{len(over)} samples est > 8192", len(over))

    # ---------------- outliers (Phase 18) ----------------
    longest = sorted(all_samples, key=lambda s: -len(last_assistant(s)))[:5]
    rep.notes["longest_assistant_examples"] = [
        {"id": s.get("step_id"), "len": len(last_assistant(s))} for s in longest]
    task_dup = Counter()
    for s in all_samples:
        for m in s.get("messages", []):
            if m.get("role") == "user":
                mm = re.search(r"Task:\s*(.{0,120})", str(m.get("content", "")))
                if mm:
                    task_dup[mm.group(1)] += 1
                break
    dup_tasks = {k: v for k, v in task_dup.items() if v > 4 and len(k) > 40}
    rep.info["task_text_repeated_gt4"] = len(dup_tasks)
    rep.notes["task_dup_examples"] = list(dup_tasks)[:10]

    # ---------------- regression-risk proxies (Phase 19) ----------------
    cu = [s for s in all_samples if s.get("source") != "replay"]
    rep.info["risk_proxies"] = {
        "finish_share_pct": round(100 * len(finish_samples) / max(1, len(cu)), 2),
        "wait_share_pct": round(100 * len(waits) / max(1, len(cu)), 2),
        "move_share_pct": round(100 * verbs.get("move", 0) / max(1, sum(verbs.values())), 3),
        "reasoning_rate_pct": rep.info["reasoning_rate_pct_of_action"],
        "window_share_pct": rep.info["representation_shares_pct"].get("window", 0),
    }
    return all_samples


def _check_conversion(s, verb, pts, kv, meta, state):
    """Independent original->final coordinate math check.

    For norm_0_1000 sources: final == original/1000 * actual_final_size (+round).
    For pixel sources: final == original * final/claimed_original (+round).
    Returns number of mismatches (only counts when all inputs are present).
    """
    if not pts or verb in ("scroll",):
        return 0
    orig = meta.get("action_original") or ""
    space = meta.get("action_original_space")
    fin = meta.get("final_image_size")
    osize = meta.get("original_image_size")
    if not (space and fin and osize):
        return 0
    m = re.findall(r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)", orig)
    if len(m) < len(pts):
        return 0
    bad = 0
    fw, fh = float(fin[0]), float(fin[1])
    ow, oh = float(osize[0]), float(osize[1])
    for (x, y), (ox, oy) in zip(pts, m):
        ox, oy = float(ox), float(oy)
        if space == "norm_0_1000":
            ex, ey = ox / 1000.0 * fw, oy / 1000.0 * fh
        else:  # pixel space in original image
            ex, ey = ox * fw / ow, oy * fh / oh
        if abs(x - ex) > 2.0 or abs(y - ey) > 2.0:
            bad += 1
    return bad


def _rng(seed=12345):
    state = [seed]

    def nxt():
        state[0] = (1103515245 * state[0] + 12345) % (1 << 31)
        return state[0] / float(1 << 31)
    return nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--decode-samples", type=int, default=500)
    ap.add_argument("--osworld-cache", default=os.path.join(os.path.dirname(__file__), "..", ".cache", "osworld_instructions.json"))
    ap.add_argument("--reports-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"))
    args = ap.parse_args()

    rep = Report()
    t0 = time.time()
    audit(args.dataset_root, args, rep)
    rep.info["audit_seconds"] = round(time.time() - t0, 1)
    rep.info["dataset_root"] = os.path.abspath(args.dataset_root)

    os.makedirs(args.reports_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    root_tag = os.path.basename(os.path.normpath(args.dataset_root)) or "dataset"
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hard_failures": rep.hard,
        "soft_findings": rep.soft,
        "info": dict(rep.info),
        "notes": {k: v for k, v in rep.notes.items()},
    }
    jpath = os.path.join(args.reports_dir, f"audit_{root_tag}_{ts}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    mpath = os.path.join(args.reports_dir, f"audit_{root_tag}_{ts}.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write("# Independent dataset audit (automated portion)\n\n")
        f.write(f"Generated: {out['generated']}  \nRoot: `{rep.info['dataset_root']}`\n\n")
        f.write("## HARD failures\n\n")
        f.write("\n".join(f"- `{h['code']}`: {h['detail']}" for h in rep.hard) or "- none")
        f.write("\n\n## Soft findings\n\n")
        f.write("\n".join(f"- `{h['code']}`: {h['detail']}" for h in rep.soft) or "- none")
        f.write("\n\n## Key stats\n\n")
        f.write("```json\n" + json.dumps(rep.info, ensure_ascii=False, indent=1) + "\n```\n")
    print(f"[audit] hard={len(rep.hard)} soft={len(rep.soft)} -> {jpath}")
    return 1 if rep.hard else 0


if __name__ == "__main__":
    sys.exit(main())
