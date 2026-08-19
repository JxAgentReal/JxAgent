"""GroundCUA adapter (ServiceNow/GroundCUA, MIT).

Verified schema (2026-08):
    data/<Platform>/<sha256>.json  -> list of {image_path, bbox [x1,y1,x2,y2]
                                      (float PIXELS), text, category, id}
    images/<Platform>/<sha256>.png -> screenshot (same hash name)
    instruction_tuning.tar.gz      -> derived subset, NOT used here

Selection policy (spec section 10): difficult examples only — tiny targets,
dense interfaces, similar adjacent controls; size buckets with small-target
oversampling; application diversity via platform round-robin; obvious giant
buttons avoided.

Storage: annotation JSONs are fetched per file (KBs); the image PNG header
(first bytes) is fetched via range to learn dimensions; the full image is
downloaded ONLY when the example is selected.
"""
from __future__ import annotations

import io
import json
import math
from typing import Dict, Iterator, List, Optional, Tuple

from processing.assemble import assemble_grounding
from processing.images import load_image, png_dimensions
from processing.remote_access import fetch_bytes, hf_tree, hf_url
from .common import BuildContext
from .revisions import source_revision

REPO = "ServiceNow/GroundCUA"
REVISION = source_revision("groundcua")  # immutable Run 1 snapshot

# Target-size bucket definitions and quota shares live ONCE in
# processing/quality.py (GROUNDCUA_SIZE_BUCKETS / BucketQuota) — the old
# divergent local copy (35/30/20/15) was dead code and is removed.

GIANT_PX = 300          # skip elements wider than this (obvious giant buttons)
MIN_W_PX, MIN_H_PX = 4, 4

# coarse categories we prefer (dense professional controls)
PREFERRED_CATEGORIES = {"Button", "Menu", "Input Elements", "Navigation",
                        "Sidebar", "Visual Elements", "Information Display"}
LOW_INFO_TEXT = {"cursor", "", "icon", "image", "logo"}


def usable_candidates(entries: List[dict]) -> List[Tuple[float, float, str, dict]]:
    """(w, h, text, entry) for every element usable as a grounding target."""
    out = []
    for e in entries:
        bbox = e.get("bbox") or []
        if len(bbox) != 4:
            continue
        w = float(bbox[2]) - float(bbox[0])
        h = float(bbox[3]) - float(bbox[1])
        if w < MIN_W_PX or h < MIN_H_PX or w > GIANT_PX:
            continue
        text = str(e.get("text") or "").strip()
        if text.lower() in LOW_INFO_TEXT:
            continue
        out.append((w, h, text, e))
    return out


def select_element(entries: List[dict],
                    quota: "BucketQuota|None" = None) -> Optional[dict]:
    """Pick the most training-worthy element of one screenshot.

    With a BucketQuota, preference goes to the size bucket currently most
    under-represented (audit 2026-08: the old scorer picked 63/70 'small',
    starving tiny and medium). The quota is a PREFERENCE, not enforcement:
    buckets scarce in the source are simply under-delivered (reported), never
    padded with duplicates."""
    from processing.quality import grounding_bucket, score_grounding
    candidates = usable_candidates(entries)
    # Ambiguous visible referents are avoidable supervision noise. Keep only
    # text+category pairs that identify exactly one visible element on screen.
    from collections import Counter
    ref_counts = Counter((t.casefold(), str(e.get("category") or "").casefold())
                         for _w, _h, t, e in candidates)
    candidates = [(w, h, t, e) for w, h, t, e in candidates
                  if ref_counts[(t.casefold(), str(e.get("category") or "").casefold())] == 1]

    def base_score(w, h, text, e):
        qs = score_grounding(target_width_px=int(w), target_height_px=int(h),
                             text=text, category=e.get("category"), app="any")
        return qs.score

    if quota is None:
        best, best_score = None, -1.0
        for w, h, text, e in candidates:
            s = base_score(w, h, text, e)
            if s > best_score:
                best, best_score = e, s
        return best

    # rank buckets by deficit (share - realized), pick best element within
    order = sorted(quota.shares,
                   key=lambda b: quota.shares[b] - (quota.counts.get(b, 0) / max(1, sum(quota.counts.values()))),
                   reverse=True)
    for want_bucket in order:
        pool = [(w, h, t, e) for w, h, t, e in candidates
                if grounding_bucket(int(w), int(h)) == want_bucket]
        if not pool:
            continue
        best, best_score = None, -1.0
        for w, h, text, e in pool:
            s = base_score(w, h, text, e)
            if s > best_score:
                best, best_score = e, s
        return best
    return None


def platform_directories(ctx: BuildContext) -> List[str]:
    tree = hf_tree(REPO, "data", revision=REVISION, session=ctx.http())
    return [t["path"].split("/")[-1] for t in tree if t.get("type") == "directory"]


def list_annotation_files(ctx: BuildContext, platform: str, limit: int,
                          offset: int = 0) -> List[str]:
    """One page of annotation filenames for a platform."""
    tree = hf_tree(REPO, f"data/{platform}", revision=REVISION, session=ctx.http())
    names = [t["path"] for t in tree if t.get("type") == "file" and t["path"].endswith(".json")]
    return names[offset:offset + limit]


def png_header_dimensions(ctx: BuildContext, platform: str, sha: str) -> Optional[Tuple[int, int]]:
    """Fetch only the PNG IHDR header (33 bytes) via range request."""
    url = hf_url(REPO, f"images/{platform}/{sha}.png", revision=REVISION)
    rf = None
    try:
        from processing.remote_access import HTTPRangeFile
        rf = HTTPRangeFile(url, ctx.http())
        head = rf.read(33)
        return png_dimensions(head)
    except Exception:
        return None
    finally:
        if rf is not None:
            try:
                rf.close()
            except Exception:
                pass


def describe_element(entry: dict) -> str:
    text = str(entry.get("text") or "").strip()
    cat = str(entry.get("category") or "").strip().lower()
    if text and cat:
        return f"{text} ({cat})"
    return text or cat or "indicated control"


def run(ctx: BuildContext) -> List[dict]:
    """Round-robin platforms; per screenshot select one difficult element."""
    remaining = int(ctx.quota.get("groundcua", 0))
    full_target = int(ctx.state.selected_total("groundcua")) + remaining
    ctx.state.set_target("groundcua", full_target)
    out: List[dict] = []
    from processing.quality import BucketQuota, grounding_bucket
    platform_counts: Dict[str, int] = dict(
        (ctx.config.get("_resume_app_counts", {}) or {}).get("groundcua", {}))
    size_quota = BucketQuota()
    try:
        platforms = platform_directories(ctx)
    except Exception:
        if ctx.offline:
            platforms = []
        else:
            raise
    # deterministic order; smaller index = more diverse start
    platforms = sorted(platforms)
    # Resume invariant: application diversity is defined against the original
    # full cohort, never the smaller remaining tail after a crash.  Derive the
    # cap from the *actual* platform count and leave 20% reachability headroom
    # so an exact 4k quota is not mathematically impossible under uneven source
    # availability.
    import math
    if platforms:
        min_reachable = math.ceil(full_target / len(platforms))
        per_platform_quota = max(20, math.ceil(min_reachable * 1.20))
    else:
        per_platform_quota = max(20, full_target)
    page_size = 50
    cursor = {p: 0 for p in platforms}
    progress = True
    while ctx.remaining("groundcua") > 0 and progress and platforms:
        progress = False
        for platform in platforms:
            if ctx.remaining("groundcua") <= 0:
                break
            if platform_counts.get(platform, 0) >= per_platform_quota:
                continue
            files = list_annotation_files(ctx, platform, page_size, cursor[platform])
            cursor[platform] += page_size
            if files:
                progress = True
            for path in files:
                if ctx.remaining("groundcua") <= 0:
                    break
                if platform_counts.get(platform, 0) >= per_platform_quota:
                    break
                if ctx.state.is_shard_done("groundcua", path):
                    continue
                sha = path.rsplit("/", 1)[-1].removesuffix(".json")
                key = f"groundcua::{platform}::{sha}"
                if ctx.state.is_shard_done("groundcua", key):
                    continue
                try:
                    ann = json.loads(fetch_bytes(hf_url(REPO, path, revision=REVISION), session=ctx.http()).decode("utf-8"))
                except Exception:
                    ctx.reject("groundcua", "annotation_fetch_failed_retryable")
                    continue
                # available distribution (what the source actually contains)
                # vs the selected distribution — reported, quotas stay soft
                for cw, _ch, _t, _e in usable_candidates(ann):
                    ctx.note_stat("groundcua",
                                  f"bucket_available_{grounding_bucket(int(cw))}", 1)
                entry = select_element(ann, quota=size_quota)
                if entry is None:
                    ctx.reject("groundcua", "no_suitable_element")
                    ctx.state.mark_shard_done("groundcua", key)
                    continue
                dims = png_header_dimensions(ctx, platform, sha)
                if not dims:
                    ctx.reject("groundcua", "image_header_failed_retryable")
                    continue
                x1, y1, x2, y2 = [float(v) for v in entry["bbox"]]
                # GroundCUA contains tiny floating-point edge spill (audit max
                # ~1.3px). Clamp only that proven annotation noise; larger
                # excursions remain a hard correctness failure.
                eps = 1.5
                if (x1 < -eps or y1 < -eps or x2 > (dims[0] - 1) + eps or
                        y2 > (dims[1] - 1) + eps or x2 <= x1 or y2 <= y1):
                    ctx.reject("groundcua", "coordinate_out_of_bounds")
                    ctx.state.mark_shard_done("groundcua", key)
                    continue
                x1 = min(max(x1, 0.0), float(dims[0] - 1))
                x2 = min(max(x2, 0.0), float(dims[0] - 1))
                y1 = min(max(y1, 0.0), float(dims[1] - 1))
                y2 = min(max(y2, 0.0), float(dims[1] - 1))
                if x2 <= x1 or y2 <= y1:
                    ctx.reject("groundcua", "degenerate_bbox_after_clip")
                    ctx.state.mark_shard_done("groundcua", key)
                    continue
                target_w = int(round(x2 - x1))
                target_h = int(round(y2 - y1))
                cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
                instruction = describe_element(entry)
                # near-duplicate screen protection is applied on the image
                try:
                    img = fetch_bytes(hf_url(REPO, f"images/{platform}/{sha}.png", revision=REVISION),
                                      session=ctx.http(), max_bytes=32 << 20)
                except Exception:
                    ctx.reject("groundcua", "image_fetch_failed_retryable")
                    continue
                from processing.dedup import phash
                h = phash(load_image(img))
                dup, reason = ctx.dedup.consider(image_phash=h, signals=["small_target"],
                                                 task_text=f"point to {instruction}",
                                                 action_text=f"point({cx},{cy})")
                if dup:
                    ctx.reject("groundcua", reason)
                    ctx.state.mark_shard_done("groundcua", key)
                    continue
                sample = assemble_grounding(
                    source="groundcua", trajectory_id=f"groundcua_{platform}_{sha[:16]}",
                    step_id=sha[:24], image_bytes=img, instruction=instruction,
                    target_xy=(cx, cy), image_size=dims, target_width_px=target_w,
                    target_height_px=target_h, app=platform, ctx=ctx,
                    extra_meta={"element_category": entry.get("category"),
                                "bbox_original": [x1, y1, x2, y2],
                                "group_id": f"groundcua::{platform}",
                                "referent_unique": True})
                if sample is None:
                    ctx.state.mark_shard_done("groundcua", key)
                    continue
                out.append(sample)
                size_quota.record(grounding_bucket(target_w, target_h))
                ctx.note_stat("groundcua", f"bucket_selected_{grounding_bucket(target_w, target_h)}")
                ctx.consume("groundcua")
                platform_counts[platform] = platform_counts.get(platform, 0) + 1
                ctx.persist_samples([sample])
                ctx.state.add_selected("groundcua")
                ctx.state.mark_shard_done("groundcua", key)
                if len(out) % 5 == 0:
                    print(f"[groundcua] {len(out)} selected "
                          f"({ctx.remaining('groundcua')} to go)", flush=True)
                    ctx.state.save()
        ctx.state.save()
    return out
