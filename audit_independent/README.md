# Independent audit tooling (read-only)

Authored by the independent data auditor. These scripts never modify the
dataset or the project; they only read `<dataset-root>` and write into this
folder (`reports/`, review packs).

## Usage

```bash
# automated audit (Phases 1, 4*, 5, 6, 7, 9-16, 18, 19 of the audit brief)
.venv/Scripts/python.exe audit_independent/audit_final_dataset.py \
    --dataset-root <path-to-clean-downloaded-JxAgentData> --decode-samples 2000

# stratified manual-review sampler (Phases 2-3)
.venv/Scripts/python.exe audit_independent/sample_review_pack.py \
    --dataset-root <path-to-clean-downloaded-JxAgentData>
```

`*` Phase 4 (screenshot/action alignment) is only partially automatable; the
review checklist tells the human reviewer exactly what to verify per item.

## What audit_final_dataset.py checks (independently re-implemented)

It deliberately does NOT import builder code — action parsing, coordinate
math, and WebP header reading are re-implemented so a builder bug cannot
hide behind shared code.

| Area | Checks |
|---|---|
| Integrity | train/val counts vs stats.json + manifest; per-source counts vs 95,503 quota split; JSON validity; POSIX relative image paths; every referenced image exists; `<image>` placeholder count == image count (per sample); stats/manifest consistency |
| Duplicates | duplicate (source, trajectory_id, step_id); exact duplicate samples; duplicate image content (size+head hash); train/validation trajectory overlap; task-text repetition clusters |
| Coordinates | action syntax (independent parser); points within bounds of CLAIMED final size; decoded WebP dims vs claimed `final_image_size` (metadata lies); points within bounds of ACTUAL decoded dims; aspect-ratio consistency original vs final (flags silent coordinate-space mixing, e.g. non-1080p VideoCUA video or GUI-360 metadata mismatch); independent original→final conversion math check for both `norm_0_1000` and `pixel` spaces (±2 px) |
| Finish | counts/status per source; `finish(status="failure")` trained; finish anchored mid-trajectory (premature-finish candidates list) |
| Wait | wait share; waits without seconds; consecutive wait→wait pairs inside multi-turn samples (loop-bait) |
| Reasoning | rate vs 12% target; category distribution; reasoning on grounding/understanding/replay (must be 0); missing `Action:` line; `Plan: Plan:`; overlong plans |
| Windows | representation counts/shares vs 55/40/5 design; window length distribution; per-step message structure |
| Apps | top apps, HHI concentration, office share >45% flag, per-source single-app dominance >60% flag |
| Verbs | overall + per-source verb distributions; `move` anchored (hard fail); pathological shares |
| Replay | per-source counts vs 1600/1500/1700/1400/1300; VQA without image; license metadata present |
| Contamination | local 8-gram Jaccard + containment vs `.cache/osworld_instructions.json` (369 refs); any final sample ≥0.5 = decontamination failure (hard); 0.25–0.5 band exported for manual review (paraphrase leakage, ProCUA risk) |
| Tokens | mean/median/p90/p95/p99 overall, per source, per representation; epoch token total vs ~234M planning figure; samples over 8192 budget (hard fail) |
| Outliers | longest assistant targets; extreme sizes; empty targets; unexpected sources |

Severity model: **hard** = would make the dataset NOT READY as-is;
**soft** = needs explanation or a small patch. Exit code 1 on any hard
failure.

## What remains manual (reviewer instructions in the checklist)

- Phase 4 action↔screenshot alignment (target exists at the coordinates,
  typing matches the task, scroll direction, wait/finish justification,
  history coherence)
- Phase 8 recovery semantics (was the previous action actually ineffective?)
- Phase 14 replay answer correctness spot checks (math/coding/VQA)
- Phase 17 image readability of the smallest targets (list exported by the
  automated report)
