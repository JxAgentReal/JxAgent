# JxAgent Evaluation Protocol (exact future sequence)

The harness is offline-validated (dry-run). The real sequence below runs on
the MI300X evaluation host ONLY after the external dependencies listed in
EVALUATION_PATCH_REPORT.md are resolved. Nothing here is started by the
harness itself.

## Stage 0 — freeze (before training results are inspected; ~0 GPU·h)

```bash
# 0.1 pin the benchmark identity (fill every REQUIRES_EXTERNAL_VERIFICATION
#     field in evaluation/osworld_config.yaml from the pinned OSWorld commit)
# 0.2 canonical Verified task list -> evaluation/benchmark_task_lists/osworld_verified.json
# 0.3 generate + hash the frozen subset (deterministic; refuses silent regeneration)
python evaluation/make_frozen_subset.py --benchmark osworld_verified \
    --task-list evaluation/benchmark_task_lists/osworld_verified.json \
    --size 100 --seed 1337
# 0.4 contamination check of the FINAL published dataset against the pinned list
python evaluation/check_contamination.py \
    --dataset-root <JxAgentData_Run1_Final> \
    --task-list evaluation/benchmark_task_lists/osworld_verified.json \
    --benchmark osworld_verified \
    --out evaluation/contamination_report_verified.json
# 0.5 verify scaffold completeness and arm parity (no GPU)
python -c "from evaluation.scaffold import load_scaffold, assert_arm_parity; \
import yaml; s=load_scaffold(); a=yaml.safe_load(open('evaluation/osworld_config.yaml'))['defaults']['arms']; \
assert_arm_parity(s, a['base'], a['adapter']); print('parity OK')"
```

Acceptance: no unresolved Verified protocol fields; frozen subset status
"ready" (subset_sha256 recorded); contamination report clean or its
blockers consciously resolved; parity OK.

## Stage 1 — sanity (both arms, 5-10 tasks, <1 GPU·h)

Purpose: verify the real serving stack, parser, coordinate transforms and
scorer end-to-end BEFORE spending budget; measure per-task wall time.

```bash
python evaluation/run_osworld.py --arm base --output-dir runs/sanity_base \
    --frozen-subset <first 5-10 ids or a tiny sanity list> --tasks 5 \
    --confirm-real-benchmark --model-revision <base sha>
python evaluation/run_osworld.py --arm adapter --output-dir runs/sanity_adapter \
    --train-output-dir <train out> --checkpoint-gate 100 --total-steps <N> \
    --base-run-dir runs/sanity_base --contamination-report evaluation/contamination_report_verified.json \
    --confirm-real-benchmark --model-revision <base sha>
```

Acceptance: both arms complete with accounting_complete=true; zero
parser_failure attributable to the Plan format; per-task latency measured
(plug into the Stage 3/4 budget estimate: tasks x avg-steps x latency).

## Stage 2 — frozen-subset checkpoint comparison (~1-2 h wall at 8-way)

Run the BASE arm ONCE on the frozen subset; reuse it for every checkpoint
comparison (paired, same tasks):

```bash
python evaluation/run_osworld.py --arm base --output-dir runs/subset_base \
    --frozen-subset evaluation/osworld_verified_frozen_subset.json \
    --confirm-real-benchmark --model-revision <base sha>
# per checkpoint gate (20 / 55):
python evaluation/run_osworld.py --arm adapter --output-dir runs/subset_ckpt20 \
    --train-output-dir <train out> --checkpoint-gate 20 --total-steps <N> \
    --base-run-dir runs/subset_base --contamination-report <report> \
    --confirm-real-benchmark --model-revision <base sha>
python evaluation/run_regression.py --dry-run   # placeholder until providers run
```

Decision rule: compare subset_base vs subset_ckptXX with the paired
statistics in comparison.json (McNemar + bootstrap CI). Escalate to Stage 4
with the checkpoint that dominates on the subset; if 100% does not beat 20%/55%,
investigate early-peaking before the full run. Regression panel per
checkpoint; investigate >=3 pp, hard stop >=5 pp.

## Stage 3 — full local untouched base (one pass, full pinned task set)

```bash
python evaluation/run_osworld.py --arm base --output-dir runs/full_base \
    --task-list evaluation/benchmark_task_lists/osworld_verified.json \
    --contamination-report evaluation/contamination_report_verified.json \
    --confirm-real-benchmark --model-revision <base sha>
```

This is the ONLY authoritative baseline. The quoted 84.3 is never used.
Resumable: re-run the same command after interruption (completed tasks are
skipped). Archive runs/full_base entirely (manifest, tasks/, trajectories/,
aggregate.json).

## Stage 4 — full best adapter checkpoint

```bash
python evaluation/run_osworld.py --arm adapter --output-dir runs/full_adapter \
    --train-output-dir <train out> --checkpoint-gate <best> --total-steps <N> \
    --task-list evaluation/benchmark_task_lists/osworld_verified.json \
    --base-run-dir runs/full_base \
    --contamination-report evaluation/contamination_report_verified.json \
    --confirm-real-benchmark --model-revision <base sha>
```

comparison.json carries the paired statistics and the claim label. Only
BEATS_LOCAL_BASELINE or better is quotable, and POTENTIAL_SOTA requires the
full condition set (see sota_guard).

## Stage 5 — conditional repeat (only if statistically warranted)

Repeat ONLY when |delta| < 2xSE or McNemar p >= 0.05 AND the decision
matters; repeat BOTH arms (or target the discordant tasks identified in
comparison.json). Then run the failure-taxonomy manual annotation pass:

```bash
# annotate ambiguous model failures (grounding vs planning vs ...) for Run 2
# mining: failure_taxonomy.apply_manual_annotations over runs/*/tasks/*.json
```

## Invariants (enforced by code, restated for humans)

- The base and adapter arms differ ONLY in adapter loading (parity abort).
- The adapter is NEVER merged before evaluation (manifest records merged=false).
- Every run writes its manifest before the first task; comparative scoring
  refuses incompatible or missing base manifests.
- The strict denominator covers every expected task; failed launches are
  never silently excluded.
- No "Verified"/"SOTA" label while any Verified protocol field is unresolved.
- Dry-run results are never a baseline (guard rejects dry-run base manifests).
