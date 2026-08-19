#!/usr/bin/env python3
"""SOTA claim guard: a higher number is never sufficient.

Labels (in strength order):
  NOT_COMPARABLE           - prerequisites missing; no comparative claim
  BEATS_LOCAL_BASELINE     - statistically supported win over the LOCAL base
                             run under an identical, manifest-verified scaffold
  COMPARABLE_TO_PUBLISHED_RESULT - additionally, scaffold equivalence to the
                             published number's conditions is demonstrated
  POTENTIAL_SOTA           - additionally, the Verified protocol is fully
                             pinned and no contamination blocker remains

sota_claim_allowed is YES only under POTENTIAL_SOTA with every condition met.
"""
from __future__ import annotations

from typing import List, Optional

from evaluation.benchmarks import protocol_resolved, unresolved_fields
from evaluation.run_manifest import check_compatibility
from evaluation.stats import paired_comparison


def evaluate_claim(*, base_manifest: dict, adapter_manifest: dict,
                   base_outcomes: dict, adapter_outcomes: dict,
                   base_aggregate: Optional[dict] = None,
                   adapter_aggregate: Optional[dict] = None,
                   benchmark: Optional[dict] = None,
                   contamination_blockers: Optional[List[dict]] = None,
                   published_scaffold_equivalence: Optional[dict] = None,
                   significance_alpha: float = 0.05) -> dict:
    conditions = {}
    failed: List[str] = []

    def require(name: str, ok: bool, detail=None):
        conditions[name] = {"met": bool(ok), "detail": detail}
        if not ok:
            failed.append(name)

    compat_ok, diffs = check_compatibility(base_manifest, adapter_manifest)
    require("same_benchmark_identity_and_task_set",
            compat_ok and base_manifest.get("benchmark", {}).get("name") ==
            adapter_manifest.get("benchmark", {}).get("name"),
            diffs or None)
    require("same_step_budget",
            base_manifest.get("step_budget") == adapter_manifest.get("step_budget"))
    require("same_observation_protocol",
            base_manifest.get("observation") == adapter_manifest.get("observation"))
    require("same_action_interface",
            base_manifest.get("action_space") == adapter_manifest.get("action_space"))
    require("same_scoring_protocol",
            base_manifest.get("benchmark", {}).get("repository_revision") ==
            adapter_manifest.get("benchmark", {}).get("repository_revision"))
    require("local_base_run_exists",
            base_manifest is not None and not base_manifest.get("dry_run", False),
            "quoted published numbers are never a baseline")
    acct_ok = True
    if adapter_aggregate is not None:
        acct_ok = bool(adapter_aggregate.get("accounting_complete"))
    if base_aggregate is not None:
        acct_ok = acct_ok and bool(base_aggregate.get("accounting_complete"))
    require("complete_denominator_accounting", acct_ok,
            "strict denominator must cover every expected task")
    require("no_unresolved_contamination_blocker",
            not contamination_blockers, contamination_blockers or None)
    require("matching_environment_metadata",
            base_manifest.get("environment_revision") ==
            adapter_manifest.get("environment_revision") and
            base_manifest.get("vm_identifiers") == adapter_manifest.get("vm_identifiers"))

    stats = paired_comparison(base_outcomes, adapter_outcomes)
    delta_positive = stats["delta_percentage_points"] > 0
    significant = stats["likely_meaningful"]
    require("statistically_supported_win", delta_positive and significant,
            {"delta_pp": stats["delta_percentage_points"],
             "mcnemar_p": stats["mcnemar_exact_p"],
             "ci95": stats["bootstrap_delta_ci_95_pp"]})

    if failed:
        label = "NOT_COMPARABLE"
    else:
        label = "BEATS_LOCAL_BASELINE"
        pub_ok = bool(published_scaffold_equivalence and
                      published_scaffold_equivalence.get("demonstrated"))
        require("published_scaffold_equivalence_demonstrated", pub_ok,
                published_scaffold_equivalence or
                "step budget, task set, observation format and action interface "
                "of the published number must be shown equivalent, not assumed")
        if not failed:
            label = "COMPARABLE_TO_PUBLISHED_RESULT"
            bm = benchmark or {}
            require("verified_protocol_fully_pinned", protocol_resolved(bm),
                    {"unresolved_fields": unresolved_fields(bm)})
            if not failed:
                label = "POTENTIAL_SOTA"

    return {
        "label": label,
        "sota_claim_allowed": "YES" if label == "POTENTIAL_SOTA" else "NO",
        "conditions": conditions,
        "failed_conditions": failed,
        "statistics": stats,
        "note": ("A score is NOT SOTA when harness, prompts, action interface, "
                 "task set or step budget differ from the compared result."),
    }
