#!/usr/bin/env python3
"""JxAgent evaluation runner (base arm / adapter arm, identical scaffold).

Offline dry-run is the primary validation mode:
  python evaluation/run_agent.py --dry-run --arm base \
      --output-dir ./tmp_dryrun/base --tasks 12
  python evaluation/run_agent.py --dry-run --arm adapter \
      --output-dir ./tmp_dryrun/adapter --tasks 12 \
      --base-run-dir ./tmp_dryrun/base

Real benchmark runs are entered via evaluation/run_osworld.py and are refused
here unless every benchmark protocol field is pinned and --confirm-real-benchmark
is set. Comparative scoring always requires a compatible LOCAL base run
manifest (baseline-first); --allow-without-baseline exists for syntax testing
only and marks results NOT_COMPARABLE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import benchmarks as bm
from evaluation import failure_taxonomy as ftx
from evaluation import scoring
from evaluation.action_parser import parse_model_output
from evaluation.backends import (ModelBackendError, ModelTimeout,
                                 OSWorldEnvBackend, ScriptedModelBackend,
                                 SyntheticEnvBackend, make_synthetic_tasks)
from evaluation.checkpoints import discover_checkpoints, require_gates
from evaluation.failure_taxonomy import auto_category
from evaluation.preprocessing import preprocess_screenshot
from evaluation.run_manifest import (build_manifest, require_baseline,
                                     save_manifest, load_manifest)
from evaluation.scaffold import assert_arm_parity, load_scaffold
from evaluation.sota_guard import evaluate_claim
from evaluation.stats import paired_comparison
from evaluation.trajectory_logger import TrajectoryLogger

INTERPOLATION = {"LANCZOS": "LANCZOS"}


class HarnessError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# prompt rendering (must mirror the training format exactly)
# --------------------------------------------------------------------------

def render_user_prompt(instruction: str, history: List[str],
                       history_limit: int) -> str:
    """Training format: step 0 has no Previous-actions block; later steps
    list the model's own emitted actions, oldest dropped beyond the limit."""
    prompt = f"Task: {instruction}"
    if history:
        kept = history[-history_limit:]
        bullets = "\n".join(f"- {a}" for a in kept)
        prompt += f"\nPrevious actions:\n{bullets}"
    return prompt


# --------------------------------------------------------------------------
# task execution
# --------------------------------------------------------------------------

def run_task(task: dict, env, model, scaffold: dict, out_dir: str, arm: str,
             checkpoint: Optional[str], manifest_model_info: dict,
             dry_run: bool) -> scoring.TaskResult:
    exec_cfg = scaffold["execution"]
    history_limit = int(scaffold["history"]["limit"])
    started = time.time()
    env_error: Optional[str] = None
    harness_error: Optional[str] = None
    parse_error_count = 0
    finished = False
    steps_logged = 0

    if hasattr(env, "is_invalid") and env.is_invalid(task):
        return scoring.TaskResult(task_id=task["task_id"],
                                  status=scoring.STATUS_INVALID_TASK,
                                  error_detail="malformed synthetic task")

    logger = TrajectoryLogger(out_dir, task["task_id"], arm, checkpoint)
    logger.log_header(instruction=task["instruction"],
                      system_prompt=scaffold["agent"]["system_prompt"],
                      base_model_revision=manifest_model_info.get("model_revision"),
                      adapter_revision=manifest_model_info.get("adapter_revision"))
    status = None
    detail: Optional[str] = None
    env_success_at_finish: Optional[bool] = None
    env_success_at_budget: Optional[bool] = None
    history: List[str] = []
    observation = None
    try:
        screenshot_dir = os.path.join(out_dir, "screenshots")
        observation = env.reset(task, screenshot_dir)
        for step in range(int(exec_cfg["step_budget"])):
            image, transform = preprocess_screenshot(
                observation.image,
                max_long_side=int(scaffold["observation"]["resize_max_long_side"]))
            messages = [
                {"role": "system", "content": scaffold["agent"]["system_prompt"]},
                {"role": "user", "content": render_user_prompt(
                    task["instruction"], history, history_limit)}]
            raw, retries = _generate_with_retries(
                model, messages, image, scaffold, dry_run)
            parsed = parse_model_output(raw)
            if not parsed.ok:
                parse_error_count += 1
                logger.log_step(step=step, observation_ref=observation.ref_path,
                                observation_transform=transform.as_log_dict(),
                                raw_model_output=raw, parsed_plan=None,
                                parsed_action=None, executed_action=None,
                                parse_error=parsed.error,
                                latency_s=0.0, retries=retries)
                if parse_error_count > int(exec_cfg["retry_count"]):
                    status = scoring.STATUS_PARSER_FAILURE
                    detail = f"repeated parse errors (last: {parsed.error})"
                    break
                continue
            executed = None
            point_to_env = transform.to_env_space
            if parsed.action.verb == "finish":
                finished = True
            outcome = env.step(parsed.action, point_to_env)
            if not outcome.ok:
                env_error = outcome.error or "environment step failed"
                status = scoring.STATUS_ENVIRONMENT_FAILURE
                logger.log_step(step=step, observation_ref=observation.ref_path,
                                observation_transform=transform.as_log_dict(),
                                raw_model_output=raw, parsed_plan=parsed.plan,
                                parsed_action=parsed.action_text,
                                executed_action=executed, parse_error=None,
                                latency_s=0.0, retries=retries,
                                env_state_meta={"env_error": env_error})
                break
            executed = parsed.action_text
            logger.log_step(step=step, observation_ref=observation.ref_path,
                            observation_transform=transform.as_log_dict(),
                            raw_model_output=raw, parsed_plan=parsed.plan,
                            parsed_action=parsed.action_text,
                            executed_action=executed, parse_error=None,
                            latency_s=0.0, retries=retries)
            steps_logged += 1
            history.append(parsed.action_text)
            if finished:
                env_success_at_finish = env.evaluate()
                if env_success_at_finish:
                    status = scoring.STATUS_SUCCESS
                else:
                    status = scoring.STATUS_MODEL_FAILURE
                break
            observation = env.observe(screenshot_dir) if hasattr(env, "observe") else observation
        if status is None:
            status = scoring.STATUS_TIMEOUT
            env_success_at_budget = (env.goal_state_at_budget()
                                     if hasattr(env, "goal_state_at_budget") else None)
            detail = "step budget exhausted without finish"
    except ModelTimeout as e:
        status = scoring.STATUS_TIMEOUT
        detail = f"model timeout: {e}"
    except ModelBackendError as e:
        status = scoring.STATUS_HARNESS_FAILURE
        harness_error = str(e)
    except Exception as e:  # noqa: BLE001 - any unexpected failure is harness-side
        status = scoring.STATUS_HARNESS_FAILURE
        harness_error = f"{type(e).__name__}: {e}"

    success = status == scoring.STATUS_SUCCESS
    category = (auto_category(status, finished, env_success_at_finish,
                              env_success_at_budget)
                if not success else None)
    annotation = "none" if (success or category) else "pending"
    logger.log_summary(status=status, success=success,
                       failure_category=category, total_steps=steps_logged,
                       total_latency_s=time.time() - started)
    logger.close()
    return scoring.TaskResult(
        task_id=task["task_id"], status=status, steps=steps_logged,
        latency_s=round(time.time() - started, 4),
        failure_category=category, failure_annotation=annotation,
        error_detail=detail or harness_error or env_error,
        finished=finished, env_success_at_finish=env_success_at_finish,
        extra={"env_success_at_budget": env_success_at_budget,
               "parse_error_count": parse_error_count})


def _generate_with_retries(model, messages, image, scaffold, dry_run):
    exec_cfg = scaffold["execution"]
    retries = 0
    last_exc: Optional[Exception] = None
    attempts = int(exec_cfg["retry_count"]) + 1
    for attempt in range(attempts):
        try:
            return model.generate(messages, image, scaffold["sampling"],
                                  float(exec_cfg["model_timeout_s"])), retries
        except ModelTimeout:
            raise                      # task-level timeout, not retried here
        except ModelBackendError as e:
            last_exc = e
            retries += 1
            if not dry_run and attempt < attempts - 1:
                time.sleep(float(exec_cfg["retry_backoff_s"]))
    raise last_exc if last_exc else ModelBackendError("unknown model error")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def resolve_task_list(args, benchmark: dict, dry_run: bool,
                      default_seed: int = 1337) -> List[dict]:
    if dry_run:
        seed = args.seed if args.seed is not None else default_seed
        return make_synthetic_tasks(args.tasks or 12, seed=seed)
    source = (args.frozen_subset or args.task_list
              or benchmark.get("task_list_source"))
    if not source or source == bm.REQUIRES_EXTERNAL_VERIFICATION:
        raise HarnessError(
            "no usable task list for a real run; pass --task-list or "
            "--frozen-subset pointing at the pinned task set")
    if source.endswith(".json") and os.path.basename(source).startswith("osworld_verified_frozen"):
        with open(source, "r", encoding="utf-8") as f:
            subset = json.load(f)
        if subset.get("status") != "ready":
            raise HarnessError(
                f"frozen subset {source} is {subset.get('status')!r}; generate "
                "the real subset first (make_frozen_subset.py)")
        ids = set(subset["task_ids"])
        full = bm.load_task_list(args.task_list or benchmark.get("task_list_source"))
        return [t for t in full if t["task_id"] in ids]
    return bm.load_task_list(source)


def resolve_checkpoint(args, manifest_placeholder: dict) -> Optional[dict]:
    if args.checkpoint_gate is None:
        return None
    if args.train_output_dir:
        ckpts = discover_checkpoints(args.train_output_dir,
                                     total_optimizer_steps=args.total_steps)
        gates = require_gates(ckpts, [args.checkpoint_gate])
        ck = gates[args.checkpoint_gate]
        return {"path": ck.path, "global_step": ck.global_step,
                "epoch_fraction_pct": round(ck.epoch_fraction_pct, 2),
                "gate": args.checkpoint_gate}
    if args.adapter_path:
        return {"path": args.adapter_path, "gate": args.checkpoint_gate}
    raise HarnessError("--checkpoint-gate needs --train-output-dir or --adapter-path")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["base", "adapter"], required=True)
    p.add_argument("--benchmark", default="osworld_verified",
                   choices=bm.list_benchmarks())
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--confirm-real-benchmark", action="store_true")
    p.add_argument("--tasks", type=int, default=12,
                   help="dry-run: number of synthetic tasks")
    p.add_argument("--seed", type=int, default=None,
                   help="override task-set seed (dry-run synthetic set)")
    p.add_argument("--task-list")
    p.add_argument("--frozen-subset")
    p.add_argument("--scaffold")
    p.add_argument("--benchmark-config")
    p.add_argument("--model-revision")
    p.add_argument("--adapter-path")
    p.add_argument("--checkpoint-gate", type=int, choices=[20, 55, 100])
    p.add_argument("--train-output-dir")
    p.add_argument("--total-steps", type=int)
    p.add_argument("--base-run-dir",
                   help="base arm run dir (manifest.json + tasks/) for "
                        "comparative scoring")
    p.add_argument("--allow-without-baseline", action="store_true")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--model-backend", choices=["scripted", "openai"],
                   default="scripted")
    p.add_argument("--script-file",
                   help="JSON file with a list of scripted model outputs")
    p.add_argument("--contamination-report",
                   help="contamination check report JSON (see "
                        "evaluation/check_contamination.py)")
    args = p.parse_args(argv)

    scaffold_path = args.scaffold or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "scaffold_config.yaml")
    scaffold = load_scaffold(scaffold_path)

    benchmark_cfg_path = args.benchmark_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "osworld_config.yaml")
    benchmark = bm.get_benchmark(args.benchmark, benchmark_cfg_path)

    if not args.dry_run:
        if not args.confirm_real_benchmark:
            raise SystemExit(
                "real benchmark runs require --confirm-real-benchmark "
                "(and must run via evaluation/run_osworld.py on the "
                "evaluation host)")
        unresolved = bm.unresolved_fields(benchmark)
        if unresolved:
            raise SystemExit(
                f"refusing real run: benchmark protocol fields not pinned: "
                f"{unresolved}")

    # ---- arms and scaffold parity lock -----------------------------------
    import yaml
    with open(benchmark_cfg_path, "r", encoding="utf-8") as f:
        bench_full = yaml.safe_load(f)
    arms = bench_full.get("defaults", {}).get("arms", {})
    assert_arm_parity(scaffold, arms.get("base", {}), arms.get("adapter", {}))

    checkpoint_info = None
    if args.arm == "adapter":
        checkpoint_info = resolve_checkpoint(args, {})

    tasks = resolve_task_list(args, benchmark, args.dry_run,
                              default_seed=int(scaffold["sampling"]["seed"]))
    task_ids = [t["task_id"] for t in tasks]
    task_set_hash = bm.hash_task_ids(task_ids)

    # ---- backends ----------------------------------------------------------
    if args.model_backend == "scripted":
        outputs = None
        if args.script_file:
            with open(args.script_file, "r", encoding="utf-8") as f:
                outputs = json.load(f)
        model = ScriptedModelBackend(outputs=outputs)
    else:
        from evaluation.backends import OpenAICompatibleBackend
        base_url = os.environ.get("JXAGENT_EVAL_BASE_URL")
        if not base_url:
            raise SystemExit("JXAGENT_EVAL_BASE_URL not set for openai backend")
        model = OpenAICompatibleBackend(
            base_url=base_url,
            model=os.environ.get("JXAGENT_EVAL_MODEL", "Qwen3.8-27B"))

    env = SyntheticEnvBackend() if args.dry_run else _real_env(args, benchmark)

    # ---- manifest ----------------------------------------------------------
    manifest = build_manifest(
        arm=args.arm, scaffold_cfg=scaffold, benchmark=benchmark,
        task_ids=task_ids, task_set_hash=task_set_hash,
        model_repo=os.environ.get("JXAGENT_MODEL_REPO", "Qwen/Qwen3.8-27B"),
        model_revision=args.model_revision,
        adapter_revision=(checkpoint_info or {}).get("path") if args.arm == "adapter" else None,
        checkpoint_gate=str(args.checkpoint_gate) if args.checkpoint_gate else None,
        environment_revision=benchmark.get("environment_definition"),
        vm_identifiers={"note": "pinned VM snapshot ids belong to the "
                                "benchmark identity"},
        scaffold_config_path=scaffold_path, dry_run=args.dry_run)
    manifest["model_backend"] = model.describe()
    if checkpoint_info:
        manifest["adapter"]["checkpoint_metadata"] = checkpoint_info
    os.makedirs(args.output_dir, exist_ok=True)
    save_manifest(args.output_dir, manifest)

    # ---- resume ------------------------------------------------------------
    prior = {r.task_id: r for r in scoring.load_task_results(args.output_dir)}
    contamination_blockers = []
    if args.contamination_report:
        with open(args.contamination_report, "r", encoding="utf-8") as f:
            crep = json.load(f)
        if crep.get("status") != "clean":
            contamination_blockers = crep.get("blockers", [{"report": "flagged"}])
    else:
        contamination_blockers = [{
            "reason": "no contamination report supplied for this run"}]

    outcomes: Dict[str, bool] = {}
    for task in tasks:
        tid = task["task_id"]
        prev = prior.get(tid)
        if prev is not None and prev.status in scoring.TERMINAL_STATUSES \
                and not args.force_rerun:
            outcomes[tid] = prev.status == scoring.STATUS_SUCCESS
            continue
        if prev is not None and args.force_rerun:
            old = os.path.join(args.output_dir, "tasks",
                               "".join(c if c.isalnum() or c in "-_." else "_"
                                       for c in tid) + ".json")
            if os.path.exists(old):
                os.replace(old, old + ".old")
        result = run_task(task, env, model, scaffold, args.output_dir,
                          args.arm, manifest["adapter"]["checkpoint_gate"],
                          {"model_revision": args.model_revision,
                           "adapter_revision": manifest["adapter"]["revision"]},
                          args.dry_run)
        scoring.save_task_result(args.output_dir, result)
        outcomes[tid] = result.status == scoring.STATUS_SUCCESS

    env.close()
    results = scoring.load_task_results(args.output_dir)
    agg = scoring.aggregate(results, task_ids,
                            scaffold["failure_accounting"]["protocol_rate_excludes"])
    breakdown = ftx.failure_breakdown([r.to_dict() for r in results])
    agg["failure_breakdown"] = breakdown
    agg["arm"] = args.arm
    agg["benchmark"] = args.benchmark
    scoring.write_aggregate(args.output_dir, agg)

    # ---- baseline-first comparative scoring --------------------------------
    comparison = None
    if args.base_run_dir:
        base_manifest = load_manifest(os.path.join(args.base_run_dir,
                                                   "manifest.json"))
        gate = require_baseline(manifest, base_manifest,
                                allow_without_baseline=args.allow_without_baseline)
        base_results = scoring.load_task_results(args.base_run_dir)
        base_agg_path = os.path.join(args.base_run_dir, "aggregate.json")
        base_agg = None
        if os.path.exists(base_agg_path):
            with open(base_agg_path, "r", encoding="utf-8") as f:
                base_agg = json.load(f)
        base_outcomes = {r.task_id: r.status == scoring.STATUS_SUCCESS
                         for r in base_results}
        claim = evaluate_claim(
            base_manifest=base_manifest, adapter_manifest=manifest,
            base_outcomes=base_outcomes, adapter_outcomes=outcomes,
            base_aggregate=base_agg, adapter_aggregate=agg,
            benchmark=benchmark,
            contamination_blockers=contamination_blockers)
        comparison = {"baseline_gate": gate, "statistics": claim["statistics"],
                      "claim": {k: v for k, v in claim.items()
                                if k != "statistics"},
                      "published_reference_note": (
                          "the quoted 84.3 is a non-authoritative reference "
                          "and is never the comparison anchor")}
        with open(os.path.join(args.output_dir, "comparison.json"), "w",
                  encoding="utf-8") as f:
            json.dump(comparison, f, indent=1, sort_keys=True)

    print(json.dumps({"arm": args.arm, "benchmark": args.benchmark,
                      "dry_run": args.dry_run,
                      "strict_success_rate": agg["strict_success_rate"],
                      "protocol_success_rate": agg["protocol_success_rate"],
                      "status_counts": agg["status_counts"],
                      "accounting_complete": agg["accounting_complete"],
                      "claim_label": (comparison or {}).get("claim", {}).get("label")},
                     indent=1))
    return 0 if agg["accounting_complete"] else 2


def _real_env(args, benchmark: dict):
    return OSWorldEnvBackend(os.environ.get("JXAGENT_OSWORLD_REPO", ""),
                             benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
