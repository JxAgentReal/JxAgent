#!/usr/bin/env python3
"""REAL freeze validation for the smoke gate.

snapshot (run BEFORE the smoke training):
  - load the base model, wrap it with the EXACT LoRA config that training
    will use, then record:
      * every trainable parameter name (requires_grad=True after wrapping)
        -> must contain ZERO vision-encoder and ZERO aligner parameters,
           and must cover the expected LM modules (full-path check, not
           substring matching)
      * sha256 digests of a deterministic sample of vision+aligner base
        weight tensors
      * the set of LM base modules that received LoRA adapters
verify (run AFTER the smoke training, on the saved checkpoint):
  - reload the base weights and re-digest the vision+aligner sample:
    digests must be IDENTICAL (base tensors untouched)
  - read adapter_model.safetensors: every LoRA key maps back to a base
    module path that must classify as text (never vision/aligner)
  - the adapter's base-module set must EQUAL the snapshot's trainable set
    (catches peft/ms-swift applying a different target selection than
     the inspected one)
  - at least one lora_B tensor must be non-zero (peft initialises lora_B to
    zero, so any non-zero value proves an optimizer update happened)
  - the checkpoint must contain optimizer.pt + scheduler.pt (faithful
    resume requires them; --save_only_model must stay OFF)

Pure helpers (lora_param_to_base, is_lora_param, sample_names) are torch-free
and unit tested offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inspect_modules as im  # noqa: E402  (same-directory import)

SAMPLE_CAP = 64


# --------------------------------------------------------------------------
# Pure helpers (torch-free, unit tested)
# --------------------------------------------------------------------------

def strip_peft_prefix(name: str) -> str:
    return re.sub(r"^base_model\.model\.", "", name)


def is_lora_param(name: str) -> bool:
    return bool(re.search(r"\.lora_[AB](?:\.default)?\.weight$", strip_peft_prefix(name)))


def lora_param_to_base(name: str) -> str:
    """base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight
       -> model.layers.0.mlp.gate_proj"""
    n = strip_peft_prefix(name)
    n = re.sub(r"\.lora_[AB](?:\.default)?\.weight$", "", n)
    return n


def sample_names(names, cap: int = SAMPLE_CAP) -> list:
    """Deterministic, evenly spread, reproducible subset (first+last kept)."""
    names = sorted(set(names))
    if len(names) <= cap:
        return names
    step = (len(names) - 1) / (cap - 1)
    return [names[round(i * step)] for i in range(cap)]


# --------------------------------------------------------------------------
# Torch-dependent parts (imported lazily; run on the instance)
# --------------------------------------------------------------------------

def tensor_digest(t) -> str:
    import torch
    b = t.detach().contiguous()
    if b.dtype in (torch.bfloat16,):  # numpy has no bf16
        b = b.view(torch.int16)
    return hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest()


def load_and_wrap(model_dir: str, regex: str, rank: int, alpha: int, device: str):
    import peft
    model = im.load_model_with_weights(model_dir, "bfloat16", device)
    lcfg = peft.LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                           target_modules=regex, bias="none")
    return peft.get_peft_model(model, lcfg), model


def cmd_snapshot(args) -> int:
    mods = json.load(open(args.modules))
    regex = mods["target_modules"][0]

    pm, model = load_and_wrap(args.model, regex, args.rank, args.alpha, args.device)
    trainable = sorted(n for n, p in pm.named_parameters() if p.requires_grad)
    if not trainable:
        print("[freeze][FAIL] no trainable parameters at all (target regex matched nothing)")
        return 1
    if not all(is_lora_param(n) for n in trainable):
        print("[freeze][FAIL] non-LoRA parameters are trainable")
        return 1

    bad = [n for n in trainable
           if im.classify_path(lora_param_to_base(n)) in ("vision", "aligner")]
    if bad:
        print(f"[freeze][FAIL] vision/aligner parameters are trainable, e.g. {bad[:5]}")
        return 1

    lm_bases = sorted({lora_param_to_base(n) for n in trainable})
    off_regex = [b for b in lm_bases if not re.fullmatch(regex, b)]
    if off_regex:
        print(f"[freeze][FAIL] trainable modules outside the target regex: {off_regex[:5]}")
        return 1

    base_params = dict(model.named_parameters())
    va_weights = [n for n in base_params
                  if n.endswith(".weight") and im.classify_path(n) in ("vision", "aligner")]
    if not va_weights:
        print("[freeze][FAIL] no vision/aligner weights found in the base model "
              "(wrong loader or unexpected architecture)")
        return 1
    va_sample = sample_names(va_weights)
    va_digests = {n: tensor_digest(base_params[n]) for n in va_sample}

    lora_b = [n for n in trainable if re.search(r"\.lora_B", n)]
    b_sample = sample_names(lora_b)

    snapshot = {
        "target_regex": regex,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_count": len(trainable),
        "lm_bases": lm_bases,
        "vision_aligner_weight_count": len(va_weights),
        "vision_aligner_digests": va_digests,
        "lora_B_sample": b_sample,
    }
    with open(args.out, "w") as f:
        json.dump(snapshot, f, indent=1)
    print(f"[freeze] snapshot ok: {len(trainable)} trainable LoRA params, "
          f"{len(lm_bases)} LM modules wrapped, {len(va_digests)} vision/aligner tensors digested")
    print(f"[freeze] vision/aligner trainable parameters: 0  (verified by full path)")
    return 0


def cmd_verify(args) -> int:
    import torch
    from safetensors import safe_open

    snap = json.load(open(args.snapshot))
    mods = json.load(open(args.modules))
    regex = mods["target_modules"][0]
    ckpt = args.checkpoint

    adapter = os.path.join(ckpt, "adapter_model.safetensors")
    if not os.path.exists(adapter):
        print(f"[freeze][FAIL] missing {adapter}")
        return 1

    # 1) every LoRA tensor in the adapter must belong to a TEXT module
    with safe_open(adapter, framework="pt") as st:
        keys = list(st.keys())
        if not keys:
            print("[freeze][FAIL] adapter file has no tensors")
            return 1
        base_of = {k: lora_param_to_base(k) for k in keys}
        bad = sorted(k for k in keys
                     if im.classify_path(base_of[k]) in ("vision", "aligner"))
        if bad:
            print(f"[freeze][FAIL] adapter contains vision/aligner LoRA tensors: {bad[:5]}")
            return 1

        # 2) adapter coverage must equal the snapshot's trainable coverage
        adapter_bases = sorted(set(base_of.values()))
        if adapter_bases != snap["lm_bases"]:
            missing = sorted(set(snap["lm_bases"]) - set(adapter_bases))[:5]
            extra = sorted(set(adapter_bases) - set(snap["lm_bases"]))[:5]
            print(f"[freeze][FAIL] adapter module set differs from snapshot "
                  f"(missing e.g. {missing}, extra e.g. {extra})")
            return 1

        # 3) at least one lora_B tensor non-zero => optimizer really updated LoRA
        updated = [k for k in keys
                   if re.search(r"\.lora_B", k) and st.get_tensor(k).abs().max().item() > 0]
        if not updated:
            print("[freeze][FAIL] every lora_B tensor is still zero - LoRA was never updated")
            return 1

    # 4) base vision/aligner tensors identical to the pre-training snapshot
    model = im.load_model_with_weights(args.model, "bfloat16", args.device)
    base_params = dict(model.named_parameters())
    changed = [n for n, d in snap["vision_aligner_digests"].items()
               if tensor_digest(base_params[n]) != d]
    if changed:
        print(f"[freeze][FAIL] vision/aligner base tensors changed on disk: {changed[:5]}")
        return 1
    del model, base_params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) checkpoint must carry optimizer/scheduler state for faithful resume
    for fname in ("optimizer.pt", "scheduler.pt"):
        if not os.path.exists(os.path.join(ckpt, fname)):
            print(f"[freeze][FAIL] {fname} missing in checkpoint "
                  "(--save_only_model must stay OFF for faithful resume)")
            return 1

    print(f"[freeze] verify ok:")
    print(f"  adapter LoRA tensors      : {len(keys)} (all text modules, full-path checked)")
    print(f"  vision/aligner LoRA       : 0")
    print(f"  vision/aligner base       : {len(snap['vision_aligner_digests'])} sampled tensors byte-identical")
    print(f"  LM LoRA tensors updated   : {len(updated)} lora_B tensors non-zero")
    print(f"  resume state present      : optimizer.pt + scheduler.pt")
    print("[smoke] PASS: freeze invariants hold, LoRA actually trained, resume state saved")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("snapshot", help="run BEFORE smoke training")
    p.add_argument("--model", required=True)
    p.add_argument("--modules", required=True, help="lora_modules.json")
    p.add_argument("--out", required=True, help="snapshot json path")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--device", default="cpu",
                   help="cpu (default, keeps GPU free) or cuda/auto")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("verify", help="run AFTER smoke training")
    p.add_argument("--model", required=True)
    p.add_argument("--modules", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
