#!/usr/bin/env python3
"""Select LoRA target modules for Qwen/Qwen3.8-27B from the REAL architecture.

Selection works on FULL module paths, never bare leaf names:
  - vision tower paths (visual./vision_tower/vision_model/patch_embed/...) excluded
  - aligner paths (merger / multi_modal_projector / mm_projector / aligner) excluded
  - lm_head and embeddings excluded
  - every other Linear in the text decoder stays eligible even when the vision
    tower uses the IDENTICAL leaf name (gate_proj/up_proj/down_proj/qkv/...)

The selected paths are emitted as ONE regex consumed by
`swift sft --target_modules <regex>` (peft applies re.fullmatch against module
paths), so a vision module can never be re-included by leaf-name collision.

Model loading uses the concrete class named in config.architectures when the
installed transformers provides it, otherwise falls back to
AutoModelForImageTextToText -> AutoModelForVision2Seq -> AutoModelForCausalLM
(instantiated from the config on the meta device; no weights are read).

Writes mi300x/lora_modules.json next to this script:
  target_modules: [<single regex string>]
plus counts, leaf-name summaries, the loader class used, and a trainable
parameter estimate at the configured rank.

The classification/selection/regex helpers are torch-free and unit tested
offline (tests/test_training_infrastructure.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Pure helpers (torch-free, unit tested)
# --------------------------------------------------------------------------

# Aligner first: a merger/projector under the vision prefix is the aligner,
# not "vision" (covers merger, vision_merger, multi_modal_projector,
# mm_projector, aligner, projector variants).
ALIGNER_SEGMENTS = {"merger", "aligner", "projector"}
ALIGNER_SUBSTRINGS = ("merger", "projector", "aligner")
VISION_SEGMENTS = {"visual", "vision_tower", "vision_model",
                   "vision_encoder", "vision", "vit"}
VISION_SEGMENT_PREFIXES = ("visual", "vision", "patch_embed")


def classify_path(path: str) -> str:
    """Classify a module path as vision/aligner/lm_head/embedding/text."""
    segs = [s.lower() for s in path.split(".")]
    for s in segs:
        if s in ALIGNER_SEGMENTS or any(k in s for k in ALIGNER_SUBSTRINGS):
            return "aligner"
    for s in segs:
        if s in VISION_SEGMENTS or s.startswith(VISION_SEGMENT_PREFIXES):
            return "vision"
    if segs[-1] == "lm_head":
        return "lm_head"
    for s in segs:
        if s.startswith("embed"):
            return "embedding"
    return "text"


def select_lm_linears(paths: Sequence[str]) -> Dict[str, List[str]]:
    """Split linear module paths into selected / vision / aligner / excluded."""
    out: Dict[str, List[str]] = {"selected": [], "vision": [], "aligner": [],
                                 "excluded": []}
    for p in paths:
        kind = classify_path(p)
        if kind == "text":
            out["selected"].append(p)
        elif kind in ("vision", "aligner"):
            out[kind].append(p)
        else:  # lm_head, embedding
            out["excluded"].append(p)
    return out


def path_to_pattern(path: str) -> str:
    """model.layers.12.mlp.gate_proj -> model\\.layers\\.\\d+\\.mlp\\.gate_proj"""
    parts = [r"\d+" if seg.isdigit() else re.escape(seg) for seg in path.split(".")]
    return r"\.".join(parts)


def build_target_regex(paths: Sequence[str]) -> str:
    """One full-match regex covering exactly the given module paths."""
    patterns = sorted({path_to_pattern(p) for p in paths})
    if not patterns:
        raise ValueError("no LM linear modules selected")
    return "(?:" + "|".join(patterns) + ")"


def leaf_names(paths: Sequence[str]) -> List[str]:
    return sorted({p.split(".")[-1] for p in paths})


# --------------------------------------------------------------------------
# Model loading (torch imported lazily)
# --------------------------------------------------------------------------

def _auto_classes():
    import transformers
    candidates = [getattr(transformers, "AutoModelForImageTextToText", None),
                  transformers.AutoModelForVision2Seq,
                  transformers.AutoModelForCausalLM]
    return [c for c in candidates if c is not None]


def build_model_on_meta(model_dir: str):
    """Instantiate the model from its config on the meta device (no weights).

    Returns (model, config, description_of_how_it_was_loaded).
    """
    import torch
    import transformers
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])

    for name in archs:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            with torch.device("meta"):
                model = cls(cfg)
            return model, cfg, f"transformers.{name} (config.architectures)"
        except Exception as e:  # try the next candidate
            print(f"[inspect] {name} failed on meta device: {e}")

    for auto in _auto_classes():
        try:
            with torch.device("meta"):
                model = auto.from_config(cfg, trust_remote_code=True)
            return model, cfg, f"{auto.__name__}.from_config (fallback)"
        except Exception:
            continue
    raise SystemExit("[inspect] could not instantiate the model from its config "
                     "(tried config.architectures, then AutoModel fallbacks)")


def load_model_with_weights(model_dir: str, dtype_name: str = "bfloat16",
                            device: str = "cpu"):
    """Load real weights with the same class resolution (used by verify_freeze)."""
    import inspect as _inspect
    import torch
    import transformers
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])
    cls = None
    for name in archs:
        c = getattr(transformers, name, None)
        if c is not None:
            cls = c
            break
    if cls is None:
        cls = _auto_classes()[0]

    # transformers renamed torch_dtype -> dtype; use the spelling the installed
    # version actually accepts.
    params = _inspect.signature(cls.from_pretrained).parameters
    dtype_kw = {"dtype" if "dtype" in params else "torch_dtype": getattr(torch, dtype_name)}
    if device in ("cpu",):
        return cls.from_pretrained(model_dir, trust_remote_code=True, **dtype_kw)
    return cls.from_pretrained(model_dir, trust_remote_code=True,
                               device_map=device, **dtype_kw)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(model_dir: str, lora_rank: int) -> int:
    model, cfg, loaded_via = build_model_on_meta(model_dir)

    linear_paths: List[str] = []
    io_features: Dict[str, Tuple[int, int]] = {}
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Linear" not in cls_name:
            continue
        linear_paths.append(name)
        in_f = getattr(module, "in_features", None)
        out_f = getattr(module, "out_features", None)
        if isinstance(in_f, int) and isinstance(out_f, int):
            io_features[name] = (in_f, out_f)

    parts = select_lm_linears(linear_paths)
    selected = parts["selected"]
    vision = parts["vision"]
    aligner = parts["aligner"]
    excluded = parts["excluded"]
    target_regex = build_target_regex(selected)

    # Self-check: the regex must full-match every selected path and must NOT
    # match any vision/aligner/lm_head/embedding path.
    for p in selected:
        assert re.fullmatch(target_regex, p), f"regex misses selected path {p}"
    for p in vision + aligner + excluded:
        assert not re.fullmatch(target_regex, p), f"regex leaks excluded path {p}"

    est_params = sum(lora_rank * (io_features[p][0] + io_features[p][1])
                     for p in selected if p in io_features)

    print(f"model_type    : {cfg.model_type}")
    print(f"architectures : {getattr(cfg, 'architectures', None)}")
    print(f"loaded via    : {loaded_via}")
    print(f"\ntotal linear modules           : {len(linear_paths)}")
    print(f"vision excluded modules        : {len(vision)}  (leaf names: {leaf_names(vision)})")
    print(f"aligner excluded modules       : {len(aligner)}  (leaf names: {leaf_names(aligner)})")
    print(f"lm_head/embedding excluded     : {len(excluded)}  (leaf names: {leaf_names(excluded)})")
    print(f"LM SELECTED modules            : {len(selected)}  (leaf names: {leaf_names(selected)})")
    print(f"target regex                   : {target_regex}")
    if est_params:
        print(f"trainable LoRA params @ r={lora_rank} : {est_params/1e6:.1f}M")
    print("self-check passed: regex covers selected text paths only")

    out = {
        "target_modules": [target_regex],
        "counts": {
            "total_linear": len(linear_paths),
            "vision_excluded": len(vision),
            "aligner_excluded": len(aligner),
            "lm_head_embedding_excluded": len(excluded),
            "lm_selected": len(selected),
        },
        "lm_leaf_names": leaf_names(selected),
        "vision_leaf_names": leaf_names(vision),
        "aligner_leaf_names": leaf_names(aligner),
        "estimated_trainable_params": est_params,
        "lora_rank_used_for_estimate": lora_rank,
        "loaded_via": loaded_via,
        "architectures": getattr(cfg, "architectures", None),
        "model_type": cfg.model_type,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "lora_modules.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--lora_rank", type=int, default=32,
                    help="rank used only for the parameter estimate")
    args = ap.parse_args()
    raise SystemExit(main(args.model_dir, args.lora_rank))
