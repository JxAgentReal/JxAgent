"""Contract-driven rendering for JxAgent native Computer Use supervision.

The production builder only receives a contract after the Qwen interface
manifest has been verified. Development/smoke builds may omit it and keep the
legacy JxAgent text action representation. No Qwen-specific values live here.
"""
from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from .coordinates import Action, parse_rendered_action, rebase_action_points

MODEL_ID = os.environ.get("JXAGENT_MODEL_ID", "Qwen/Qwen3.8-27B")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_verified_manifest(path: str) -> Tuple[dict, dict]:
    """Read a frozen interface manifest for dataset generation.

    Dataset generation may happen on a different machine from the model files,
    so model-file drift is rechecked by tools/verify_interface_manifest.py on
    the training host. Here we require the signed-off embedded contract and
    hash the manifest into the build identity.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"interface manifest missing: {p}")
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("status") != "verified":
        raise ValueError("interface manifest is not verified: " + ",".join(m.get("unresolved") or []))
    if m.get("model_id") != MODEL_ID:
        raise ValueError(f"unexpected interface model_id: {m.get('model_id')}")
    if int(m.get("schema_version") or 0) < 3:
        raise ValueError("interface manifest schema too old")
    native = m.get("native_cua") or {}
    contract = native.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("verified manifest has no embedded native contract")
    if contract.get("model_id") != MODEL_ID:
        raise ValueError("embedded native contract model mismatch")
    adapter = contract.get("adapter") or {}
    if adapter.get("family") != "jxagent_text_action_v1":
        raise ValueError(f"unsupported native adapter family: {adapter.get('family')}")
    coord = contract.get("coordinate_space") or {}
    ctype = coord.get("type") if isinstance(coord, dict) else coord
    if ctype not in {"processed_image_pixels", "normalized_0_1000"}:
        raise ValueError(f"unsupported coordinate space: {ctype}")
    history = contract.get("history_policy") or {}
    if history.get("mode") not in {"text_actions", "visual_recent_rounds"}:
        raise ValueError(f"unsupported history mode: {history.get('mode')}")
    return contract, {"path": str(p), "sha256": file_sha256(p),
                      "contract_sha256": native.get("contract_sha256")}


def clone_action(action: Action) -> Action:
    return Action(verb=action.verb, args=dict(action.args), points=list(action.points),
                  original=action.original, original_space=action.original_space,
                  original_points=list(action.original_points))


def action_in_final_pixels(action: Action, original_size: Sequence[int],
                           final_size: Sequence[int]) -> Action:
    out = clone_action(action)
    if tuple(map(int, original_size)) != tuple(map(int, final_size)):
        out = rebase_action_points(out, original_size, final_size)
    return out


def _to_norm_1000(action: Action, final_size: Sequence[int]) -> Action:
    out = clone_action(action)
    w, h = int(final_size[0]), int(final_size[1])
    pts = []
    for x, y in out.points:
        nx = 0 if w <= 1 else int(round(float(x) * 1000.0 / (w - 1)))
        ny = 0 if h <= 1 else int(round(float(y) * 1000.0 / (h - 1)))
        pts.append((max(0, min(1000, nx)), max(0, min(1000, ny))))
    out.points = pts
    return out


def render_action_for_contract(action: Action, original_size: Sequence[int],
                               final_size: Sequence[int], contract: Optional[dict]) -> Tuple[str, str]:
    """Return (serialized native target, canonical final-pixel action)."""
    pixel_action = action_in_final_pixels(action, original_size, final_size)
    canonical = pixel_action.render()
    if not contract:
        return canonical, canonical
    coord = contract.get("coordinate_space") or {}
    coord_type = coord.get("type") if isinstance(coord, dict) else coord
    interface_action = pixel_action
    if coord_type == "normalized_0_1000":
        interface_action = _to_norm_1000(pixel_action, final_size)
    elif coord_type != "processed_image_pixels":
        raise ValueError(f"unsupported coordinate space {coord_type!r}")
    action_text = interface_action.render()
    layout = contract.get("message_layout") or {}
    template = layout.get("assistant_action_template") or "{action}"
    return template.format(action=action_text), canonical


def coordinate_free_action(text: str) -> str:
    """Remove stale screen coordinates while retaining non-spatial intent."""
    act = parse_rendered_action(text.strip())
    if act is None:
        # Never regex-copy unknown coordinate syntax into old history.
        return "previous_action"
    if act.verb in {"click", "double_click", "right_click", "middle_click", "move", "point", "drag"}:
        return act.verb
    if act.verb == "scroll":
        return f"scroll(clicks={int(act.args.get('clicks', 0) or 0)})"
    # typing/hotkeys/waits/finish contain no stale pixel coordinates in JxAgent.
    return act.render()


def interface_layout(contract: Optional[dict]) -> Dict[str, str]:
    if not contract:
        return {"system_prompt": "You are a computer-use agent.",
                "image_placeholder": "<image>",
                "task_template": "Task: {task}",
                "history_heading": "Previous actions:"}
    layout = contract.get("message_layout") or {}
    return {"system_prompt": str(layout["system_prompt"]),
            "image_placeholder": str(layout["image_placeholder"]),
            # Native rendering is driven by the exact text/visual user templates.
            # These compatibility fields must never make an otherwise verified
            # contract fail merely because an unused legacy label is absent.
            "task_template": str(layout.get("task_template") or "{task}"),
            "history_heading": str(layout.get("history_heading") or "")}
