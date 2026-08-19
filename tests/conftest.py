"""Shared synthetic fixtures: no network, no real datasets, no large files."""
from __future__ import annotations

import io
import os
import sys

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.dedup import DedupIndex
from processing.reasoning import ReasoningGate
from processing.state import BuildState
from sources.common import BuildContext


def make_png_bytes(w: int = 640, h: int = 480, marker: int = 0, blocks: int = 8) -> bytes:
    """Deterministic, visually distinct screenshot fixture.

    Distinct markers place a large high-contrast block at very different
    positions/colors so pHash separates them reliably."""
    img = Image.new("RGB", (w, h), (18 + marker % 40, 24, 30))
    d = ImageDraw.Draw(img)
    bx = (marker * 97) % max(1, w - 120)
    by = (marker * 53) % max(1, h - 120)
    d.rectangle([bx, by, bx + 100, by + 80],
                fill=(30 + (marker * 37) % 220, (marker * 71) % 220, (marker * 13) % 220))
    for i in range(blocks):
        x0 = (i * w // blocks + marker * 7) % (w - 20)
        y0 = (i * h // blocks + marker * 13) % (h - 20)
        d.rectangle([x0, y0, x0 + 12, y0 + 8], fill=(200 - marker % 100, 90, 40 + i * 5))
        d.text((4, i * 12), f"fixture {marker}", fill=(250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def png_factory():
    return make_png_bytes


def make_ctx(tmp_path, quota=None, config=None) -> BuildContext:
    cfg = {
        "context_budget": 8192,
        "per_trajectory_cap": 4,
    }
    if config:
        cfg.update(config)
    state = BuildState(str(tmp_path / "state"))
    ctx = BuildContext(
        dataset_root=str(tmp_path), state=state, config=cfg,
        dedup=DedupIndex(), reasoning_gate=ReasoningGate(),
        offline=True, smoke=False, quota=quota or {},
    )
    return ctx


@pytest.fixture
def ctx_factory(tmp_path):
    def _factory(quota=None, config=None):
        return make_ctx(tmp_path, quota, config)
    return _factory


def make_step(idx: int, size=(1920, 1080), marker=None, action=None, signals=None,
              prev_phash=None, task_prefix="t"):
    from processing.coordinates import Action, CoordSpace, Point
    from processing.dedup import phash
    from processing.windows import Step
    marker = idx if marker is None else marker
    data = make_png_bytes(size[0], size[1], marker=marker)
    if action is None:
        action = Action("click", points=[(100 + idx * 10, 200 + idx * 5)],
                        original="pyautogui.click(x=..., y=...)",
                        original_space=CoordSpace.PIXEL)
    from processing.images import load_image
    return Step(
        step_id=f"{task_prefix}_s{idx}", image_bytes=data, image_size=size,
        action=action, phash=phash(load_image(data)), prev_phash=prev_phash,
        signals=set(signals or []), metadata={},
    )


def make_trajectory(n=10, task="Sort the files by size and export the report as PDF.",
                    source="procua", tid="traj_1", with_recovery=False, size=(1920, 1080)):
    from processing.windows import Trajectory
    steps = []
    prev = None
    for i in range(n):
        marker = i
        prev_phash = None
        if with_recovery and i == 4:
            marker = 3  # near-identical screenshot to step 3 -> no state change
            prev_phash = prev
        s = make_step(i, size=size, marker=marker, prev_phash=prev_phash,
                      task_prefix=tid)
        if with_recovery and i == 4:
            s.signals.update({"no_state_change", "recovery_evidenced"})
        steps.append(s)
        prev = s.phash
    return Trajectory(trajectory_id=tid, task=task, steps=steps, app="excel",
                      source=source, metadata={})


@pytest.fixture
def traj_factory():
    return make_trajectory


# ---- shared synthetic source-row builders -------------------------------

def make_events(n=3, size=(1920, 1080)):
    """PC-Agent-E style events + screenshots mirroring the verified zip layout."""
    import json
    shots = {}
    lines = []
    for i in range(n):
        name = f"abc_{i:04d}_{i + 1}.png"
        shots[name] = make_png_bytes(*size, marker=i)
        lines.append(json.dumps({
            "action": f"click ({100 + i * 50}, {200 + i * 30})",
            "screenshot": f"screenshot/{name}",
            "element": f"Menu item {i}",
            "rect": {"left": 90 + i * 50, "top": 190 + i * 30,
                     "right": 130 + i * 50, "bottom": 210 + i * 30},
            "marked_screenshot": f"screenshot/{name[:-4]}_marked.png",
            "thought": "Verbose human walkthrough text that must never be trained.",
        }))
    return "\n".join(lines), shots


def gui360_use_row(n_steps=3, row_id="excel_1_105", task="Save the file as read-only."):
    """GUI-360 desktop.use style row (messages/metadata as JSON strings)."""
    import json
    images = [{"bytes": make_png_bytes(1040, 736, marker=i), "path": f"{i}.jpg"}
              for i in range(n_steps)]
    messages = []
    for i in range(n_steps):
        content = [{"type": "image", "index": i}]
        if i == 0:
            content.append({"type": "text", "text": task})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant",
                         "content": [{"type": "inline_reasoning", "text": "verbose..."},
                                     {"type": "action_description", "text": "verbose..."}],
                         "tool_calls": [{"type": "function", "function": {
                             "name": "click",
                             "arguments": {"coordinate": [40 + i * 100, 70 + i * 50]}}} ]})
    return {
        "images": images,
        "messages": json.dumps(messages),
        "metadata": json.dumps({"platform": "desktop", "task_type": "use",
                                "others": {"id": row_id, "resolution": [1040, 736],
                                           "os": "windows", "source": "vyokky/GUI-360"}}),
    }
