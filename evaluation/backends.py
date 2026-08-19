#!/usr/bin/env python3
"""Model and environment backends for the evaluation harness.

Dry-run (this session's validation mode) uses ScriptedModelBackend +
SyntheticEnvBackend: fully offline, deterministic, exercising the real
parser / preprocessing / coordinate-transform / logging / scoring path.

Real runs use OpenAICompatibleBackend (vLLM/swift serve on MI300X) and an
OSWorld environment backend. Neither is startable from this repo: starting
servers/VMs is explicitly out of scope for the harness code.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from evaluation.action_parser import parse_model_output


# --------------------------------------------------------------------------
# Model backends
# --------------------------------------------------------------------------

class ModelBackendError(RuntimeError):
    pass


class ModelTimeout(ModelBackendError):
    pass


class ModelBackend:
    def generate(self, messages: List[dict], image: Optional[Image.Image],
                 sampling: dict, timeout_s: float) -> str:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class ScriptedModelBackend(ModelBackend):
    """Deterministic scripted outputs (dry-run). Either an explicit output
    list (cycles) or the built-in synthetic-task policy."""

    def __init__(self, outputs: Optional[List[str]] = None,
                 plan_flavor: bool = True):
        self.outputs = list(outputs or [])
        self.plan_flavor = plan_flavor
        self._i = 0

    def generate(self, messages, image, sampling, timeout_s) -> str:
        if self.outputs:
            out = self.outputs[self._i % len(self.outputs)]
            self._i += 1
            return out
        return self._synthetic_policy(messages, image)

    _TARGET_RE = re.compile(
        r"target square at \((\d+),\s*(\d+)\) on a (\d+)x(\d+) screen")

    def _synthetic_policy(self, messages, image) -> str:
        """Emulates a computer-use model for SyntheticEnvBackend tasks: it
        sees the PROCESSED image and must emit coordinates in model space."""
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        m = self._TARGET_RE.search(user)
        if not m:
            return "finish(status=\"success\")"  # degenerate: claims done
        tx, ty, ew, eh = (int(v) for v in m.groups())
        obs_w, obs_h = image.size if image is not None else (ew, eh)
        scale = obs_w / float(ew)
        # deterministic failure flavor from the task text itself
        flavor = random.Random(user).randrange(8)
        mx, my = int(round(tx * scale)), int(round(ty * scale))
        if flavor == 0:      # grounding miss (offset far beyond radius)
            mx += int(120 * scale)
        plan = None
        if self.plan_flavor and flavor != 1:
            plan = "Click the indicated target, then finish."
        if re.search(r"- click\(", _history_summary(user)):
            action = 'finish(status="success")' if flavor != 1 else "wait(seconds=1)"
        else:
            action = f"click(x={mx}, y={my})"
        if plan:
            return f"Plan: {plan}\nAction: {action}"
        return action

    def describe(self) -> dict:
        return {"kind": "scripted", "plan_flavor": self.plan_flavor,
                "n_scripted_outputs": len(self.outputs)}


def _history_summary(user_text: str) -> str:
    idx = user_text.find("Previous actions:")
    return user_text[idx:] if idx >= 0 else ""


class OpenAICompatibleBackend(ModelBackend):
    """Chat-completions client for the later MI300X serving stack.

    All sampling parameters are passed explicitly from the scaffold config;
    no library/server defaults are relied upon. Nothing in this repo starts
    the server.
    """

    def __init__(self, base_url: str, model: str,
                 api_key_env: str = "JXAGENT_EVAL_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env

    def generate(self, messages, image, sampling, timeout_s) -> str:
        import requests
        api_key = os.environ.get(self.api_key_env, "EMPTY")
        content = []
        if image is not None:
            import io
            import base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
        for m in messages:
            if m.get("role") == "user":
                item = {"type": "text", "text": m.get("content", "")}
                content.append(item)
        body = {
            "model": self.model,
            "messages": [{"role": m["role"], "content": m.get("content")}
                         if m.get("role") != "user" else
                         {"role": "user", "content": content or m.get("content")}],
            "temperature": sampling.get("temperature", 0.0),
            "top_p": sampling.get("top_p", 1.0),
            "max_tokens": sampling.get("max_new_tokens", 512),
            "seed": sampling.get("seed"),
        }
        try:
            resp = requests.post(f"{self.base_url}/v1/chat/completions",
                                 json=body, timeout=timeout_s,
                                 headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.Timeout as e:
            raise ModelTimeout(str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise ModelBackendError(str(e)) from e

    def describe(self) -> dict:
        return {"kind": "openai_compatible", "base_url": self.base_url,
                "model": self.model}


# --------------------------------------------------------------------------
# Environment backends
# --------------------------------------------------------------------------

@dataclass
class Observation:
    image: Image.Image
    env_size: Tuple[int, int]           # environment pixel size (w, h)
    ref_path: Optional[str] = None      # persisted screenshot path


@dataclass
class StepOutcome:
    ok: bool
    error: Optional[str] = None
    env_meta: Optional[dict] = None


class EnvironmentBackend:
    def reset(self, task: dict, screenshot_dir: str) -> Observation:
        raise NotImplementedError

    def step(self, action, point_to_env) -> StepOutcome:
        raise NotImplementedError

    def observe(self, screenshot_dir: str) -> Observation:
        """Post-step observation (new screenshot)."""
        raise NotImplementedError

    def evaluate(self) -> Optional[bool]:
        """True/False when the scorer can judge; None when unavailable."""
        raise NotImplementedError

    def goal_state_at_budget(self) -> Optional[bool]:
        """Scorer value at budget exhaustion (for failure_to_finish)."""
        return None

    def close(self) -> None:
        pass


class SyntheticEnvBackend(EnvironmentBackend):
    """Offline deterministic environment for dry runs.

    Task format (instruction): "... target square at (x, y) on a WxH screen
    ..." plus an optional flavor tag:
      [invalid]  - malformed task -> invalid_task
    Success = target square clicked (within radius) AND finish emitted.
    """

    RADIUS = 40

    def __init__(self, env_size: Tuple[int, int] = (1920, 1080)):
        self.env_size = env_size
        self._clicked = False
        self._finished = False
        self._step = 0

    def reset(self, task: dict, screenshot_dir: str) -> Observation:
        self._clicked = False
        self._finished = False
        self._step = 0
        self._task = task
        w, h = self.env_size
        img = Image.new("RGB", (w, h), (24, 26, 32))
        draw = ImageDraw.Draw(img)
        m = re.search(r"target square at \((\d+),\s*(\d+)\)", task.get("instruction", ""))
        if m:
            tx, ty = int(m.group(1)), int(m.group(2))
            draw.rectangle([tx - 30, ty - 30, tx + 30, ty + 30], fill=(220, 120, 40))
        os.makedirs(screenshot_dir, exist_ok=True)
        ref = os.path.join(screenshot_dir,
                           f"{_safe(task['task_id'])}_s{self._step}.png")
        img.save(ref)
        return Observation(image=img, env_size=self.env_size, ref_path=ref)

    def observe(self, screenshot_dir: str) -> Observation:
        self._step += 1
        w, h = self.env_size
        img = Image.new("RGB", (w, h), (24, 26, 32))
        draw = ImageDraw.Draw(img)
        m = re.search(r"target square at \((\d+),\s*(\d+)\)",
                      self._task.get("instruction", ""))
        if m:
            tx, ty = int(m.group(1)), int(m.group(2))
            draw.rectangle([tx - 30, ty - 30, tx + 30, ty + 30], fill=(220, 120, 40))
        os.makedirs(screenshot_dir, exist_ok=True)
        ref = os.path.join(screenshot_dir,
                           f"{_safe(self._task['task_id'])}_s{self._step}.png")
        img.save(ref)
        return Observation(image=img, env_size=self.env_size, ref_path=ref)

    def step(self, action, point_to_env) -> StepOutcome:
        self._step += 1
        verb = action.verb if action is not None else ""
        if verb == "finish":
            self._finished = True
            return StepOutcome(True, env_meta={"finished": True})
        if verb in ("click", "double_click", "right_click", "middle_click"):
            if not action.points:
                return StepOutcome(False, "click without coordinates")
            ex, ey = point_to_env(*action.points[0])
            m = re.search(r"target square at \((\d+),\s*(\d+)\)",
                          self._task.get("instruction", ""))
            if m:
                tx, ty = int(m.group(1)), int(m.group(2))
                if (ex - tx) ** 2 + (ey - ty) ** 2 <= self.RADIUS ** 2:
                    self._clicked = True
            return StepOutcome(True, env_meta={"executed_point": [ex, ey]})
        # any other verb: accepted, no state change (enough for dry-run)
        return StepOutcome(True)

    def evaluate(self) -> Optional[bool]:
        return self._clicked and self._finished

    def goal_state_at_budget(self) -> Optional[bool]:
        return self._clicked

    def is_invalid(self, task: dict) -> bool:
        return "[invalid]" in task.get("instruction", "") or \
            not re.search(r"target square at \(\d+,\s*\d+\)", task.get("instruction", ""))


def make_synthetic_tasks(n: int, seed: int = 1337) -> List[dict]:
    """Deterministic synthetic task set with mixed outcomes."""
    rng = random.Random(seed)
    tasks: List[dict] = []
    domains = ["chrome", "os", "libreoffice_calc", "multi_apps"]
    for i in range(n):
        domain = domains[i % len(domains)]
        task_id = f"{domain}/synthetic-{i:04d}"
        if i == n - 1 and n > 2:   # one malformed task for invalid_task path
            tasks.append({"task_id": task_id, "domain": domain,
                          "instruction": "Broken task with no coordinates [invalid]"})
            continue
        tx = rng.randrange(200, 1720)
        ty = rng.randrange(150, 930)
        tasks.append({
            "task_id": task_id, "domain": domain,
            "instruction": f"Click the target square at ({tx}, {ty}) on a "
                           f"1920x1080 screen, then finish."})
    return tasks


def _safe(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in task_id)


class OSWorldEnvBackend(EnvironmentBackend):
    """Integration point for real OSWorld runs.

    Requires the OSWorld repository checked out at the PINNED revision
    recorded in the benchmark identity, its Python environment available,
    and VM snapshots provisioned. This class intentionally fails loudly
    otherwise - the harness must never silently fake a real benchmark.
    """

    def __init__(self, osworld_repo: str, benchmark_cfg: dict):
        missing = [k for k, v in benchmark_cfg.items()
                   if isinstance(v, str) and v == "REQUIRES_EXTERNAL_VERIFICATION"]
        if missing:
            raise RuntimeError(
                f"OSWorld backend refused: benchmark protocol fields not "
                f"pinned: {missing}. Resolve them (exact repo revision, task "
                "list, environment definition) before any real run.")
        try:
            import desktop_env  # noqa: F401  (OSWorld package)
        except ImportError as e:
            raise RuntimeError(
                "OSWorld package (desktop_env) not installed. Clone the "
                "OSWorld repository at the pinned revision and install its "
                "requirements; evaluation must run inside its environment."
            ) from e
        self.repo = osworld_repo
        raise NotImplementedError(
            "OSWorld runner glue is implemented at evaluation time on the "
            "MI300X host; keep this the ONLY place environment code enters "
            "the harness.")
