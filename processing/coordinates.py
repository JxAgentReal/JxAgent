"""Coordinate systems, conversion and validation.

Verified source conventions (2026-08, inspected from public repos):
  - ProCUA-SFT     : absolute pixels in the screenshot's native resolution
                     (pyautogui commands, e.g. pyautogui.click(x=512, y=384)).
  - cua-lite/GUI-360: integers normalized to [0, 1000] in tool_calls
                     ("coordinate": [38, 96], "point"); reference resolution is
                     metadata.others.resolution = [width, height].
  - ServiceNow/VideoCUA: absolute pixels in the task's source coordinate
                     frame.  The frame is accepted only when explicit source
                     metadata proves it, or every raw point fits the decoded
                     frame; JxAgent never silently assumes 1920x1080.
  - ServiceNow/GroundCUA: float pixel bounding boxes [x1, y1, x2, y2].
  - PC-Agent-E     : absolute pixels inside action strings, "click (654, 191)".

The unified training action is rendered in FINAL image pixel space. Original
coordinates and their space are always kept in metadata; conversion is
explicit and round-tripped by tests.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class CoordSpace(str, Enum):
    PIXEL = "pixel"          # absolute pixels, requires (width, height)
    NORM_0_1 = "norm_0_1"    # normalized floats 0..1
    NORM_0_1000 = "norm_0_1000"  # normalized integers 0..1000 (GUI-360)


@dataclass
class Point:
    x: float
    y: float
    space: CoordSpace = CoordSpace.PIXEL

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        if width <= 0 or height <= 0:
            raise CoordinateError(f"invalid image size {(width, height)}")
        if self.space == CoordSpace.PIXEL:
            px, py = self.x, self.y
        elif self.space == CoordSpace.NORM_0_1:
            # Normalized coordinates describe the closed image domain.  Map
            # endpoints to the last valid pixel rather than width/height.
            px, py = self.x * max(0, width - 1), self.y * max(0, height - 1)
        elif self.space == CoordSpace.NORM_0_1000:
            px = self.x / 1000.0 * max(0, width - 1)
            py = self.y / 1000.0 * max(0, height - 1)
        else:  # pragma: no cover
            raise ValueError(f"unknown space {self.space}")
        return int(round(px)), int(round(py))


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    space: CoordSpace = CoordSpace.PIXEL

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        p1 = Point(self.x1, self.y1, self.space).to_pixels(width, height)
        p2 = Point(self.x2, self.y2, self.space).to_pixels(width, height)
        return p1[0], p1[1], p2[0], p2[1]

    def center_pixels(self, width: int, height: int) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.to_pixels(width, height)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def width_height(self, width: int, height: int) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.to_pixels(width, height)
        return max(1, x2 - x1), max(1, y2 - y1)


class CoordinateError(ValueError):
    """Raised when a coordinate cannot be represented within image bounds."""


def scale_point(x: float, y: float, old_size: Sequence[int], new_size: Sequence[int]) -> Tuple[int, int]:
    """Scale a pixel point when an image is resized (spec section 17)."""
    ow, oh = int(old_size[0]), int(old_size[1])
    nw, nh = int(new_size[0]), int(new_size[1])
    if ow <= 0 or oh <= 0 or nw <= 0 or nh <= 0:
        raise CoordinateError(f"invalid sizes {old_size}->{new_size}")
    # Preserve pixel-domain endpoints exactly: 0 -> 0 and old-1 -> new-1.
    # The old x*nw/ow formula could shift edge targets and made independent
    # coordinate round-trips disagree with normalized endpoint semantics.
    nx = 0.0 if ow == 1 else x * (nw - 1) / (ow - 1)
    ny = 0.0 if oh == 1 else y * (nh - 1) / (oh - 1)
    return int(round(nx)), int(round(ny))


def scale_bbox(bbox: Sequence[float], old_size: Sequence[int], new_size: Sequence[int]) -> Tuple[int, int, int, int]:
    x1, y1 = scale_point(bbox[0], bbox[1], old_size, new_size)
    x2, y2 = scale_point(bbox[2], bbox[3], old_size, new_size)
    # guard against equal edges after rounding on tiny targets
    if x2 <= x1:
        x2 = min(x1 + 1, new_size[0] - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, new_size[1] - 1)
    return x1, y1, x2, y2


def validate_point(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def validate_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> bool:
    return (
        0 <= x1 < x2 < width and 0 <= y1 < y2 < height
    )


# --------------------------------------------------------------------------
# Unified internal action representation
# --------------------------------------------------------------------------

SUPPORTED_VERBS = {
    "click", "double_click", "right_click", "middle_click", "move", "drag",
    "scroll", "type", "press", "hotkey", "key_down", "key_up",
    "mouse_down", "mouse_up", "wait", "finish", "point",
}
# verbs whose arguments include pixel points in the FINAL image space
POINT_VERBS = {"click", "double_click", "right_click", "middle_click", "move", "point"}
DRAG_VERBS = {"drag"}


@dataclass
class Action:
    verb: str
    args: Dict[str, Any] = field(default_factory=dict)
    # points in FINAL image pixel space; drag keeps start+end
    points: List[Tuple[int, int]] = field(default_factory=list)
    # original representation kept verbatim for metadata / debugging
    original: str = ""
    original_space: Optional[CoordSpace] = None
    original_points: List[Tuple[float, float]] = field(default_factory=list)

    def render(self) -> str:
        a = self.args
        if self.verb in POINT_VERBS and self.points:
            x, y = self.points[-1]
            extra = ""
            if self.verb == "point":
                return f"point(x={x}, y={y})"
            if a.get("button") in ("right",):
                name = "right_click"
            elif self.verb == "double_click" or (self.verb == "click" and a.get("count") == 2):
                name = "double_click"
            elif a.get("button") == "middle":
                name = "middle_click"
            else:
                name = self.verb if self.verb != "click" or "button" not in a or a.get("button") == "left" else "click"
                if self.verb == "click" and a.get("button") == "middle":
                    name = "middle_click"
                if self.verb == "double_click":
                    name = "double_click"
            # normalize click variants
            if self.verb == "click":
                if a.get("count") == 2:
                    name = "double_click"
                elif a.get("button") == "right":
                    name = "right_click"
                elif a.get("button") == "middle":
                    name = "middle_click"
                else:
                    name = "click"
            return f"{name}(x={x}, y={y}){extra}"
        if self.verb == "drag" and len(self.points) >= 2:
            (x1, y1), (x2, y2) = self.points[0], self.points[1]
            btn = a.get("button", "left")
            return f"drag(x1={x1}, y1={y1}, x2={x2}, y2={y2}, button=\"{btn}\")"
        if self.verb == "scroll":
            clicks = int(a.get("clicks", a.get("amount", 0)) or 0)
            if self.points:
                x, y = self.points[-1]
                return f"scroll(clicks={clicks}, x={x}, y={y})"
            return f"scroll(clicks={clicks})"
        if self.verb == "type":
            return f"type(text={json.dumps(a.get('text', ''), ensure_ascii=False)})"
        if self.verb == "press":
            return f"press(key={json.dumps(str(a.get('key', '')), ensure_ascii=False)})"
        if self.verb == "hotkey":
            keys = a.get("keys") or []
            rendered = ", ".join(json.dumps(str(k), ensure_ascii=False) for k in keys)
            return f"hotkey({rendered})"
        if self.verb in ("key_down", "key_up", "mouse_down", "mouse_up"):
            key = a.get("key") or a.get("button", "left")
            return f"{self.verb}({json.dumps(str(key), ensure_ascii=False)})"
        if self.verb == "wait":
            secs = a.get("seconds", a.get("seconds"))
            return f"wait(seconds={float(secs)})" if secs is not None else "wait()"
        if self.verb == "finish":
            status = a.get("status", "success")
            return f"finish(status=\"{status}\")"
        # fallback: verb with json args (should not happen for supported verbs)
        return f"{self.verb}({json.dumps(a, ensure_ascii=False)})"


# regex for the FINAL rendered action strings (used by validation + evaluation)
_RENDER_RE = re.compile(
    r"^(?P<verb>click|double_click|right_click|middle_click|move|point|drag|scroll|type|press|hotkey|"
    r"key_down|key_up|mouse_down|mouse_up|wait|finish)\((?P<args>.*)\)$",
    re.DOTALL,
)


def parse_rendered_action(text: str) -> Optional[Action]:
    """Parse a rendered action string back into an Action (for validation/tests)."""
    text = text.strip()
    m = _RENDER_RE.match(text)
    if not m:
        return None
    verb, args_str = m.group("verb"), m.group("args").strip()
    act = Action(verb=verb, original=text)
    # split top-level args respecting quotes/brackets and escaped quotes
    parts: List[str] = []
    depth = 0
    quote = ""
    cur = ""
    for ch in args_str:
        if quote:
            cur += ch
            if ch == quote and not cur.endswith("\\" + ch):
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())

    def coerce(tok: str):
        tok = tok.strip()
        try:
            return int(tok)
        except ValueError:
            pass
        try:
            return float(tok)
        except ValueError:
            pass
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            inner = tok[1:-1]
            try:
                import json as _json
                return _json.loads(tok)
            except Exception:
                return inner
        return tok

    kv: Dict[str, Any] = {}
    for p in parts:
        if "=" in p and not p.startswith(("\"", "'")):
            k, v = p.split("=", 1)
            kv[k.strip()] = coerce(v)
    try:
        if verb in POINT_VERBS:
            x = kv.get("x")
            y = kv.get("y")
            if x is None or y is None:
                return None
            act.points = [(int(x), int(y))]
            act.args = kv
            return act
        if verb == "drag":
            need = ("x1", "y1", "x2", "y2")
            if any(kv.get(k) is None for k in need):
                return None
            act.points = [(int(kv["x1"]), int(kv["y1"])), (int(kv["x2"]), int(kv["y2"]))]
            act.args = kv
            return act
        if verb == "scroll":
            if "clicks" not in kv:
                return None
            act.args = kv
            if kv.get("x") is not None and kv.get("y") is not None:
                act.points = [(int(kv["x"]), int(kv["y"]))]
            return act
        if verb == "type":
            if "text" not in kv:
                return None
            act.args = {"text": str(kv["text"])}
            return act
        if verb == "press":
            if "key" not in kv:
                return None
            act.args = {"key": str(kv["key"])}
            return act
        if verb == "hotkey":
            keys = [coerce(p) for p in parts if "=" not in p or p.startswith(("\"", "'"))]
            act.args = {"keys": [str(k) for k in keys]}
            return act
        act.args = kv
        return act
    except (TypeError, ValueError):
        return None


def action_in_bounds(action: Action, width: int, height: int) -> bool:
    for (x, y) in action.points:
        if not validate_point(x, y, width, height):
            return False
    return True


# --------------------------------------------------------------------------
# Source action parsers
# --------------------------------------------------------------------------

_CALL_RE = re.compile(
    r"^\s*(?:pyautogui\.)?(?P<fn>[a-zA-Z_][a-zA-Z0-9_]*)\s*\((?P<args>.*)\)\s*;?\s*$",
    re.DOTALL,
)


def _split_args(args_str: str) -> List[str]:
    parts, depth, quote, cur = [], 0, "", ""
    for ch in args_str:
        if quote:
            cur += ch
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            cur += ch
        elif ch in "([{":
            depth += 1
            cur += ch
        elif ch in ")]}":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _coerce(tok: str) -> Any:
    tok = tok.strip()
    if not tok or tok.lower() == "none":
        return None
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        return tok[1:-1]
    return tok


def _literal(node: ast.AST) -> Any:
    """Safely evaluate a pyautogui argument.

    ProCUA stores Python source, not a bespoke action DSL.  Using ``ast`` is
    both more accurate (triple-quoted strings, escaped text, lists) and safer
    than regex/string splitting.  Anything that is not a literal is rejected.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        raise CoordinateError("non-literal pyautogui argument")


def _pyautogui_call(node: ast.AST) -> Optional[Tuple[str, List[Any], Dict[str, Any]]]:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    fn = call.func
    if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
            and fn.value.id == "pyautogui"):
        return None
    try:
        args = [_literal(a) for a in call.args]
        kwargs = {kw.arg: _literal(kw.value) for kw in call.keywords if kw.arg}
    except CoordinateError:
        return None
    return fn.attr.lower(), args, kwargs


def parse_pyautogui(command: str, width: int, height: int,
                    space: CoordSpace = CoordSpace.PIXEL) -> Optional[Action]:
    """Parse a *safe, deterministic* subset of ProCUA pyautogui source.

    The previous hand-written parser used only the first physical line and
    broke Python triple-quoted strings (for example a triple-quoted Product A literal).  This
    implementation parses the complete Python snippet with ``ast`` and only
    accepts literal pyautogui calls.  Ambiguous multi-statement snippets are
    rejected rather than teaching an action against the wrong visual state.

    Two multi-statement idioms can be collapsed without ambiguity:
      * ``keyDown(k); keyUp(k)`` -> ``press(k)``
      * ``moveTo(x1,y1); dragTo(x2,y2,...)`` -> one explicit drag
    """
    if not command or width <= 0 or height <= 0:
        return None
    try:
        tree = ast.parse(command.strip(), mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    calls = [_pyautogui_call(stmt) for stmt in tree.body]
    if not calls or any(c is None for c in calls):
        return None

    # Deterministic collapse of a key press encoded as down/up.
    if len(calls) == 2 and calls[0][0] == "keydown" and calls[1][0] == "keyup":
        a0, a1 = calls[0], calls[1]
        k0 = a0[2].get("key", a0[1][0] if a0[1] else None)
        k1 = a1[2].get("key", a1[1][0] if a1[1] else None)
        if k0 is not None and str(k0) == str(k1):
            return Action("press", args={"key": str(k0)}, original=command,
                          original_space=space)
        return None

    # A drag endpoint is only useful if the visual-state cursor start is known.
    if len(calls) == 2 and calls[0][0] in ("moveto", "move") and calls[1][0] in ("dragto", "drag"):
        f0, a0, k0 = calls[0]
        f1, a1, k1 = calls[1]
        def _xy(args, kw):
            x = kw.get("x", kw.get("xcoordinate", args[0] if len(args) > 0 else None))
            y = kw.get("y", kw.get("ycoordinate", args[1] if len(args) > 1 else None))
            return None if x is None or y is None else (float(x), float(y))
        p0, p1 = _xy(a0, k0), _xy(a1, k1)
        if p0 is None or p1 is None:
            return None
        pp0 = Point(*p0, space).to_pixels(width, height)
        pp1 = Point(*p1, space).to_pixels(width, height)
        if not (validate_point(*pp0, width, height) and validate_point(*pp1, width, height)):
            return None
        return Action("drag", args={"button": str(k1.get("button", "left") or "left").lower()},
                      points=[pp0, pp1], original=command, original_space=space,
                      original_points=[p0, p1])

    # No fabricated sequencing: other multi-call snippets have no intermediate
    # screenshot and are rejected as a single next-action target.
    if len(calls) != 1:
        return None
    fn, args, kw = calls[0]

    def arg(index: int, name: str, default=None):
        if name in kw:
            return kw[name]
        return args[index] if len(args) > index else default

    def xy(default_start: int = 0) -> Optional[Tuple[float, float]]:
        x = kw.get("x", kw.get("xcoordinate"))
        y = kw.get("y", kw.get("ycoordinate"))
        if x is None and len(args) > default_start:
            x = args[default_start]
        if y is None and len(args) > default_start + 1:
            y = args[default_start + 1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        return float(x), float(y)

    def mk(verb: str, raw_xy=None, **extra) -> Optional[Action]:
        pts: List[Tuple[int, int]] = []
        originals: List[Tuple[float, float]] = []
        if raw_xy is not None:
            p = Point(raw_xy[0], raw_xy[1], space).to_pixels(width, height)
            if not validate_point(p[0], p[1], width, height):
                return None
            pts = [p]
            originals = [raw_xy]
        return Action(verb=verb, args=dict(extra), points=pts, original=command,
                      original_space=space, original_points=originals)

    if fn in ("click", "clickxy", "doubleclick", "rightclick", "middleclick"):
        raw_xy = xy()
        if raw_xy is None:
            return None
        button = str(kw.get("button", "left") or "left").lower()
        clicks = kw.get("clicks", kw.get("numclicks", 1))
        if fn == "rightclick":
            button = "right"
        elif fn == "middleclick":
            button = "middle"
        verb = "double_click" if fn == "doubleclick" or clicks == 2 else "click"
        return mk(verb, raw_xy, button=button, count=int(clicks) if isinstance(clicks, int) else 1)
    if fn in ("moveto", "move"):
        raw_xy = xy()
        return mk("move", raw_xy) if raw_xy else None
    if fn in ("dragto", "drag"):
        # pyautogui.dragTo only supplies an endpoint.  Without an explicit
        # cursor start this cannot be represented faithfully as x1/y1->x2/y2.
        return None
    if fn == "scroll":
        clicks = kw.get("clicks", args[0] if args else None)
        if not isinstance(clicks, (int, float)):
            return None
        raw_xy = None
        if "x" in kw or "y" in kw:
            x, y = kw.get("x"), kw.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return None
            raw_xy = (float(x), float(y))
        return mk("scroll", raw_xy, clicks=int(clicks))
    if fn in ("typewrite", "write", "type"):
        text = kw.get("text", args[0] if args else None)
        if text is None:
            return None
        if isinstance(text, (list, tuple)):
            text = "".join(str(x) for x in text)
        elif not isinstance(text, (str, int, float, bool)):
            return None
        return mk("type", text=str(text))
    if fn == "press":
        key = kw.get("key", kw.get("keys", args[0] if args else None))
        if isinstance(key, (list, tuple)):
            if len(key) != 1:
                return None
            key = key[0]
        return mk("press", key=str(key)) if key not in (None, "") else None
    if fn == "hotkey":
        keys = kw.get("keys", args)
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+") if k.strip()]
        if not isinstance(keys, (list, tuple)) or not keys:
            return None
        return mk("hotkey", keys=[str(k) for k in keys])
    if fn == "keydown":
        key = kw.get("key", args[0] if args else None)
        return mk("key_down", key=str(key)) if key not in (None, "") else None
    if fn == "keyup":
        key = kw.get("key", args[0] if args else None)
        return mk("key_up", key=str(key)) if key not in (None, "") else None
    if fn == "mousedown":
        return mk("mouse_down", button=str(kw.get("button", args[0] if args else "left") or "left").lower())
    if fn == "mouseup":
        return mk("mouse_up", button=str(kw.get("button", args[0] if args else "left") or "left").lower())
    if fn in ("sleep", "wait", "pause"):
        secs = kw.get("seconds", kw.get("duration", args[0] if args else 1.0))
        if not isinstance(secs, (int, float)):
            return None
        return mk("wait", seconds=float(secs))
    return None


# PC-Agent-E action strings, e.g. "click (654, 191)", "type hello world",
# "press ctrl+s", "scroll down", "hotkey alt+f4", "drag from (a,b) to (c,d)"
_PCA_CLICK_RE = re.compile(r"^(?P<verb>click|double[- ]?click|right[- ]?click)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.I)
_PCA_COORD_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
# canonical drag form observed in 9/9 audited drag actions:
#   "drag from (383, 299) to (763, 299)"
_PCA_DRAG_RE = re.compile(
    r"^drag\s+from\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s+to\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$", re.I)
# parenthesized key lists: "hotkey (Ctrl, A)"
_PCA_HOTKEY_LIST_RE = re.compile(r"^\(\s*[A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)+\s*\)$")
# numeric scroll payloads: "(-2)" or "-2"
_PCA_SCROLL_NUM_RE = re.compile(r"^\(?(-?\d+)\)?$")


def parse_pc_agent_e(action: str, width: int, height: int) -> Optional[Action]:
    """Parse PC-Agent-E 'action' strings (absolute pixels)."""
    if not action:
        return None
    text = action.strip()
    low = text.lower()
    m = _PCA_CLICK_RE.match(low)
    coords = _PCA_COORD_RE.search(low)
    if m:
        verb = "click"
        if "double" in m.group("verb"):
            verb = "double_click"
        elif "right" in m.group("verb"):
            verb = "right_click"
        x, y = int(m.group(2)), int(m.group(3))
        p = Point(x, y, CoordSpace.PIXEL).to_pixels(width, height)
        return Action(verb, points=[p], original=action, original_space=CoordSpace.PIXEL,
                      original_points=[(x, y)])
    if low.startswith("type"):
        payload = text[4:].strip().lstrip(">").strip()
        # real source format is "type text: <content>" (audit 2026-08-16) —
        # the literal "text:" prefix is markup, not typed content
        payload = re.sub(r"^text\s*:\s*", "", payload, count=1, flags=re.I)
        return Action("type", args={"text": payload}, original=action,
                      original_space=CoordSpace.PIXEL)
    if low.startswith(("press", "hotkey", "key")):
        payload = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        if not payload:
            return None
        # "hotkey (Ctrl, A)" — parenthesized key list (audit-confirmed format)
        m = _PCA_HOTKEY_LIST_RE.match(payload)
        if m:
            keys = [k.strip() for k in payload.strip("()").split(",") if k.strip()]
            return Action("hotkey", args={"keys": keys}, original=action,
                          original_space=CoordSpace.PIXEL)
        if "+" in payload:
            keys = [k.strip() for k in payload.split("+") if k.strip()]
            return Action("hotkey", args={"keys": keys}, original=action, original_space=CoordSpace.PIXEL)
        # "press key enter" — the word "key" is source markup, not the key name
        key = re.sub(r"^key\s+", "", payload, count=1, flags=re.I)
        return Action("press", args={"key": key}, original=action,
                      original_space=CoordSpace.PIXEL)
    if low.startswith("finish"):
        payload = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        status = "success" if "fail" not in payload.lower() else "failure"
        return Action("finish", args={"status": status}, original=action,
                      original_space=CoordSpace.PIXEL)
    if low.startswith("wait"):
        payload = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        secs = None
        m = re.search(r"(\d+(?:\.\d+)?)", payload)
        if m:
            secs = float(m.group(1))
        a = Action("wait", args={"seconds": secs} if secs is not None else {},
                   original=action, original_space=CoordSpace.PIXEL)
        return a
    if low.startswith("scroll"):
        payload = text.split(None, 1)[1].strip() if len(text.split(None, 1)) > 1 else ""
        m = _PCA_SCROLL_NUM_RE.match(payload.strip())
        if m:
            # "scroll (-2)" — signed click count (audit: sign was silently lost)
            clicks = int(m.group(1))
            return Action("scroll", args={"clicks": clicks}, original=action,
                          original_space=CoordSpace.PIXEL)
        direction = payload.split()[0].lower() if payload.split() else "down"
        clicks = 3 if direction in ("down", "up") else 2
        if direction == "up":
            clicks = -clicks
        if coords:
            x, y = int(coords.group(1)), int(coords.group(2))
            p = Point(x, y, CoordSpace.PIXEL).to_pixels(width, height)
            return Action("scroll", args={"clicks": clicks}, points=[p], original=action,
                          original_space=CoordSpace.PIXEL, original_points=[(x, y)])
        return Action("scroll", args={"clicks": clicks}, original=action, original_space=CoordSpace.PIXEL)
    m = _PCA_DRAG_RE.match(low)
    if m:
        # structural parse: only "drag from (x1,y1) to (x2,y2)" is accepted;
        # unrelated numbers elsewhere in the string can never shift coordinates
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        p1 = Point(x1, y1, CoordSpace.PIXEL).to_pixels(width, height)
        p2 = Point(x2, y2, CoordSpace.PIXEL).to_pixels(width, height)
        return Action("drag", points=[p1, p2], original=action, original_space=CoordSpace.PIXEL,
                      original_points=[(x1, y1), (x2, y2)])
    return None


# VideoCUA action_log entries -> unified Action
def parse_videocua_action(entry: Dict[str, Any], width: int, height: int) -> Optional[Action]:
    at = str(entry.get("action_type", "")).upper()
    params = entry.get("action_params", {}) or {}

    def px(x, y):
        return Point(x, y, CoordSpace.PIXEL).to_pixels(width, height)

    def mk(verb, xy=None, **kw):
        pts = [px(*xy)] if xy else []
        return Action(verb, args=dict(kw), points=pts, original=json.dumps(entry),
                      original_space=CoordSpace.PIXEL,
                      original_points=[xy] if xy else [])

    if at in ("TERMINATE_SUCCESS", "TERMINATE", "AFTER_LAST_ACTION"):
        return Action("finish", args={"status": "success"}, original=at,
                      original_space=CoordSpace.PIXEL)
    if at == "CLICK":
        x, y = params.get("x"), params.get("y")
        if x is None or y is None:
            return None
        button = str(params.get("text", "left") or "left").lower()
        n = params.get("numClicks") or 1
        verb = "click"
        if button == "right":
            return mk("click", (x, y), button="right")
        if button == "middle":
            return mk("click", (x, y), button="middle")
        if int(n) >= 2:
            return mk("double_click", (x, y))
        return mk(verb, (x, y))
    if at == "MOVE_TO":
        x, y = params.get("x"), params.get("y")
        return mk("move", (x, y)) if x is not None and y is not None else None
    if at == "DRAG_TO":
        x, y = params.get("x"), params.get("y")
        sx, sy = params.get("start_x"), params.get("start_y")
        # A destination alone does not define a drag.  Source adapters may
        # reconstruct start_x/start_y only from an explicit preceding pointer
        # state (VideoCUA MOUSE_DOWN sequence); otherwise reject.
        if x is None or y is None or sx is None or sy is None:
            return None
        sp = Point(sx, sy, CoordSpace.PIXEL).to_pixels(width, height)
        ep = Point(x, y, CoordSpace.PIXEL).to_pixels(width, height)
        return Action("drag", args={"button": str(params.get("button", "left") or "left").lower()},
                      points=[sp, ep], original=json.dumps(entry),
                      original_space=CoordSpace.PIXEL,
                      original_points=[(sx, sy), (x, y)])
    if at == "SCROLL":
        amount = params.get("scrollY", params.get("scroll", 0)) or 0
        x, y = params.get("x"), params.get("y")
        xy = (x, y) if (x is not None and y is not None) else None
        return mk("scroll", xy, clicks=int(amount))
    if at in ("TYPE", "TYPING", "TEXT"):
        return mk("type", text=str(params.get("text", "")))
    if at == "PRESS":
        key = params.get("key", params.get("text", ""))
        return mk("press", key=str(key)) if key else None
    if at == "HOTKEY":
        keys = params.get("keys") or []
        if not keys:
            key_str = str(params.get("key", params.get("text", "")) or "")
            keys = [k.strip() for k in key_str.split("+") if k.strip()]
        return mk("hotkey", keys=[str(k) for k in keys]) if keys else None
    if at == "KEY_DOWN":
        return mk("key_down", key=str(params.get("key", params.get("text", "")) or ""))
    if at == "KEY_UP":
        return mk("key_up", key=str(params.get("key", params.get("text", "")) or ""))
    if at == "MOUSE_DOWN":
        return mk("mouse_down", button=str(params.get("button", "left") or "left").lower())
    if at == "MOUSE_UP":
        return mk("mouse_up", button=str(params.get("button", "left") or "left").lower())
    return None


# GUI-360 tool_calls -> unified Action. Coordinates are NORM_0_1000 against
# metadata.others.resolution.
def _gui360_norm_point(raw: Any, width: int, height: int) -> Optional[Tuple[int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None
    p = Point(x, y, CoordSpace.NORM_0_1000).to_pixels(width, height)
    return p if validate_point(p[0], p[1], width, height) else None


def _simple_hotkey(keys: Sequence[Any]) -> Optional[List[str]]:
    """Return a canonical hotkey only for an unambiguous simultaneous combo.

    GUI-360's raw ``key`` tool sometimes stores arbitrary key *sequences* in
    the same list (including repeated modifiers and typed characters).  Those
    cannot safely be converted to one ``hotkey`` action.  We accept one key or
    conventional modifiers followed by one terminal key only.
    """
    kk = [str(k).strip().lower() for k in keys if str(k).strip()]
    if not kk:
        return None
    if len(kk) == 1:
        return kk
    modifiers = {"ctrl", "control", "alt", "shift", "cmd", "command", "win", "windows", "meta"}
    if len(set(kk)) != len(kk):
        return None
    if all(k in modifiers for k in kk[:-1]) and kk[-1] not in modifiers:
        return kk
    return None


def parse_gui360_tool_call(name: str, arguments: Dict[str, Any], width: int, height: int) -> Optional[Action]:
    """Parse one GUI-360 tool call using the verified [0,1000] convention.

    The parser is intentionally strict.  GUI-360 contains a ``key`` tool whose
    list can mean either one key/hotkey or a longer key sequence; only the
    former is safe for the one-action JxAgent target contract.  Drag, scroll and
    coordinate endpoints are validated before an Action is returned.
    """
    if width <= 0 or height <= 0 or not isinstance(arguments, dict):
        return None
    name = (name or "").lower().strip()
    original = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
    if name in ("terminate", "done", "finish", "stop"):
        status = str(arguments.get("status") or "success").lower()
        if status not in {"success", "failure", "failed", "cancelled", "canceled"}:
            status = "success"
        return Action("finish", args={"status": status}, original=original,
                      original_space=CoordSpace.NORM_0_1000)

    coord = arguments.get("coordinate") or arguments.get("point")
    if name in ("point", "click", "double_click", "right_click", "move"):
        p = _gui360_norm_point(coord, width, height)
        if p is None:
            return None
        if name == "point":
            verb, extra = "point", {}
        elif name == "right_click":
            verb, extra = "click", {"button": "right"}
        else:
            verb = "double_click" if name == "double_click" or int(arguments.get("clicks", 1) or 1) >= 2 else name
            extra = {"button": str(arguments.get("button", "left") or "left").lower()}
        return Action(verb, points=[p], args=extra, original=original,
                      original_space=CoordSpace.NORM_0_1000,
                      original_points=[(float(coord[0]), float(coord[1]))])

    if name == "drag":
        start = arguments.get("start_coordinate")
        end = arguments.get("coordinate")
        if start is None or end is None:
            pts_raw = arguments.get("path") or arguments.get("points") or []
            if isinstance(pts_raw, (list, tuple)) and len(pts_raw) >= 2:
                start, end = pts_raw[0], pts_raw[-1]
        p1 = _gui360_norm_point(start, width, height)
        p2 = _gui360_norm_point(end, width, height)
        if p1 is None or p2 is None or p1 == p2:
            return None
        return Action("drag", points=[p1, p2], args={"button": str(arguments.get("button", "left") or "left").lower()},
                      original=original, original_space=CoordSpace.NORM_0_1000,
                      original_points=[(float(start[0]), float(start[1])),
                                       (float(end[0]), float(end[1]))])

    if name in ("type", "typewrite"):
        text = arguments.get("text")
        if text is None or not str(text):
            return None
        return Action("type", args={"text": str(text)}, original=original,
                      original_space=CoordSpace.NORM_0_1000)

    if name == "scroll":
        amount = arguments.get("clicks", arguments.get("amount", 0))
        try:
            amount = int(amount or 0)
        except (TypeError, ValueError):
            return None
        direction = str(arguments.get("direction") or "").lower().strip()
        if direction in {"up", "left"}:
            amount = -abs(amount)
        elif direction in {"down", "right"}:
            amount = abs(amount)
        elif direction and direction not in {"vertical", "horizontal"}:
            return None
        if amount == 0:
            return None
        p = _gui360_norm_point(coord, width, height) if coord is not None else None
        if coord is not None and p is None:
            return None
        return Action("scroll", args={"clicks": amount}, points=[p] if p else [],
                      original=original, original_space=CoordSpace.NORM_0_1000,
                      original_points=[(float(coord[0]), float(coord[1]))] if p else [])

    if name in ("press", "key"):
        if name == "press":
            key = arguments.get("key")
            keys = [key] if key is not None else arguments.get("keys") or []
        else:
            keys = arguments.get("keys") or []
        canonical = _simple_hotkey(keys if isinstance(keys, (list, tuple)) else [keys])
        if not canonical:
            return None
        if len(canonical) == 1:
            return Action("press", args={"key": canonical[0]}, original=original,
                          original_space=CoordSpace.NORM_0_1000)
        return Action("hotkey", args={"keys": canonical}, original=original,
                      original_space=CoordSpace.NORM_0_1000)

    if name == "hotkey":
        keys = arguments.get("keys") or []
        canonical = _simple_hotkey(keys if isinstance(keys, (list, tuple)) else [keys])
        if not canonical or len(canonical) < 2:
            return None
        return Action("hotkey", args={"keys": canonical}, original=original,
                      original_space=CoordSpace.NORM_0_1000)
    return None


def rebase_action_points(action: Action, old_size: Sequence[int], new_size: Sequence[int]) -> Action:
    """Rescale an Action's pixel points when its image is resized."""
    pts = [scale_point(x, y, old_size, new_size) for (x, y) in action.points]
    action.points = pts
    return action
