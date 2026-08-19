"""General Replay adapter: capability preservation data (spec section 5).

Fully public sources with permissive licenses (verified 2026-08). Canonical
mixture = CATEGORIES below (single source of truth, mirrored in
configs/dataset.yaml):

  Coding   1,600  ise-uiuc/Magicoder-Evol-Instruct-110K   Apache-2.0
                   {instruction, response}
  Math     1,500  microsoft/orca-math-word-problems-200k  MIT
                   {question, answer}
  Instruct 1,700  HuggingFaceTB/smoltalk 'smol-magpie-ultra'  Apache-2.0
                   {messages: [{role, content}]}
  VQA      1,400  HuggingFaceM4/the_cauldron 'aokvqa' (+ 'ai2d' fallback)
                   {images: [Image], texts: [{user, assistant}...]}
                   (per-subset licenses; A-OKVQA annotations CC-BY-family,
                    COCO images CC-BY; prompts CC-BY-4.0)
  Tool     1,300  NousResearch/hermes-function-calling-v1 (Apache-2.0)
                   glaive_func_calling + func_calling_singleturn
                   {conversations: [{from, value}], tools: str}

All consumed via HF datasets streaming (no local mirroring). Replay sample
ids are content hashes, so identical content always lands in the same split
(no near-duplicate leakage across train/validation).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

from processing.assemble import assemble_replay
from processing.images import load_image
from .common import BuildContext
from .revisions import revision_for


def _iter_dataset(ctx: BuildContext, repo: str, config: Optional[str], split: str = "train"):
    from datasets import load_dataset
    from itertools import chain
    if ctx.offline:
        raise RuntimeError(f"offline mode cannot stream {repo}")
    rev = revision_for(repo)  # immutable Run 1 snapshot per component dataset
    if config:
        ds = load_dataset(repo, config, split=split, streaming=True, revision=rev)
    else:
        ds = load_dataset(repo, split=split, streaming=True, revision=rev)
    # Eagerly verify the stream is reachable.  HF load_dataset(streaming=True)
    # is lazy — auth, network, and revision errors only surface when the first
    # row is requested.  Probing here converts them into immediate stream-
    # creation failures that callers can catch and handle explicitly.
    it = iter(ds)
    try:
        first = next(it)
    except StopIteration:
        return iter([])  # genuinely empty dataset
    return chain([first], it)


def _select(row_index: int, repo: str, category: str, take_prob: float = 0.12) -> bool:
    """Deterministic row selection so reruns pick identical rows."""
    h = int(hashlib.md5(f"replay:{repo}:{category}:{row_index}".encode()).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < take_prob


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).strip().splitlines())


def _content_id(text: str) -> str:
    """Legacy deterministic text id retained for compatibility/tests.

    Production replay ids use :func:`_canonical_sample_id` so images, tool
    schemas and full conversations participate in identity.
    """
    return hashlib.sha1(str(text).encode("utf-8", "ignore")).hexdigest()[:16]


def _pil_digest(img: Image.Image) -> str:
    """Stable image-content digest independent of temporary file/path names."""
    im = img if img.mode in ("RGB", "L") else img.convert("RGB")
    h = hashlib.sha256()
    h.update(f"{im.mode}:{im.size[0]}x{im.size[1]}\0".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def _canonical_sample_id(prefix: str, messages: List[Dict[str, str]],
                         images: Optional[List[Image.Image]] = None,
                         extra: Optional[dict] = None) -> str:
    payload = {
        "messages": [{"role": str(m.get("role", "")),
                      "content": _normalize_text(m.get("content", ""))}
                     for m in messages],
        "images": [_pil_digest(im) for im in (images or [])],
        "extra": extra or {},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:24]}"


_FENCED_CODE_RE = re.compile(r"```\s*([A-Za-z0-9_+.-]*)\s*\n(.*?)```", re.S)


def _detect_code_language(instr: str, resp: str) -> str:
    m = _FENCED_CODE_RE.search(resp)
    if m and m.group(1):
        lang = m.group(1).strip().lower()
        return {"py": "python", "python3": "python", "js": "javascript", "ts": "typescript"}.get(lang, lang)
    text = (instr + "\n" + resp[:500]).lower()
    for key, lang in (("python", "python"), ("javascript", "javascript"), ("typescript", "typescript"),
                      ("java", "java"), ("c++", "cpp"), ("sql", "sql"), ("rust", "rust")):
        if key in text:
            return lang
    return "unknown"


def _coding_audit(instr: str, resp: str) -> Optional[dict]:
    if len(instr.strip()) < 20 or len(resp.strip()) < 40:
        return None
    low = resp.lower().strip()
    if low in {"n/a", "none", "todo", "placeholder", "cannot answer"}:
        return None
    if any(x in low for x in ("<placeholder>", "your code here", "lorem ipsum", "todo: implement", "pass  # todo")):
        return None
    if resp.count("```") % 2:
        return None
    lines = [x.strip() for x in resp.splitlines() if x.strip()]
    if len(lines) >= 8 and len(set(lines)) <= max(2, len(lines) // 4):
        return None
    lang = _detect_code_language(instr, resp)
    syntax = "not_applicable"
    if lang == "python":
        import ast
        blocks = [b for l, b in _FENCED_CODE_RE.findall(resp) if l.lower() in {"python", "py", "python3"}]
        if not blocks and ("def " in resp or "import " in resp):
            blocks = [resp]
        parsed_any = False
        for block in blocks:
            # Indented fragments may intentionally be partial. Parse only
            # self-contained-looking blocks, failing closed if they claim to
            # be complete but are syntactically invalid.
            nonempty = [ln for ln in block.splitlines() if ln.strip()]
            if not nonempty or (nonempty[0].startswith((" ", "\t")) and not re.match(r"\s*(def|class)\b", nonempty[0])):
                continue
            if "..." in block and len(block) < 400:
                continue
            try:
                ast.parse(block)
                parsed_any = True
            except SyntaxError:
                return None
        syntax = "python_ast_ok" if parsed_any else "partial_or_unchecked"
    problem_type = "debugging" if re.search(r"\b(debug|fix|bug|error)\b", instr, re.I) else \
                   "algorithm" if re.search(r"\b(algorithm|complexity|implement|function)\b", instr, re.I) else "general"
    return {"code_language": lang, "syntax_check": syntax, "coding_problem_type": problem_type}


def _coding_quality_ok(instr: str, resp: str) -> bool:
    return _coding_audit(instr, resp) is not None


def _math_difficulty(question: str) -> str:
    """Conservative structural difficulty heuristic for Orca math.

    It is used only for balancing the already-approved 1,500 replay examples,
    not to assert mathematical correctness or benchmark difficulty.
    """
    q = question.lower()
    score = 0
    score += min(3, len(re.findall(r"\d+(?:\.\d+)?", q)) // 3)
    score += min(3, len(re.findall(
        r"[%/$]|fraction|ratio|percent|probab|average|speed|rate|area|volume|"
        r"discount|interest|exchange|per\s", q)))
    score += min(3, len(re.findall(
        r"\b(and|then|after|before|remaining|altogether|difference|more than|"
        r"less than|each|respectively|remainder)\b", q)))
    if len(q) > 180:
        score += 1
    if len(q) > 320:
        score += 1
    if score <= 1:
        return "easy"
    if score <= 4:
        return "medium"
    return "medium_hard"


def _safe_arithmetic_value(expr: str):
    import ast
    from fractions import Fraction
    node = ast.parse(expr, mode="eval")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand); return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            l, r = ev(n.left), ev(n.right)
            return l+r if isinstance(n.op, ast.Add) else l-r if isinstance(n.op, ast.Sub) else l*r if isinstance(n.op, ast.Mult) else l/r
        raise ValueError("unsupported")
    return ev(node)


def _parse_exact_number(text: str):
    from fractions import Fraction
    raw = text.strip().replace(",", "")
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", raw):
        a, b = re.split(r"/", raw)
        return Fraction(int(a), int(b))
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw):
        return Fraction(raw)
    raise ValueError("not an exact number")


def _verify_math_when_possible(q: str, a: str) -> Optional[bool]:
    """Conservative objective verifier for arithmetic and simple equations.

    Returns None whenever parsing is not unambiguous. A sample is rejected only
    when a supported verifier can prove the source answer wrong.
    """
    from fractions import Fraction
    qq = q.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    candidates = []
    for pat in (r"(?:what is|calculate|compute|evaluate)\s*[:]?\s*([-+*/().\d\s]+)\??$",
                r"^\s*([-+*/().\d\s]+)\s*=\s*\??\s*$"):
        m = re.search(pat, qq, re.I)
        if m:
            candidates.append(m.group(1).strip())
    if candidates:
        try:
            expected = _safe_arithmetic_value(candidates[0])
        except Exception:
            expected = None
        if expected is not None:
            nums = re.findall(r"(?<!\w)[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?", a.replace(",", ""))
            if not nums:
                return None
            try:
                got = Fraction(nums[-1])
            except Exception:
                return None
            return got == expected

    # Simple one-variable algebra only. Require explicit solve/equation wording
    # and a literal equation so ordinary word problems are never over-parsed.
    if not re.search(r"\b(solve|equation)\b", qq, re.I):
        return None
    eqs = re.findall(r"(?<![A-Za-z0-9_])([0-9xyXY+\-*/().\s]+=[0-9xyXY+\-*/().\s]+)", qq)
    if len(eqs) != 1:
        return None
    eq = eqs[0].strip()
    vars_found = sorted(set(re.findall(r"[xyXY]", eq.lower())))
    if len(vars_found) != 1:
        return None
    var = vars_found[0].lower()
    # Insert only the conventional unambiguous digit-variable multiplication.
    eq = re.sub(r"(?<=\d)(?=[xyXY])", "*", eq).lower()
    try:
        import sympy as sp
        sym = sp.Symbol(var)
        lhs, rhs = eq.split("=", 1)
        lhs_e = sp.sympify(lhs, locals={var: sym}, evaluate=True)
        rhs_e = sp.sympify(rhs, locals={var: sym}, evaluate=True)
        sols = sp.solve(sp.Eq(lhs_e, rhs_e), sym)
    except Exception:
        return None
    if len(sols) != 1 or not bool(getattr(sols[0], "is_rational", False)):
        return None
    m = re.search(rf"\b{re.escape(var)}\s*=\s*([-+]?\d+(?:\.\d+)?(?:/[-+]?\d+)?)", a.lower())
    if not m:
        return None
    try:
        got = _parse_exact_number(m.group(1))
        expected = Fraction(int(sp.numer(sols[0])), int(sp.denom(sols[0])))
    except Exception:
        return None
    return got == expected

def _math_quality_ok(q: str, a: str) -> bool:
    if len(q.strip()) < 12 or len(a.strip()) < 4:
        return False
    if len(q) > 6000 or len(a) > 10000:
        return False
    low = a.lower().strip()
    if low in {"n/a", "none", "unknown"} or "lorem ipsum" in low:
        return False
    verified = _verify_math_when_possible(q, a)
    return verified is not False


def _instruction_quality_ok(row: dict, messages: List[Dict[str, str]]) -> bool:
    quality = str(row.get("quality") or "").strip().lower()
    if quality and quality not in {"good", "excellent"}:
        return False
    # Coding and math have dedicated replay anchors; preserve this quota for
    # broader instruction-following rather than duplicating those domains.
    category = str(row.get("category") or "").strip().lower()
    if category in {"coding", "math"}:
        return False
    roles = [m["role"] for m in messages]
    if not roles or roles[0] == "assistant" or "user" not in roles or "assistant" not in roles:
        return False
    # Reject obvious role corruption / monologue loops.
    for a, b in zip(roles, roles[1:]):
        if a == b == "assistant":
            return False
    assistant = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if len(assistant.strip()) < 30:
        return False
    return True


def _json_type_ok(value, schema: dict) -> bool:
    """Conservative recursive subset of JSON-Schema used by Hermes tools.

    Unknown schema keywords are ignored rather than guessed, while declared
    types, enums, required fields, arrays, nested objects and common union
    forms are enforced. This catches structurally wrong tool calls without
    pretending to implement the entire JSON-Schema specification.
    """
    if not isinstance(schema, dict):
        return True
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    for union_key in ("oneOf", "anyOf"):
        choices = schema.get(union_key)
        if isinstance(choices, list) and choices:
            return any(_json_type_ok(value, c) for c in choices if isinstance(c, dict))
    typ = schema.get("type")
    if isinstance(typ, list):
        return any(_json_type_ok(value, {**schema, "type": t}) for t in typ)
    if typ == "null":
        return value is None
    if typ == "string":
        if not isinstance(value, str): return False
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]: return False
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]: return False
    elif typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool): return False
    elif typ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool): return False
    elif typ == "boolean":
        if not isinstance(value, bool): return False
    elif typ == "array":
        if not isinstance(value, list): return False
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]: return False
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]: return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict) and any(not _json_type_ok(v, item_schema) for v in value): return False
    elif typ == "object":
        if not isinstance(value, dict): return False
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if any(k not in value for k in required): return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value): return False
        for k, v in value.items():
            child = props.get(k)
            if isinstance(child, dict) and not _json_type_ok(v, child): return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]: return False
        if "maximum" in schema and value > schema["maximum"]: return False
    return True


def _canonical_tools(raw) -> Optional[Tuple[List[dict], Dict[str, dict]]]:
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if not isinstance(raw, list) or not raw:
        return None
    by_name: Dict[str, dict] = {}
    canonical: List[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None
        fn = entry.get("function") if entry.get("type") == "function" else entry
        if not isinstance(fn, dict):
            return None
        name = str(fn.get("name") or "").strip()
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        if not name or not isinstance(params, dict):
            return None
        clean = {"type": "function", "function": {
            "name": name,
            "description": str(fn.get("description") or "").strip(),
            "parameters": params,
        }}
        sig = json.dumps(clean, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if name in by_name:
            prev_sig = json.dumps(by_name[name]["entry"], sort_keys=True,
                                  ensure_ascii=False, separators=(",", ":"))
            if prev_sig != sig:
                return None  # same function name with conflicting schemas
            continue
        by_name[name] = {"entry": clean, "schema": params}
        canonical.append(clean)
    return canonical, by_name


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.I | re.S)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.I | re.S)
_ASSIGN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*([\"'])(.*?)\2")


def _tool_row_valid(convs: list, tools_by_name: Dict[str, dict]) -> bool:
    """High-precision structural and literal-consistency validation."""
    seen_call_names: List[str] = []
    literal_constraints: Dict[str, str] = {}
    for c in convs:
        if not isinstance(c, dict):
            return False
        frm = str(c.get("from") or "").lower()
        val = str(c.get("value") or "").strip()
        if not val:
            continue
        if frm == "human":
            # Only explicit assignment syntax is trusted as a semantic
            # constraint.  This catches rows such as lighting_scene="dim"
            # without guessing values from natural-language paraphrases.
            for key, _quote, value in _ASSIGN_RE.findall(val):
                literal_constraints[key] = value
        if frm in {"gpt", "assistant"} and "<tool_call>" in val:
            matches = _TOOL_CALL_RE.findall(val)
            if not matches:
                return False
            for raw_call in matches:
                try:
                    call = json.loads(raw_call)
                except Exception:
                    return False
                name = str(call.get("name") or "")
                args = call.get("arguments")
                if name not in tools_by_name or not isinstance(args, dict):
                    return False
                schema = tools_by_name[name]["schema"]
                props = schema.get("properties") or {}
                required = schema.get("required") or []
                if any(k not in args for k in required):
                    return False
                if schema.get("additionalProperties") is False and any(k not in props for k in args):
                    return False
                for key, value in args.items():
                    if key in props and isinstance(props[key], dict) and not _json_type_ok(value, props[key]):
                        return False
                    if key in literal_constraints and isinstance(value, str) and value != literal_constraints[key]:
                        return False
                seen_call_names.append(name)
        if frm in {"tool", "function"} and "<tool_response>" in val:
            matches = _TOOL_RESPONSE_RE.findall(val)
            if not matches:
                return False
            for raw_resp in matches:
                try:
                    response = json.loads(raw_resp)
                except Exception:
                    return False
                name = str(response.get("name") or "")
                if name and name not in tools_by_name:
                    return False
                if name and name not in seen_call_names:
                    return False
    return bool(seen_call_names)


# --------------------------------------------------------------- categories

def replay_coding(ctx: BuildContext, want: int) -> List[dict]:
    out = []
    idx = 0
    for row in _iter_dataset(ctx, "ise-uiuc/Magicoder-Evol-Instruct-110K", None):
        if len(out) >= want:
            break
        if not _select(idx, "magicoder", "coding"):
            idx += 1
            continue
        idx += 1
        instr = (row.get("instruction") or "").strip()
        resp = (row.get("response") or "").strip()
        audit = _coding_audit(instr, resp) if instr and resp else None
        if audit is None:
            continue
        messages = [{"role": "user", "content": instr},
                    {"role": "assistant", "content": resp}]
        sid = _canonical_sample_id("replay_coding", messages, extra={
            "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "revision": revision_for("ise-uiuc/Magicoder-Evol-Instruct-110K")})
        sample = assemble_replay(
            messages=messages,
            images_pil=[], source_name="replay",
            sample_id=sid, task_type="replay_coding",
            metadata={"replay_source": "Magicoder-Evol-Instruct-110K",
                      "license": "apache-2.0", **audit,
                      "content_family": f"coding::{audit['code_language']}::{audit['coding_problem_type']}"}, ctx=ctx)
        if sample:
            out.append(sample)
    return out


def replay_math(ctx: BuildContext, want: int) -> List[dict]:
    # Slightly harder anchor mix without changing the approved 1,500 quota.
    # The buckets are intentionally modest; this is capability preservation,
    # not olympiad post-training.
    targets = {
        "easy": int(round(want * 0.15)),
        "medium": int(round(want * 0.60)),
    }
    targets["medium_hard"] = want - targets["easy"] - targets["medium"]
    out: List[dict] = []
    counts = {k: 0 for k in targets}
    idx = 0
    repo = "microsoft/orca-math-word-problems-200k"
    rev = revision_for(repo)
    for row in _iter_dataset(ctx, repo, None):
        if len(out) >= want:
            break
        # Keep deterministic sparse scanning but prefer every structurally
        # harder candidate so the requested distribution is reachable quickly.
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip()
        tier = _math_difficulty(q) if q else "easy"
        selected = _select(idx, "orca-math", f"math:{tier}",
                           take_prob=0.24 if tier == "medium_hard" else 0.14)
        idx += 1
        if not selected or counts[tier] >= targets[tier] or not _math_quality_ok(q, a):
            continue
        messages = [{"role": "user", "content": q},
                    {"role": "assistant", "content": a}]
        sid = _canonical_sample_id("replay_math", messages,
                                   extra={"repo": repo, "revision": rev})
        sample = assemble_replay(
            messages=messages, images_pil=[], source_name="replay",
            sample_id=sid, task_type="replay_math",
            metadata={"replay_source": "orca-math-word-problems-200k",
                      "license": "mit", "difficulty_bucket": tier,
                      "math_verified": _verify_math_when_possible(q, a),
                      "content_family": f"math::{tier}"}, ctx=ctx)
        if sample:
            out.append(sample)
            counts[tier] += 1
    return out


def replay_instruction(ctx: BuildContext, want: int) -> List[dict]:
    out = []
    idx = 0
    for row in _iter_dataset(ctx, "HuggingFaceTB/smoltalk", "smol-magpie-ultra"):
        if len(out) >= want:
            break
        if not _select(idx, "smoltalk", "instruction"):
            idx += 1
            continue
        idx += 1
        msgs = row.get("messages") or []
        converted = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))}
                     for m in msgs
                     if m.get("role") in ("system", "user", "assistant")
                     and str(m.get("content", "")).strip()]
        if not converted or not _instruction_quality_ok(row, converted):
            continue
        repo = "HuggingFaceTB/smoltalk"
        sid = _canonical_sample_id("replay_instr", converted, extra={
            "repo": repo, "revision": revision_for(repo),
            "category": row.get("category"), "quality": row.get("quality")})
        sample = assemble_replay(
            messages=converted, images_pil=[], source_name="replay",
            sample_id=sid, task_type="replay_instruction",
            metadata={"replay_source": "smoltalk/smol-magpie-ultra",
                      "license": "apache-2.0",
                      "source_quality": row.get("quality"),
                      "source_category": row.get("category"),
                      "source_difficulty": row.get("difficulty")}, ctx=ctx)
        if sample:
            out.append(sample)
    return out


def _cauldron_messages(row: dict) -> Tuple[List[Dict[str, str]], List]:
    texts = row.get("texts") or []
    messages: List[Dict[str, str]] = []
    images = []
    raw_images = row.get("images") or []
    for i, t in enumerate(texts):
        q = str(t.get("user") or t.get("question") or "").strip()
        a = str(t.get("assistant") or t.get("answer") or "").strip()
        if not q or not a:
            continue
        if i == 0 and raw_images:
            images.append(raw_images[0])
            messages.append({"role": "user", "content": "<image>\n" + q})
        else:
            messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages, images


def replay_vqa(ctx: BuildContext, want: int) -> List[dict]:
    out = []
    configs = ["aokvqa", "ai2d"]
    for ci, config in enumerate(configs):
        if len(out) >= want:
            break
        idx = 0
        # Fail closed: a stream-creation failure on the last fallback
        # config aborts the build.  Earlier configs may fail so the
        # fallback chain has a chance to serve.
        try:
            stream = _iter_dataset(ctx, "HuggingFaceM4/the_cauldron", config)
        except Exception:
            if ci < len(configs) - 1:
                continue  # try next fallback config
            raise  # last fallback exhausted — fail closed
        for row in stream:
            if len(out) >= want:
                break
            if not _select(idx, "cauldron", "vqa"):
                idx += 1
                continue
            idx += 1
            messages, raw_images = _cauldron_messages(row)
            if not messages or not raw_images:
                continue
            pil_images = []
            ok = True
            for imf in raw_images:
                if isinstance(imf, dict) and imf.get("bytes"):
                    pil_images.append(load_image(imf["bytes"]))
                elif isinstance(imf, Image.Image):
                    # HF streaming delivers decoded PIL objects, not dicts
                    imf.load()
                    pil_images.append(imf if imf.mode in ("RGB", "L")
                                      else imf.convert("RGB"))
                elif imf is not None:
                    try:
                        pil_images.append(load_image(imf))
                    except Exception:
                        ok = False
            if not ok or not pil_images:
                continue
            repo = "HuggingFaceM4/the_cauldron"
            sid = _canonical_sample_id("replay_vqa", messages, pil_images, extra={
                "repo": repo, "revision": revision_for(repo), "config": config})
            sample = assemble_replay(
                messages=messages, images_pil=pil_images, source_name="replay",
                sample_id=sid, task_type="replay_vqa",
                metadata={"replay_source": f"the_cauldron/{config}",
                          "license": "per-subset (A-OKVQA annotations CC-BY family)"},
                ctx=ctx)
            if sample:
                out.append(sample)
    return out


def replay_tool(ctx: BuildContext, want: int) -> List[dict]:
    out: List[dict] = []
    idx = 0
    streams = []
    last_stream_error = None
    repo = "NousResearch/hermes-function-calling-v1"
    rev = revision_for(repo)
    for config in ("glaive_func_calling", "func_calling_singleturn"):
        try:
            streams.append((config, _iter_dataset(ctx, repo, config)))
        except Exception as e:
            last_stream_error = e
    if not streams:
        raise last_stream_error or RuntimeError("no Hermes tool streams available")
    exhausted = [False] * len(streams)
    while len(out) < want and not all(exhausted):
        for si, (config, stream) in enumerate(streams):
            if len(out) >= want:
                break
            if exhausted[si]:
                continue
            try:
                row = next(stream)
            except StopIteration:
                exhausted[si] = True
                continue
            selected = _select(idx, "hermes", f"tool:{config}", take_prob=0.16)
            idx += 1
            if not selected:
                continue
            convs = row.get("conversations") or []
            tool_info = _canonical_tools(row.get("tools"))
            if not convs or tool_info is None:
                continue
            canonical_tools, tools_by_name = tool_info
            if not _tool_row_valid(convs, tools_by_name):
                continue

            messages: List[Dict[str, str]] = []
            tool_note = "Available tools: " + json.dumps(
                canonical_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for c in convs:
                frm = (c.get("from") or "").lower()
                val = str(c.get("value") or "").strip()
                if not val:
                    continue
                role = {"system": "system", "gizmo": "system", "human": "user",
                        "function": "user", "tool": "user"}.get(frm, "assistant")
                if role == "user" and frm in ("function", "tool"):
                    val = "Tool response: " + val
                messages.append({"role": role, "content": val})
            if not any(m["role"] == "assistant" for m in messages):
                continue
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + tool_note
            else:
                messages.insert(0, {"role": "system", "content": tool_note})

            sid = _canonical_sample_id("replay_tool", messages, extra={
                "repo": repo, "revision": rev, "config": config,
                "tools": canonical_tools})
            sample = assemble_replay(
                messages=messages, images_pil=[], source_name="replay",
                sample_id=sid, task_type="replay_tool",
                metadata={"replay_source": "hermes-function-calling-v1",
                          "replay_config": config, "license": "apache-2.0",
                          "validated_tool_schema": True}, ctx=ctx)
            if sample:
                out.append(sample)
    return out


# 2026-08 quality pass: instruction-following is the most LoRA-fragile
# capability and the cheapest per token; VQA protects the vision pathway a
# CUA depends on; math trimmed slightly (synthetic GSM-style seeds overlap
# the regression panel's math evals).
CATEGORIES = {
    "coding": (replay_coding, 1600),
    "math": (replay_math, 1500),
    "instruction": (replay_instruction, 1700),
    "vqa": (replay_vqa, 1400),
    "tool": (replay_tool, 1300),
}


def run(ctx: BuildContext, counts: Optional[Dict[str, int]] = None) -> List[dict]:
    counts = counts or {name: n for name, (_, n) in CATEGORIES.items()}
    total = sum(counts.values())
    # Preserve the configured full target on resume, not merely the remaining
    # sub-quota for this process.
    ctx.state.set_target("replay", ctx.state.selected_total("replay") + total)
    out: List[dict] = []
    for name, (fn, default) in CATEGORIES.items():
        want = counts.get(name, default)
        if want <= 0:
            continue
        got = fn(ctx, want)
        if len(got) != want:
            # Exact category quotas are a hard Run-1 invariant.  An underfilled
            # VQA/tool cohort must never silently turn into a different mix.
            raise RuntimeError(
                f"mandatory Replay category '{name}' emitted {len(got)} samples "
                f"(wanted exactly {want}); failing closed")
        out.extend(got)
        ctx.persist_samples(got)
        ctx.state.add_selected("replay", len(got))
        ctx.state.save()
    ctx.quota["replay"] = 0
    return out
