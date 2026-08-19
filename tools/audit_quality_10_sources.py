#!/usr/bin/env python3
"""Deep offline audit of every real source artifact bundled with JxAgent.

This is intentionally network-free. It exercises the hardened parsers and
validators against the checked-in .audit probes and every materialized Smoke*
sample. It is a pre-production gate, not a substitute for the final 95k audit.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".audit"
sys.path.insert(0, str(ROOT))

from processing.coordinates import (parse_gui360_tool_call, parse_pc_agent_e,
                                    parse_pyautogui, parse_videocua_action)
from processing.validation import validate_sample
from sources.gui360 import (parse_messages, parse_meta, sanitize_understanding_controls,
                            use_row_to_trajectory, grounding_referent)
from sources.replay import (_canonical_tools, _coding_quality_ok, _instruction_quality_ok,
                            _math_difficulty, _math_quality_ok, _tool_row_valid)
from sources.videocua import _normalize_micro_actions


def _load(name):
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


def audit_procua(report):
    rows = _load("procua_trajs.json")
    n = sum(int(r.get("n_actions", 0)) for r in rows)
    ok = sum(int(r.get("ok_actions", 0)) for r in rows)
    report["procua"] = {
        "trajectories": len(rows), "raw_actions": n, "audit_parseable_actions": ok,
        "audit_parseable_rate": round(ok / n, 6) if n else None,
        "note": "Bundled ProCUA audit artifact is summary-level; exact command regression is covered by parser tests and SmokeProCUA validation.",
    }


def _tool_calls(msg):
    tcs = msg.get("tool_calls") or []
    return tcs if isinstance(tcs, list) else []


def audit_gui360(report):
    by = _load("gui360_rows.json")
    out = {}
    use = by.get("desktop.use") or []
    assistant_turns = calls = multi = parseable_first = traj_ok = steps = 0
    bad_coord = 0
    for wrapped in use:
        row = wrapped.get("row", wrapped) if isinstance(wrapped, dict) else wrapped
        meta = parse_meta(row)
        res = (meta.get("others") or {}).get("resolution") or [1040, 736]
        for member in (row.get("data") or row.get("members") or [row]):
            for m in parse_messages(member):
                if m.get("role") != "assistant" or not _tool_calls(m):
                    continue
                tcs = _tool_calls(m); assistant_turns += 1; calls += len(tcs); multi += int(len(tcs)>1)
                fn = tcs[0].get("function") or {}; args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try: args=json.loads(args)
                    except Exception: args={}
                a = parse_gui360_tool_call(fn.get("name", ""), args, int(res[0]), int(res[1])) if isinstance(args,dict) else None
                parseable_first += int(a is not None)
                if a and any(x < 0 or y < 0 or x >= int(res[0]) or y >= int(res[1]) for x,y in a.points): bad_coord += 1
        t = use_row_to_trajectory(row)
        if t:
            traj_ok += 1; steps += len(t.steps)
    out["use"] = {
        "rows": len(use), "assistant_tool_turns": assistant_turns, "tool_calls": calls,
        "multi_tool_turns": multi, "multi_tool_turn_rate": round(multi/max(1,assistant_turns),6),
        "parseable_first_calls": parseable_first,
        "parseable_first_rate": round(parseable_first/max(1,assistant_turns),6),
        "trajectories_emitted": traj_ok, "steps_emitted": steps, "parsed_oob_actions": bad_coord,
    }

    grounding = by.get("desktop.grounding.point") or []
    rows_with_text = referents = ambiguous = 0
    for wrapped in grounding:
        row = wrapped.get("row", wrapped) if isinstance(wrapped, dict) else wrapped
        found_text = False
        for member in (row.get("data") or row.get("members") or [row]):
            for m in parse_messages(member):
                for c in (m.get("content") or []):
                    if isinstance(c,dict) and c.get("type") in {"text","inline_reasoning","action_description"}:
                        text=str(c.get("text") or "").strip()
                        if text:
                            found_text=True
                            ref=grounding_referent(text)
                            if ref: referents += 1
                            elif any(k in text.lower() for k in ("click","point","select","press")): ambiguous += 1
        rows_with_text += int(found_text)
    out["grounding"]={"rows":len(grounding),"rows_with_text":rows_with_text,
                      "high_precision_referents_found":referents,"ambiguous_action_language":ambiguous}

    und = by.get("desktop.understanding") or []
    raw_controls=clean_controls=malformed=rows_clean=0
    spill=0
    for wrapped in und:
        row=wrapped.get("row",wrapped) if isinstance(wrapped,dict) else wrapped
        answer=""
        for member in (row.get("data") or row.get("members") or [row]):
            for m in parse_messages(member):
                if m.get("role")=="assistant":
                    for c in (m.get("content") or []):
                        if isinstance(c,dict) and c.get("type")=="text": answer=str(c.get("text") or "")
        try: controls=json.loads(answer)
        except Exception: malformed += 1; continue
        if not isinstance(controls,list): malformed += 1; continue
        raw_controls += len(controls)
        for c in controls:
            rect=c.get("control_rect") if isinstance(c,dict) else None
            if isinstance(rect,(list,tuple)) and len(rect)>=4:
                try:
                    vals=list(map(float,rect[:4])); spill += int(min(vals)<0 or max(vals)>1000)
                except Exception: pass
        cleaned=sanitize_understanding_controls(controls)
        clean_controls += len(cleaned); rows_clean += int(bool(cleaned))
    out["understanding"]={"rows":len(und),"malformed_rows":malformed,"raw_controls":raw_controls,
                          "raw_rects_outside_0_1000":spill,"sanitized_controls":clean_controls,
                          "rows_with_valid_sanitized_target":rows_clean}
    report["gui360"] = out


def _find_bbox_list(x):
    out=[]
    def walk(v):
        if isinstance(v,dict):
            if isinstance(v.get("bbox"),(list,tuple)) and len(v["bbox"])>=4: out.append(v)
            for z in v.values(): walk(z)
        elif isinstance(v,list):
            for z in v: walk(z)
    walk(x); return out


def audit_groundcua(report):
    rows=_load("groundcua_annotations.json")
    boxes=[]
    for r in rows: boxes.extend(_find_bbox_list(r))
    invalid=deg=0; finite=0
    dims=[]
    for e in boxes:
        try: x1,y1,x2,y2=map(float,e["bbox"][:4]); finite += int(all(math.isfinite(v) for v in (x1,y1,x2,y2))); deg += int(x2<=x1 or y2<=y1); dims.append((x2-x1,y2-y1))
        except Exception: invalid += 1
    report["groundcua"]={"records":len(rows),"bboxes":len(boxes),"numeric_finite_bboxes":finite,
                          "malformed_bboxes":invalid,"degenerate_bboxes":deg,
                          "thin_under_6px":sum(1 for w,h in dims if min(w,h)<6)}


def audit_pcae(report):
    tasks=_load("pcae_tasks.json")
    actions=parsed=bad_json=0; verbs=Counter(); recovery_first=0; explicit_recovery=0
    from sources.pc_agent_e import _RECOVERY_RE
    for t in tasks:
        for i,line in enumerate(str(t.get("jsonl") or "").splitlines()):
            if not line.strip(): continue
            try: ev=json.loads(line)
            except Exception: bad_json += 1; continue
            actions += 1
            a=parse_pc_agent_e(ev.get("action", ""),1920,1080)
            if a: parsed += 1; verbs[a.verb]+=1
            thought=str(ev.get("thought") or "")
            if _RECOVERY_RE.search(thought):
                explicit_recovery += 1; recovery_first += int(i==0)
    report["pcagente"]={"tasks":len(tasks),"events":actions,"parseable_actions":parsed,
                         "parse_rate":round(parsed/max(1,actions),6),"bad_json_lines":bad_json,
                         "verbs":dict(verbs),"explicit_recovery_thoughts":explicit_recovery,
                         "first_step_recovery_thoughts":recovery_first}


def _video_tasks(obj):
    for v in obj.values():
        if isinstance(v,dict):
            for t in v.get("logs",[]): yield t
        elif isinstance(v,list):
            for t in v: yield t


def audit_video(report):
    data=_load("videocua_logs.json")
    tasks=list(_video_tasks(data)); raw=norm=parseable=same_ts=drags=ambiguous_drags=0; maxx=maxy=0.0
    by_app=Counter()
    for t in tasks:
        acts=list(t.get("action_log") or []); raw += len(acts); by_app[str(t.get("platform","unknown"))]+=len(acts)
        for a,b in zip(acts,acts[1:]):
            try: same_ts += int(abs(float(a.get("timestamp",0))-float(b.get("timestamp",0)))<=1e-6)
            except Exception: pass
        nas=_normalize_micro_actions(acts); norm += len(nas)
        for e in nas:
            pts=[]; p=e.get("action_params") or {}
            if isinstance(p,dict):
                for xk,yk in (("x","y"),("start_x","start_y"),("end_x","end_y")):
                    if xk in p and yk in p:
                        try: pts.append((float(p[xk]),float(p[yk])))
                        except Exception: pass
            for x,y in pts: maxx=max(maxx,x); maxy=max(maxy,y)
            a=parse_videocua_action(e,4096,2160)
            parseable += int(a is not None)
            if str(e.get("action_type","")).upper()=="DRAG_TO":
                drags += 1; ambiguous_drags += int(a is None)
    report["videocua"]={"tasks":len(tasks),"raw_actions":raw,"normalized_actions":norm,
                         "parseable_at_4096x2160":parseable,"same_timestamp_adjacent_pairs":same_ts,
                         "drags":drags,"ambiguous_drags":ambiguous_drags,
                         "max_raw_x":maxx,"max_raw_y":maxy,"actions_by_app":dict(by_app)}


def audit_replay(report):
    rr=_load("replay_rows.json")
    magic=_load("replay_Magicoder-.json"); mathrows=_load("replay_orca-math-.json")
    coding_good=sum(1 for r in magic if _coding_quality_ok(str(r.get("instruction") or ""),str(r.get("response") or "")))
    diff=Counter(); math_good=0
    for r in mathrows:
        q=str(r.get("question") or ""); a=str(r.get("answer") or "")
        if _math_quality_ok(q,a): math_good += 1; diff[_math_difficulty(q)] += 1
    sm=rr.get("smoltalk") or []; sm_good=0; quality=Counter()
    for r in sm:
        quality[str(r.get("quality") or "unknown").lower()] += 1
        msgs=[{"role":m.get("role"),"content":str(m.get("content", ""))} for m in (r.get("messages") or []) if m.get("role") in {"system","user","assistant"} and str(m.get("content","")).strip()]
        sm_good += int(_instruction_quality_ok(r,msgs))
    her=rr.get("hermes") or []; tool_good=conflicts=invalid=0
    for r in her:
        info=_canonical_tools(r.get("tools"))
        if info is None: conflicts += 1; continue
        if _tool_row_valid(r.get("conversations") or [],info[1]): tool_good += 1
        else: invalid += 1
    report["replay"]={"coding_probe_rows":len(magic),"coding_quality_accept":coding_good,
                       "math_probe_rows":len(mathrows),"math_quality_accept":math_good,"math_difficulty":dict(diff),
                       "smoltalk_rows":len(sm),"smoltalk_quality_metadata":dict(quality),"smoltalk_accept":sm_good,
                       "hermes_rows":len(her),"hermes_valid":tool_good,"hermes_conflicting_schemas":conflicts,"hermes_invalid":invalid,
                       "cauldron_rows":len(rr.get("cauldron_aokvqa") or [])}


def audit_smokes(report):
    roots=sorted([p for p in ROOT.glob("Smoke*") if p.is_dir()])
    total=valid=0; reasons=Counter(); sources=Counter(); unique=set(); dup=0
    per_root={}
    for root in roots:
        state=root/"state"/"selected_samples.jsonl"
        if not state.exists(): continue
        n=ok=0; rr=Counter()
        for line in state.read_text(encoding="utf-8",errors="replace").splitlines():
            if not line.strip(): continue
            try: s=json.loads(line)
            except Exception: rr["json_decode"]+=1; continue
            key=(s.get("source"),s.get("trajectory_id"),s.get("step_id"))
            if key in unique: dup+=1
            else: unique.add(key)
            n+=1; total+=1; sources[s.get("source","?")]+=1
            good,why=validate_sample(s,str(root))
            if good: ok+=1; valid+=1
            else: rr[why]+=1; reasons[why]+=1
        per_root[root.name]={"samples":n,"valid":ok,"invalid_reasons":dict(rr)}
    report["materialized_smokes"]={"roots":per_root,"rows":total,"valid":valid,
                                   "valid_rate":round(valid/max(1,total),6),"duplicate_keys_across_smokes":dup,
                                   "invalid_reasons":dict(reasons),"sources":dict(sources)}


def main():
    report={"audit_version":"jxagent_quality_10_offline_v1","network_used":False}
    audit_procua(report); audit_gui360(report); audit_groundcua(report); audit_pcae(report); audit_video(report); audit_replay(report); audit_smokes(report)
    out=ROOT/"MACHINE_READABLE_SOURCE_AUDIT.json"
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"\nWROTE {out}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
