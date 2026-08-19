#!/usr/bin/env python3
"""Measure JxAgent assistant-loss token balance with the frozen base tokenizer."""
from __future__ import annotations
import argparse, json, os
from collections import defaultdict
from pathlib import Path


def rows(path):
    with open(path,encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True); ap.add_argument('--model',required=True)
    ap.add_argument('--max-aux-share',type=float,default=0.20); ap.add_argument('--output',default=None)
    a=ap.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True,local_files_only=True)
    agg={axis:defaultdict(lambda:{'samples':0,'input_text_tokens':0,'image_tokens_estimated':0,'assistant_loss_tokens':0})
         for axis in ('source','task_type')}
    total={'samples':0,'input_text_tokens':0,'image_tokens_estimated':0,'assistant_loss_tokens':0}
    from processing.token_budget import estimate_image_tokens
    for split in ('train.jsonl','validation.jsonl'):
        p=Path(a.dataset)/'final'/split
        if not p.is_file(): continue
        for s in rows(p):
            inp=loss=0
            for m in s.get('messages') or []:
                n=len(tok.encode(str(m.get('content') or ''),add_special_tokens=False))
                if m.get('role')=='assistant' and m.get('loss',True) is not False: loss+=n
                else: inp+=n
            meta=s.get('metadata') or {}; img=0
            fs=meta.get('final_image_size')
            if isinstance(fs,list) and len(fs)>=2:
                img=estimate_image_tokens(int(fs[0]),int(fs[1]))*len(s.get('images') or [])
            total['samples']+=1; total['input_text_tokens']+=inp; total['assistant_loss_tokens']+=loss; total['image_tokens_estimated']+=img
            for axis,key in (('source',str(s.get('source') or 'unknown')),('task_type',str(s.get('task_type') or 'unknown'))):
                r=agg[axis][key]; r['samples']+=1; r['input_text_tokens']+=inp; r['assistant_loss_tokens']+=loss; r['image_tokens_estimated']+=img
    denom=max(1,total['assistant_loss_tokens']); out={'method':'exact_base_tokenizer_for_text__estimated_image_patches','total':total}
    for axis,d in agg.items():
        out[axis]={}
        for k,r in sorted(d.items()):
            r=dict(r); r['assistant_loss_share']=round(r['assistant_loss_tokens']/denom,6); out[axis][k]=r
    auxiliary={k:v for k,v in out['task_type'].items() if k not in {'action','grounding'}}
    violations={k:v['assistant_loss_share'] for k,v in auxiliary.items() if v['assistant_loss_share']>a.max_aux_share}
    out['gate']={'max_auxiliary_task_share':a.max_aux_share,'violations':violations,'passed':not violations}
    op=Path(a.output or Path(a.dataset)/'final'/'loss_token_report.json'); op.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(op),'assistant_loss_tokens':total['assistant_loss_tokens'],'gate':out['gate']},indent=2))
    return 0 if not violations else 3
if __name__=='__main__': raise SystemExit(main())
