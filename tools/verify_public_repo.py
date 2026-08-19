#!/usr/bin/env python3
"""Fail closed on common accidental-publication mistakes."""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv)>1 else '.').resolve()
SKIP_DIRS={'.git','.venv','__pycache__','.pytest_cache'}
BAD_SUFFIX={'.safetensors','.gguf','.pt','.pth','.bin','.onnx'}
ABS_PATTERNS=[
    re.compile(r'[A-Za-z]:\\Users\\', re.I),
    re.compile(r'/home/[^/]+/'),
]
SECRET_PATTERNS=[
    re.compile(r'(?i)(hf_[A-Za-z0-9]{20,})'),
    re.compile(r'(?i)(sk-[A-Za-z0-9_-]{20,})'),
    re.compile(r'(?i)(github_pat_[A-Za-z0-9_]{20,})'),
]
errors=[]
for p in ROOT.rglob('*'):
    if any(part in SKIP_DIRS for part in p.parts):
        continue
    if p.is_dir():
        continue
    rel=p.relative_to(ROOT)
    if p.suffix.lower() in BAD_SUFFIX:
        errors.append(f'forbidden model/checkpoint file: {rel}')
    if p.name=='.env':
        errors.append(f'private env file: {rel}')
    if p.stat().st_size>5_000_000:
        errors.append(f'unexpected large tracked file >5MB: {rel}')
    if rel.as_posix() == 'tools/verify_public_repo.py':
        continue
    if p.suffix.lower() not in {'.py','.sh','.md','.txt','.yaml','.yml','.json','.toml','.cfg','.ini','.cff',''}:
        continue
    try: text=p.read_text(encoding='utf-8',errors='ignore')
    except Exception: continue
    for rx in ABS_PATTERNS:
        if rx.search(text): errors.append(f'machine-local absolute path in {rel}')
    for rx in SECRET_PATTERNS:
        if rx.search(text): errors.append(f'possible secret in {rel}')
if errors:
    print('\n'.join('[public-repo][FAIL] '+e for e in sorted(set(errors))))
    raise SystemExit(2)
print('[public-repo] PASS')
