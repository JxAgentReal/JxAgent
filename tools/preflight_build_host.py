#!/usr/bin/env python3
"""Build-host preflight (audit patch §21). Reports readiness, starts nothing.

Checks:
  - Python version (3.10+ required, 3.12 recommended)
  - free disk on the output path: <40 GB FAIL, 40-60 GB WARN, >=60 GB PASS
  - RAM (best effort, cross platform)
  - HF token presence (presence ONLY — the value is never printed)
  - required package versions vs requirements.txt major bounds
  - source revision resolution status (PENDING entries are fatal for production)
  - git builder revision
  - config hash preview (canonical default production mixture)
  - output lock state

Exit code: 0 = ready, 1 = at least one FAIL. Never starts a build.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GB = 1 << 30
MIN_DISK_GB, RECOMMENDED_DISK_GB = 40, 60

REQUIREMENT_RANGES = {
    # artifact-affecting majors (see BUILD_REPRODUCIBILITY_PATCH_REPORT.md)
    "pillow": (10, 13), "numpy": (1, 3), "datasets": (2, 6),
    "huggingface_hub": (0, 2), "zstandard": (0, 1), "av": (12, 19),
    "PyYAML": (6, 7), "requests": (2, 3), "tqdm": (4, 5),
}

FAILS: list[str] = []
WARNS: list[str] = []


def report(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> None:
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    line = f"[{tag}] {label}" + (f": {detail}" if detail else "")
    print(line)
    if ok:
        return
    if warn_only:
        WARNS.append(label)
    else:
        FAILS.append(label)


def ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / GB
        # Linux
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / (1 << 20)
    except Exception:
        pass
    return 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-path", default=".",
                   help="path where the dataset will be built (disk check target)")
    p.add_argument("--skip-packages", action="store_true")
    args = p.parse_args()

    print("== JxAgent build-host preflight ==")

    v = sys.version_info
    report(v >= (3, 10), "Python >= 3.10", platform.python_version(),
           warn_only=(3, 10) <= v < (3, 12))
    if v < (3, 10):
        print("Python 3.10+ is required (union syntax etc.); upgrade first.")

    usage = shutil.disk_usage(os.path.abspath(args.output_path))
    free_gb = usage.free / GB
    if free_gb >= RECOMMENDED_DISK_GB:
        report(True, f"free disk >= {RECOMMENDED_DISK_GB} GB", f"{free_gb:.1f} GB free")
    elif free_gb >= MIN_DISK_GB:
        report(False, f"free disk {MIN_DISK_GB}-{RECOMMENDED_DISK_GB} GB",
               f"{free_gb:.1f} GB free (recommend >= {RECOMMENDED_DISK_GB} GB)",
               warn_only=True)
    else:
        report(False, f"free disk >= {MIN_DISK_GB} GB",
               f"{free_gb:.1f} GB free — below technical safety minimum "
               f"(build ~25 GB + dataset ~21.5 GB)")

    mem = ram_gb()
    if mem:
        report(mem >= 31, "RAM >= 32 GB", f"{mem:.0f} GB", warn_only=mem >= 15)

    has_token = bool(os.environ.get("HF_TOKEN"))
    report(has_token, "HF_TOKEN present",
           "yes (value not printed)" if has_token
           else "MISSING — required for gui360 + replay streaming",
           warn_only=True)

    if not args.skip_packages:
        from importlib.metadata import version, PackageNotFoundError
        for pkg, (lo, hi) in REQUIREMENT_RANGES.items():
            try:
                v = version(pkg)
                major = int(re.match(r"(\d+)", v).group(1))
                report(lo <= major < hi, f"{pkg} major in [{lo}, {hi})", v,
                       warn_only=True)
            except PackageNotFoundError:
                report(False, f"{pkg} installed", "not installed")

    from sources.revisions import unresolved_sources
    unresolved = unresolved_sources()
    report(not unresolved, "source revisions resolved",
           f"unresolved: {unresolved}" if unresolved else "all pinned to commit SHAs")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=15,
                             cwd=os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__)))).stdout.strip()
        report(bool(sha), "git builder revision", sha or "not a git checkout")
    except Exception:
        report(False, "git builder revision",
               "unavailable — `git init && git add -A && git commit` before "
               "a production build", warn_only=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import build_jxagent_dataset as bj  # noqa: E402
        args = bj.parse_args(["--output", os.path.abspath(args.output_path)])
        snap = bj.build_config_snapshot(args, bj.load_config(None),
                                        dict(bj.SOURCE_TARGETS))
        report(True, "config hash preview", bj.config_hash_of(snap)[:16] + "...")
    except Exception as e:  # noqa: BLE001
        report(False, "config hash preview", f"error: {e}", warn_only=True)

    lock = os.path.join(os.path.abspath(args.output_path), "state", "build.lock")
    report(not os.path.exists(lock), "no active output lock",
           f"{lock} exists — another builder may be running" if os.path.exists(lock) else "")

    print(f"== preflight: {len(FAILS)} FAIL, {len(WARNS)} WARN ==")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
