"""Canonical immutable source revisions for JxAgent Run 1.

Every upstream Hugging Face dataset is pinned to the commit SHA its 'main'
branch pointed at when resolved (2026-08-16, via
GET /api/datasets/<repo>/revision/main -> .sha). SHAs are immutable on the HF
hub, so resolve/<sha>/ URLs and load_dataset(..., revision=<sha>) keep serving
the exact bytes used for Run 1 even after upstream moves.

The OSWorld decontamination reference lives on GitHub and is pinned the same
way (commits/main -> sha); its locally cached instruction list is additionally
fingerprinted (cache_sha256) so the build manifest can prove which reference
set was actually used.

RULES
- Do NOT change a SHA casually: a different SHA is a different source snapshot
  and therefore a different dataset (Run 2+ material). Record the reason.
- Do NOT invent SHAs. An unresolved source must carry the PENDING marker
  below; production builds then fail preflight instead of guessing.
- This file is the ONLY place revisions live (no duplicated constants).
"""
from __future__ import annotations

from typing import Dict, List

PENDING = "PENDING-REVISION-RESOLUTION"

# key -> {"repo": ..., "sha": immutable revision, "license": ..., "note": ...}
REVISIONS: Dict[str, Dict[str, str]] = {
    # ---- computer-use sources ----
    "procua": {
        "repo": "nvidia/ProCUA-SFT",
        "sha": "120de7e954f851c2d24399230367f2b01ff815f9",
        "license": "CC-BY-4.0",
    },
    "gui360": {
        "repo": "cua-lite/GUI-360",
        "sha": "daf3f14b53fe1926f18c4d3eca37914b09d62af5",
        "license": "MIT (origin vyokky/GUI-360)",
    },
    "videocua": {
        "repo": "ServiceNow/VideoCUA",
        "sha": "6ecc530df848f916bf0d195c66823ef8363abb93",
        "license": "MIT",
    },
    "groundcua": {
        "repo": "ServiceNow/GroundCUA",
        "sha": "5d6845b0116029d46ec762e734701c5b8ce207c3",
        "license": "MIT",
    },
    "pcagente": {
        "repo": "henryhe0123/PC-Agent-E",
        "sha": "4f64c7b055154bed9ffea677aa86aeb51a01cb73",
        "license": "MIT",
    },
    # ---- replay component datasets (repo id is the key) ----
    "ise-uiuc/Magicoder-Evol-Instruct-110K": {
        "repo": "ise-uiuc/Magicoder-Evol-Instruct-110K",
        "sha": "b0079beaa0361d82412520b873715bee59cc7dd4",
        "license": "Apache-2.0",
    },
    "microsoft/orca-math-word-problems-200k": {
        "repo": "microsoft/orca-math-word-problems-200k",
        "sha": "29255d1770cc4eac66e5e7fa378cba542c026350",
        "license": "MIT",
    },
    "HuggingFaceTB/smoltalk": {
        "repo": "HuggingFaceTB/smoltalk",
        "sha": "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa",
        "license": "Apache-2.0",
    },
    "HuggingFaceM4/the_cauldron": {
        "repo": "HuggingFaceM4/the_cauldron",
        "sha": "847a98a779b1652d65111daf20c972dfcd333605",
        "license": "per-subset CC-BY family (aokvqa, ai2d)",
    },
    "NousResearch/hermes-function-calling-v1": {
        "repo": "NousResearch/hermes-function-calling-v1",
        "sha": "dae3e1d28cfbcf4b915c04ea1e072030529b4bda",
        "license": "Apache-2.0",
    },
    # ---- decontamination reference (GitHub) ----
    "osworld": {
        "repo": "xlang-ai/OSWorld",
        "sha": "091f5ef1d5544bc74953c77875d5feb5bed30108",
        "license": "Apache-2.0 (evaluation examples)",
        # sha256 of .cache/osworld_instructions.json at pin time; the cache is
        # advisory (regenerated fetches may serialize differently), a mismatch
        # is logged, never fatal.
        "cache_sha256": "e812ebe76559941e90788d08a9db6cb19aea33cc09fe4a624cf4ed15c3f35287",
    },
}


def revision_for(repo_id: str) -> str:
    """Immutable revision for a HF repo id (dataset repos keyed by full id)."""
    entry = REVISIONS.get(repo_id)
    if entry is None:
        raise KeyError(
            f"no pinned revision for {repo_id!r}; add it to sources/revisions.py "
            f"(resolve the current commit SHA, never invent one)")
    sha = entry["sha"]
    if sha == PENDING:
        raise KeyError(f"revision for {repo_id!r} is {PENDING}")
    return sha


def source_revision(source_key: str) -> str:
    """Immutable revision for a JxAgent source key (procua, gui360, ...)."""
    return revision_for(source_key)


def unresolved_sources() -> List[str]:
    return [k for k, v in REVISIONS.items() if v["sha"] == PENDING]


def all_resolved() -> bool:
    return not unresolved_sources()


def manifest_view() -> Dict[str, Dict[str, str]]:
    """Copy of the map safe to embed in build manifests."""
    return {k: dict(v) for k, v in REVISIONS.items()}
