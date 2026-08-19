"""Reproducibility & safety patch tests (audit 2026-08-16). All offline.

Covers: pinned revisions, config hash stability, ranged-HTTP integrity
(mocked 206/200/429/short/read scenarios), schema-fatality + tolerances,
quota acceptance, corrupt-state warnings, build lock, finalize determinism
and hash ordering, publish allow-list, secret redaction, training-handoff
generator, and the clean-download verifier."""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from processing.remote_access import (RangeIntegrityError, RangedChunkSource,
                                      get_with_retry, verify_ranged_response)
from processing.splitting import assign_splits
from processing.state import (BuildLock, BuildState, contains_secret,
                              redact_secrets)
from processing.validation import (ALL_FATAL_REASONS, SCHEMA_FATAL_REASONS,
                                   compute_quota_acceptance, config_hash_of,
                                   environment_snapshot, finalize,
                                   hash_images_tree, summarize_failures)
from sources.revisions import PENDING, REVISIONS, all_resolved, revision_for
from tests.test_dataset_schema import build_synthetic_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------- revisions

def test_all_source_revisions_are_pinned_commit_shas():
    assert all_resolved()
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for key, entry in REVISIONS.items():
        assert sha_re.match(entry["sha"]), f"{key} sha is not a commit sha"
        assert entry.get("repo")


def test_revision_for_unknown_repo_raises():
    with pytest.raises(KeyError):
        revision_for("some/future-repo")


def test_no_floating_main_in_source_adapters():
    for fn in ("procua.py", "gui360.py", "videocua.py", "groundcua.py",
               "pc_agent_e.py", "replay.py", "../processing/decontamination.py"):
        text = open(os.path.join(ROOT, "sources", fn), encoding="utf-8").read()
        assert "/resolve/main" not in text
        assert "OSWorld/main" not in text


# -------------------------------------------------------------- config hash

def test_config_hash_is_stable_and_sensitive():
    import build_jxagent_dataset as bj
    args = bj.parse_args(["--output", "x"])
    snap1 = bj.build_config_snapshot(args, {"reasoning": {"rate": 0.12}},
                                     dict(bj.SOURCE_TARGETS))
    snap2 = bj.build_config_snapshot(args, {"reasoning": {"rate": 0.12}},
                                     dict(bj.SOURCE_TARGETS))
    assert config_hash_of(snap1) == config_hash_of(snap2)
    snap2["source_targets"]["procua"] += 1
    assert config_hash_of(snap1) != config_hash_of(snap2)
    # revisions are part of the identity
    snap3 = json.loads(json.dumps(snap1))
    snap3["source_revisions"]["procua"]["sha"] = "0" * 40
    assert config_hash_of(snap1) != config_hash_of(snap3)


# ------------------------------------------------- ranged HTTP integrity

class FakeResponse:
    def __init__(self, status_code=206, content=b"", headers=None, url="http://x/y"):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


def _ranged(content=b"0123456789", start=0, end=9, total=100, status=206,
            content_range=True):
    headers = {"Content-Length": str(len(content))}
    if content_range:
        headers["Content-Range"] = f"bytes {start}-{end}/{total}"
    return FakeResponse(status, content, headers)


def test_verify_correct_206_passes():
    verify_ranged_response(_ranged(b"abcd", 0, 3), 0, 3)


def test_verify_short_206_raises():
    with pytest.raises(RangeIntegrityError):
        verify_ranged_response(_ranged(b"abc", 0, 3), 0, 3)  # 3 of 4 bytes


def test_verify_wrong_content_range_raises():
    with pytest.raises(RangeIntegrityError):
        verify_ranged_response(_ranged(b"abcd", 4, 7), 0, 3)  # wrong offset


def test_verify_unexpected_200_raises():
    with pytest.raises(RangeIntegrityError):
        verify_ranged_response(FakeResponse(200, b"whole-file"), 0, 3)


def test_get_with_retry_honors_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    class S:
        def get(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(429, headers={"Retry-After": "7"})
            return FakeResponse(200)

    r = get_with_retry(S(), "http://x")
    assert r.status_code == 200
    assert calls["n"] == 2
    assert 7.0 in sleeps


def test_get_with_retry_survives_connection_resets(monkeypatch):
    import requests as _rq
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class S:
        def get(self, *a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _rq.ConnectionError("reset by peer")
            return FakeResponse(200)

    assert get_with_retry(S(), "http://x").status_code == 200


def test_ranged_chunk_source_rejects_short_body(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    body = b"x" * 16

    class S:
        def head(self, *a, **k):
            return FakeResponse(200, headers={"Content-Length": "16"})

        def get(self, *a, **k):
            # always short: server truncates every chunk
            return _ranged(body[:8], 0, 15, 16)

    with pytest.raises(Exception):
        src = RangedChunkSource("http://x", session=S(), chunk_size=16)
        list(src)
    # offset must NOT have advanced past a bad chunk
    src2 = RangedChunkSource("http://x", session=S(), chunk_size=16)
    try:
        list(src2)
    except Exception:
        pass
    assert src2.offset == 0


def test_ranged_chunk_source_retries_integrity_error_without_advancing(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    data = bytes(range(16))
    calls = {"n": 0}

    class S:
        def head(self, *a, **k):
            return FakeResponse(200, headers={"Content-Length": "16"})

        def get(self, url, headers=None, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                # First CDN response is truncated. The exact same range must
                # be retried rather than advancing the source offset.
                return _ranged(data[:8], 0, 15, 16)
            return _ranged(data, 0, 15, 16)

    src = RangedChunkSource("http://x", session=S(), chunk_size=16)
    assert b"".join(src) == data
    assert calls["n"] == 2
    assert src.offset == 16


def test_ranged_chunk_source_correct_chunks(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    data = bytes(range(32))

    class S:
        def head(self, *a, **k):
            return FakeResponse(200, headers={"Content-Length": "32"})

        def get(self, url, headers=None, **k):
            rng = headers["Range"]
            start, end = (int(x) for x in rng.split("=")[1].split("-"))
            chunk = data[start:end + 1]
            return _ranged(chunk, start, end, 32)

    got = b"".join(RangedChunkSource("http://x", session=S(), chunk_size=8))
    assert got == data


# ---------------------------------------- schema fatality + quota acceptance

@pytest.fixture
def synth(tmp_path):
    return assign_splits(build_synthetic_dataset(tmp_path), 3.0)


def test_schema_failure_is_fatal_by_default(synth, tmp_path):
    victim = next(s for s in synth if s["task_type"] == "action")
    victim["messages"][-1]["content"] = "this is not a valid action!!"
    stats = finalize(str(tmp_path), synth)
    assert stats["failures"].get("invalid_action_syntax", 0) >= 1
    assert stats["fatal_failure"] is True


def test_schema_failure_tolerance_is_explicit_and_recorded(synth, tmp_path):
    victim = next(s for s in synth if s["task_type"] == "action")
    victim["messages"][-1]["content"] = "this is not a valid action!!"
    stats = finalize(str(tmp_path), synth,
                     tolerances={"invalid_action_syntax": 1.0})
    assert stats["fatal_failure"] is False
    tol = stats["tolerated_failures"]["invalid_action_syntax"]
    assert tol["count"] >= 1 and tol["percent"] > 0 and tol["examples"]
    manifest = json.load(open(os.path.join(tmp_path, "final", "manifest.json"),
                              encoding="utf-8"))
    assert manifest["validation_result"]["tolerated_failures"]


def test_all_schema_reasons_are_fatal_capable():
    assert SCHEMA_FATAL_REASONS <= ALL_FATAL_REASONS


def test_quota_acceptance_rules():
    qa = compute_quota_acceptance({"procua": 45999}, {"procua": 46000})
    assert qa["procua"]["accepted"] is True  # 99.998% >= 99%
    qa = compute_quota_acceptance({"procua": 40000}, {"procua": 46000})
    assert qa["procua"]["accepted"] is False
    assert qa["procua"]["realization_pct"] == 86.96
    # documented PC-Agent-E exception (~4.3k of 4503)
    qa = compute_quota_acceptance({"pcagente": 4300}, {"pcagente": 4503})
    assert qa["pcagente"]["accepted"] is True
    qa = compute_quota_acceptance({"pcagente": 3000}, {"pcagente": 4503})
    assert qa["pcagente"]["accepted"] is False
    # a source that raised is never accepted, whatever it selected
    qa = compute_quota_acceptance({"procua": 46000}, {"procua": 46000},
                                  source_errors={"procua": "429 exhaustion"})
    assert qa["procua"]["accepted"] is False
    assert qa["procua"]["source_error"] is True


def test_finalize_reports_quota_acceptance(synth, tmp_path):
    stats = finalize(str(tmp_path), synth, targets={"pcagente": 10 ** 9})
    assert stats["quota_acceptance"]["pcagente"]["accepted"] is False
    assert stats["quota_acceptance_passed"] is False
    # separate from fatal_failure: schema/validity still clean
    assert stats["fatal_failure"] is False


# ------------------------------------------- corrupt state + build lock

def test_corrupt_dedup_index_warns_and_records(tmp_path, capsys):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "dedup_index.json").write_text("{corrupt json!!", encoding="utf-8")
    st = BuildState(str(state_dir))
    idx = st.load_dedup_index()  # lazy load is where corruption is detected
    assert idx is not None  # recovers with an empty index rather than crashing
    assert any("dedup_index.json" in e["path"] for e in st.corruption_events)
    assert st.corruption_summary()["state_corruption_detected"] is True
    assert "STATE CORRUPTION" in capsys.readouterr().err


def test_truncated_selected_samples_line_is_skipped(tmp_path):
    st = BuildState(str(tmp_path / "state"))
    st.append_jsonl("selected_samples.jsonl", [{"a": 1}])
    with open(os.path.join(st.state_dir, "selected_samples.jsonl"), "a",
              encoding="utf-8") as f:
        f.write('{"a": 2, "trun')  # crash mid-write
    rows = st.read_jsonl("selected_samples.jsonl")
    assert rows == [{"a": 1}]


def test_build_lock_blocks_second_builder(tmp_path, monkeypatch):
    lock_path = tmp_path / "state" / "build.lock"
    lock_path.parent.mkdir()
    lock_path.write_text(json.dumps({"pid": 424242, "hostname": "other",
                                     "started": "now"}), encoding="utf-8")
    monkeypatch.setattr("processing.state._pid_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="another builder"):
        BuildLock(str(tmp_path / "state")).acquire()


def test_build_lock_recovers_stale_dead_pid(tmp_path, monkeypatch):
    lock_path = tmp_path / "state" / "build.lock"
    lock_path.parent.mkdir()
    lock_path.write_text(json.dumps({"pid": 999999999, "hostname": "crashed",
                                     "started": "yesterday"}), encoding="utf-8")
    monkeypatch.setattr("processing.state._pid_alive", lambda pid: False)
    lock = BuildLock(str(tmp_path / "state"))
    lock.acquire()
    assert lock.held
    lock.release()
    assert not lock_path.exists()


def test_build_lock_force_clear(tmp_path, monkeypatch):
    lock_path = tmp_path / "state" / "build.lock"
    lock_path.parent.mkdir()
    lock_path.write_text(json.dumps({"pid": 424242, "hostname": "hung",
                                     "started": "now"}), encoding="utf-8")
    monkeypatch.setattr("processing.state._pid_alive", lambda pid: True)
    BuildLock(str(tmp_path / "state")).acquire(force_clear=True)


def test_persist_before_mark_crash_window(tmp_path):
    """Crash AFTER persist_samples but BEFORE mark_shard_done: samples are
    durable; a rerun re-processes the unit and the merge dedups the overlap."""
    st = BuildState(str(tmp_path / "state"))
    samples = [{"source": "pcagente", "trajectory_id": "t1", "step_id": "s1",
                "messages": []}]
    st.append_jsonl("selected_samples.jsonl", samples)   # durable
    # (crash happens here — shard NOT marked done)
    st.append_jsonl("selected_samples.jsonl", samples)   # rerun re-emits
    prior = st.read_jsonl("selected_samples.jsonl")
    seen, merged = set(), []
    for s in prior:  # exactly main()'s merge logic
        key = (s["source"], s["trajectory_id"], s["step_id"])
        if key not in seen:
            merged.append(s)
            seen.add(key)
    assert len(merged) == 1


# -------------------------------------- finalize determinism + hash ordering

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_finalize_twice_identical_train_bytes(synth, tmp_path):
    root = str(tmp_path)
    s1 = finalize(root, synth, targets={"pcagente": 10 ** 9})
    h1 = _sha(os.path.join(root, "final", "train.jsonl"))
    t1 = s1["images_tree_hash"]
    s2 = finalize(root, synth, targets={"pcagente": 10 ** 9})
    assert h1 == _sha(os.path.join(root, "final", "train.jsonl"))
    assert t1 == s2["images_tree_hash"]


def test_sha256sums_and_manifest_hash_ordering(synth, tmp_path):
    root = str(tmp_path)
    stats = finalize(root, synth, build_identity={
        "build_id": "b" * 16, "builder_commit": "c" * 40,
        "config_hash": "d" * 64, "config_snapshot": {"x": 1},
        "environment": environment_snapshot(),
        "source_revisions": {}, "started_at": "t0", "finished_at": "t1",
        "selection_policy": "hash", "decontamination": {},
        "state_corruption": {"state_corruption_detected": False, "events": []},
        "failures_rows": [{"source": "p", "error": "e", "at": "t"}],
    })
    final = os.path.join(root, "final")
    for name in ("build_config.json", "environment.json",
                 "build_failures_summary.json", "SHA256SUMS"):
        assert os.path.exists(os.path.join(final, name)), name
    manifest = json.load(open(os.path.join(final, "manifest.json"), encoding="utf-8"))
    # manifest knows every metadata file except itself (no circular hash)
    assert "final/manifest.json" not in manifest["hashes"]
    for rel, digest in manifest["hashes"].items():
        assert _sha(os.path.join(root, rel.replace("/", os.sep))) == digest
    # SHA256SUMS covers everything INCLUDING manifest
    sums = {}
    for line in open(os.path.join(final, "SHA256SUMS"), encoding="utf-8"):
        if line.strip():
            d, rel = line.split(None, 1)
            sums[rel.strip()] = d
    assert "final/manifest.json" in sums
    for rel, digest in sums.items():
        assert _sha(os.path.join(root, rel.replace("/", os.sep))) == digest
    assert stats["images_tree_hash"] == hash_images_tree(root)["images_tree_hash"]


# ------------------------------------------------ publish allow-list + secrets

def _load_publish():
    spec = importlib.util.spec_from_file_location(
        "publish_dataset", os.path.join(ROOT, "publish_dataset.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_publish_allow_list_excludes_everything_unintended(tmp_path):
    pub = _load_publish()
    root = tmp_path
    for sub in ("final", "images/procua", "manifests", "state", ".tmp",
                ".venv/Lib", "rlvr", "images/__pycache__"):
        os.makedirs(root / sub, exist_ok=True)
    (root / "final/train.jsonl").write_text("{}", encoding="utf-8")
    (root / "final/.secret.tmp").write_text("x", encoding="utf-8")
    (root / "images/procua/a.webp").write_bytes(b"webp")
    (root / "images/procua/b.part").write_bytes(b"part")
    (root / "images/__pycache__/x.pyc").write_bytes(b"pyc")
    (root / "manifests/procua.json").write_text("{}", encoding="utf-8")
    (root / "state/failures.jsonl").write_text("secret-ish", encoding="utf-8")
    (root / "state/build.lock").write_text("{}", encoding="utf-8")
    (root / "SmokeInside").mkdir()
    (root / "SmokeInside/x.json").write_text("{}", encoding="utf-8")

    files = pub.select_publication_files(str(root))
    assert files == ["final/train.jsonl", "images/procua/a.webp",
                     "manifests/procua.json"]


def test_publish_dry_run_uploads_nothing(tmp_path, capsys):
    pub = _load_publish()
    root = tmp_path
    os.makedirs(root / "final")
    for f in ("train.jsonl", "validation.jsonl", "manifest.json", "SHA256SUMS"):
        (root / "final" / f).write_text("{}" if f.endswith("json") else "\n",
                                        encoding="utf-8")
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "publish_dataset.py"),
         "--dataset-root", str(root), "--repo", "u/r", "--dry-run"],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "selected 4 files" in out.stdout
    assert "dry run" in out.stdout


def test_redact_secrets():
    tok = "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    text = f"failed with token {tok} in header Authorization: Bearer xyz123"
    red = redact_secrets(text)
    assert tok not in red and "Bearer" not in red
    assert contains_secret(text) and not contains_secret(red)


def test_generated_artifacts_contain_no_credentials(synth, tmp_path):
    tok = "hf_" + "Z9y8X7w6V5u4T3s2R1q0P9o8"
    root = str(tmp_path)
    finalize(root, synth, build_identity={
        "build_id": "b" * 16, "builder_commit": "c" * 40,
        "config_hash": "d" * 64, "config_snapshot": {},
        "environment": {}, "source_revisions": {}, "started_at": "t",
        "finished_at": "t", "selection_policy": "hash", "decontamination": {},
        "state_corruption": {"state_corruption_detected": False, "events": []},
        "failures_rows": [{"source": "x", "error": f"403 with {tok}", "at": "t"}],
    })
    pat = re.compile(rb"hf_[A-Za-z0-9]{20,}")
    for dirpath, _d, files in os.walk(os.path.join(root, "final")):
        for fn in files:
            data = open(os.path.join(dirpath, fn), "rb").read()
            assert not pat.search(data), f"credential-like value in {fn}"
    summary = json.load(open(os.path.join(root, "final",
                                          "build_failures_summary.json"),
                             encoding="utf-8"))
    assert summary["failure_classes"][0]["count"] == 1


# ------------------------------------------------ training handoff generator

def _finalize_clean(tmp_path):
    # Keep the copied verification target outside the source tree. Copying a
    # directory into one of its own children can recurse indefinitely on fast
    # filesystems and made the clean-download tests nondeterministically hang.
    source = tmp_path / "built"
    source.mkdir()
    samples = assign_splits(build_synthetic_dataset(source), 3.0)
    finalize(str(source), samples)
    return str(source)


def test_handoff_generator_roundtrip(tmp_path):
    root = _finalize_clean(tmp_path)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_training_handoff as mth
    h = mth.build_handoff(root, "user/JxAgentData", "abc123def456")
    for key in ("dataset_repo", "dataset_revision", "builder_git_commit",
                "config_hash", "files", "hashes", "counts", "validation_status",
                "model_id", "estimated_tokens", "estimated_optimizer_steps",
                "policies"):
        assert key in h
    assert h["model_id"] == "Qwen/Qwen3.8-27B"
    assert h["hashes"]["train"] and h["hashes"]["images_tree_hash"]
    md = mth.render_md(h)
    assert "user/JxAgentData" in md and "TRAINING HANDOFF" in md


def test_handoff_refuses_tampered_dataset(tmp_path):
    root = _finalize_clean(tmp_path)
    train = os.path.join(root, "final", "train.jsonl")
    with open(train, "a", encoding="utf-8") as f:
        f.write("{}\n")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_training_handoff as mth
    with pytest.raises(SystemExit, match="hash mismatch"):
        mth.build_handoff(root, "user/JxAgentData", "rev")


def test_handoff_requires_repo_and_revision(tmp_path):
    root = _finalize_clean(tmp_path)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import make_training_handoff as mth
    with pytest.raises(SystemExit, match="dataset-repo"):
        mth.build_handoff(root, "", "")


# ------------------------------------------------ clean download verifier

def _run_verifier(root, extra=()):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "verify_clean_download.py"),
         "--dataset-root", str(root), *extra],
        capture_output=True, text=True)


def test_clean_download_verifier_passes_on_intact_copy(tmp_path):
    root = _finalize_clean(tmp_path)
    fresh = tmp_path / "fresh_download"
    shutil.copytree(root, fresh)
    out = _run_verifier(fresh)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASSED" in out.stdout


def test_clean_download_verifier_accepts_zero_count_target_sources(tmp_path):
    # finalize() records every TARGET source in manifest.sources; a target
    # that contributed 0 samples (e.g. a partial-source build) appears as a
    # zero entry while the JSONLs can only contain sources with samples.
    # The verifier must not fail an intact copy because of those zeros.
    source = tmp_path / "built_zero"
    source.mkdir()
    samples = assign_splits(build_synthetic_dataset(source), 3.0)
    finalize(str(source), samples,
             targets={"pcagente": 12, "gui360": 12, "procua": 46000})
    manifest = json.load(open(source / "final" / "manifest.json",
                              encoding="utf-8"))
    assert manifest["sources"].get("procua") == 0
    fresh = tmp_path / "fresh_download"
    shutil.copytree(source, fresh)
    out = _run_verifier(fresh)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "PASSED" in out.stdout


def test_clean_download_verifier_catches_tampering(tmp_path):
    root = _finalize_clean(tmp_path)
    fresh = tmp_path / "fresh_download"
    shutil.copytree(root, fresh)
    # tamper one image byte: tree hash must mismatch
    imgs = sorted((fresh / "images").rglob("*.webp"))
    if not imgs:
        imgs = sorted((fresh / "images").rglob("*"))
    imgs[0].write_bytes(imgs[0].read_bytes() + b"\x00")
    out = _run_verifier(fresh)
    assert out.returncode == 1
    assert "images_tree_hash" in out.stdout or "FAIL" in out.stdout
