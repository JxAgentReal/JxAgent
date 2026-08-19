"""Streaming tar.zst reader must use streaming decompression (tar.zst frames
carry no embedded content size) and must be stoppable mid-stream."""
import io
import tarfile

import pytest

zstandard = pytest.importorskip("zstandard")

from processing.remote_access import ZstHttpResponseReader


def make_tar_zst_bytes(n_files=5) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_files):
            data = f"trajectory {i}".encode()
            info = tarfile.TarInfo(f"part_1/run/traj_{i}/trajectory.json")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    out = io.BytesIO()
    with zstandard.ZstdCompressor().stream_writer(out, closefd=False) as w:
        w.write(raw)
    return out.getvalue()


def chunked(data: bytes, size: int = 7):
    return iter([data[i:i + size] for i in range(0, len(data), size)])


def test_stream_reader_reads_whole_archive():
    reader = ZstHttpResponseReader(chunked(make_tar_zst_bytes()))
    tf = tarfile.open(fileobj=reader, mode="r|")
    names = [m.name for m in tf if m.isfile()]
    tf.close()
    assert len(names) == 5
    assert names[0] == "part_1/run/traj_0/trajectory.json"


def test_stream_reader_small_chunks_and_content():
    reader = ZstHttpResponseReader(chunked(make_tar_zst_bytes(3), size=3))
    tf = tarfile.open(fileobj=reader, mode="r|")
    contents = []
    for m in tf:
        if m.isfile():
            contents.append(tf.extractfile(m).read().decode())
    tf.close()
    assert contents == ["trajectory 0", "trajectory 1", "trajectory 2"]


def test_stream_reader_stoppable_mid_stream():
    reader = ZstHttpResponseReader(chunked(make_tar_zst_bytes(50)))
    tf = tarfile.open(fileobj=reader, mode="r|")
    seen = 0
    for _ in tf:
        seen += 1
        if seen == 2:
            break  # smoke builds stop consuming exactly like this
    tf.close()
    assert seen == 2


def test_one_shot_decompress_would_fail_on_this_frame():
    """Documents why the streaming decompressobj is mandatory."""
    data = make_tar_zst_bytes()
    with pytest.raises(zstandard.ZstdError):
        zstandard.ZstdDecompressor().decompress(data)
