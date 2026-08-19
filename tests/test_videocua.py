import io
import json

import pytest

def _has_av():
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False

from sources.videocua import build_trajectory, decode_frames
from tests.conftest import make_png_bytes


def action_log(n=4, gap_wait=False):
    entries = []
    t = 1.0
    for i in range(n):
        entries.append({"action_type": "CLICK", "timestamp": round(t, 3),
                        "action_params": {"x": 100 + i * 40, "y": 200, "numClicks": 1}})
        t += 8.0 if (gap_wait and i == 1) else 0.8
    entries.append({"action_type": "TERMINATE_SUCCESS", "timestamp": round(t, 3),
                    "action_params": {}})
    return {"task_id": 45525, "task_instruction": "Open test.7z and extract it",
            "platform": "7-Zip", "action_log": entries}


def test_build_trajectory_from_frames():
    log = action_log(n=4)
    stamps = [a["timestamp"] for a in log["action_log"]]
    frames = {ts: make_png_bytes(1920, 1080, marker=i) for i, ts in enumerate(stamps)}
    traj = build_trajectory(log, frames, (1920, 1080))
    assert traj is not None
    assert len(traj.steps) == 5
    assert traj.steps[-1].action.verb == "finish"
    assert traj.app == "7-zip"
    assert traj.trajectory_id == "videocua_7-Zip_45525"


def test_wait_signal_from_timestamp_gap():
    log = action_log(n=4, gap_wait=True)
    stamps = [a["timestamp"] for a in log["action_log"]]
    frames = {ts: make_png_bytes(1920, 1080, marker=i) for i, ts in enumerate(stamps)}
    traj = build_trajectory(log, frames, (1920, 1080))
    gaps = [s for s in traj.steps if "wait" in s.signals]
    assert gaps, "a >=3s gap must produce a wait signal on the following step"


def test_no_state_change_signal():
    log = action_log(n=3)
    stamps = [a["timestamp"] for a in log["action_log"]]
    frames = {ts: make_png_bytes(1920, 1080, marker=1) for ts in stamps}  # identical frames
    traj = build_trajectory(log, frames, (1920, 1080))
    assert any("no_state_change" in s.signals for s in traj.steps[1:])


def test_missing_instruction_drops_task():
    log = action_log()
    log["task_instruction"] = ""
    traj = build_trajectory(log, {}, (1920, 1080))
    assert traj is None


@pytest.mark.skipif(not _has_av(), reason="PyAV not installed")
def test_decode_frames_persistent_decoder():
    av = pytest.importorskip("av")
    # encode 30 frames at 30fps with changing content
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    stream = container.add_stream("mpeg4", rate=30)
    stream.width, stream.height = 320, 240
    stream.pix_fmt = "yuv420p"
    from PIL import Image
    for i in range(30):
        img = Image.new("RGB", (320, 240), (i * 8 % 255, 0, 0))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    video_bytes = buf.getvalue()
    # decode_frames opens the video once and returns the wanted timestamps
    frames = decode_frames(video_bytes, [0.0, 0.5, 0.9])
    assert 0.0 in frames
    assert any(abs(t - 0.5) < 1e-6 for t in frames)

