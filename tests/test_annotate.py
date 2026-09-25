"""A copy of a clip with its shots drawn on (shottracker.annotate)."""

import json

import av
import numpy as np

from shottracker.annotate import Painter, render
from shottracker.video import iter_frames


def _clip_dict(result):
    # The result as the web player gets it: through JSON.
    return json.loads(json.dumps(result.to_dict()))


def _rate(clip):
    return clip["video"].get("playback_fps") or clip["video"]["fps"]


def test_the_marked_copy_is_h264_with_every_frame(tmp_path, angled_result):
    truth, result = angled_result
    clip = _clip_dict(result)
    out = str(tmp_path / "marked.mp4")
    seen = []
    render(truth["path"], clip, out, progress=seen.append)

    with av.open(out) as c:
        s = c.streams.video[0]
        assert s.codec_context.name == "h264"
        # No wider than 1080, shape kept, even sides for H.264.
        w, h = clip["video"]["width"], clip["video"]["height"]
        assert s.width == min(w, 1080) and s.width % 2 == 0 and s.height % 2 == 0
        assert abs(s.height - h * s.width / w) <= 1
        assert abs(float(s.average_rate) - _rate(clip)) < 0.01
        n = sum(1 for _ in c.decode(s))
    assert n == clip["video"]["frame_count"]
    assert seen and seen == sorted(seen) and seen[-1] <= 1.0


def test_each_shot_is_marked_where_it_hit_once_it_has_hit(angled_result):
    truth, result = angled_result
    clip = _clip_dict(result)
    shots = clip["shots"]
    assert shots
    size = (clip["video"]["width"], clip["video"]["height"])
    painter = Painter(clip, 1.0, _rate(clip))
    wanted = {s["impact_frame"] + 2: s for s in shots}
    for i, frame in enumerate(iter_frames(truth["path"], size)):
        if i not in wanted:
            continue
        shot = wanted[i]
        drawn = painter.draw(frame.copy(), i)
        x, y = (int(round(v)) for v in shot["impact_image"])
        patch = np.s_[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        changed = np.abs(drawn[patch].astype(int) - frame[patch].astype(int)).sum(axis=2).mean()
        assert changed > 60, f"shot {shot['index']} not marked where it hit"


def test_labels_carry_the_number_result_and_speed():
    from shottracker.annotate import _label_parts

    assert _label_parts(3, "on_net", 47.4, True) == ["#3", "tick", "47 mph"]
    assert _label_parts(3, "miss", None, True) == ["#3", "cross"]
    assert _label_parts(3, "post", 51.0, True) == ["#3", "post", "51 mph"]
    assert _label_parts(3, "on_net", 47.4, False) == ["#3"]
