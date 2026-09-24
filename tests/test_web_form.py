"""The upload form must take the numbers people actually measure."""

import re
from pathlib import Path

HTML = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text()


def test_every_number_field_takes_any_decimal():
    # A step of 0.5 made the browser refuse 18.75 ft -- the real shooting
    # distance of the first real clip -- and silently block the upload.
    fields = re.findall(r"<input[^>]*type=\"number\"[^>]*>", HTML)
    assert fields
    for f in fields:
        assert 'step="any"' in f, f
