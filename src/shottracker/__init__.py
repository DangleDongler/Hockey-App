"""Hockey shot tracking from ordinary video.

Point a phone at a net, take some shots, and get back how fast each one was
and exactly where on the net it landed.

    from shottracker import analyze, Config

    cfg = Config()
    cfg.speed.shot_distance_ft = 20.0
    result = analyze("shots.mp4", cfg)
    print(result.shots[0].speed.mph, result.shots[0].zone_label)
"""

from .config import Config, GoalSpec
from .geometry import GoalPlane, build_zones
from .pipeline import SessionResult, analyze
from .report import format_text_report, shot_chart_svg, summarize
from .shots import Shot

__version__ = "0.1.0"

__all__ = [
    "Config",
    "GoalSpec",
    "GoalPlane",
    "SessionResult",
    "Shot",
    "analyze",
    "build_zones",
    "format_text_report",
    "shot_chart_svg",
    "summarize",
    "__version__",
]
