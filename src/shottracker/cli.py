"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .config import CameraConfig, Config
from .pipeline import analyze
from .report import format_session_report, format_text_report, shot_chart_svg
from .session import ShotSession
from .targets import TARGET_CHOICES


def _parse_quad(text: str) -> np.ndarray:
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if len(vals) != 8:
        raise argparse.ArgumentTypeError("--net needs 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4")
    return np.array(vals, dtype=float).reshape(4, 2)


def _build_config(args) -> Config:
    cfg = Config()
    if args.distance is not None:
        cfg.speed.shot_distance_ft = args.distance
    if args.offset is not None:
        cfg.speed.shooter_offset_ft = args.offset
    if args.fps is not None:
        cfg.fps_override = args.fps
    if args.speed_method:
        cfg.speed.method = args.speed_method
    if getattr(args, "hfov", None):
        cfg.camera.assumed_focal_frac = CameraConfig.frac_from_hfov(args.hfov)
    if getattr(args, "target", None):
        cfg.target.kind = args.target
    if getattr(args, "target_at", None):
        cfg.target.kind = "custom"
        cfg.target.x_in, cfg.target.y_in = args.target_at
    if getattr(args, "target_radius", None):
        cfg.target.radius_in = args.target_radius
    return cfg


def _parse_point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("--target-at needs X,Y in inches, e.g. -30,42")
    return x, y


def cmd_analyze(args) -> int:
    cfg = _build_config(args)
    videos = list(args.video)
    if args.net is not None and len(videos) > 1:
        print("--net describes one clip's view; give it with a single video", file=sys.stderr)
        return 2

    named = []
    for i, path in enumerate(videos):
        def progress(stage: str, frac: float, i=i) -> None:
            if not args.quiet:
                where = f"[{i + 1}/{len(videos)}] " if len(videos) > 1 else ""
                print(f"\r  {where}{stage:<22} {frac*100:5.1f}%", end="", file=sys.stderr, flush=True)

        # One Config for every clip: they share the goal and the target.
        result = analyze(path, cfg, net_quad=args.net, progress=None if args.quiet else progress)
        named.append((os.path.basename(path), result))
    if not args.quiet:
        print("\r" + " " * 50 + "\r", end="", file=sys.stderr)

    session = ShotSession.from_results(named, cfg)
    print(format_session_report(session))

    single = named[0][1] if len(named) == 1 else None
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(single.to_dict() if single else session.to_dict(), fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.chart:
        with open(args.chart, "w") as fh:
            fh.write(shot_chart_svg(session))
        print(f"wrote {args.chart}")
    if args.overlay:
        from .overlay import render_overlay

        stem, ext = os.path.splitext(args.overlay)
        for i, (_, result) in enumerate(named):
            out = args.overlay if single else f"{stem}-{i + 1}{ext or '.mp4'}"
            render_overlay(result, out, cfg=cfg)
            print(f"wrote {out}")

    return 0 if all(r.net is not None for _, r in named) else 2


def cmd_demo(args) -> int:
    """Render a synthetic clip with known answers and analyze it."""
    from .synth import SynthShot, render_session

    os.makedirs(args.out, exist_ok=True)
    video = os.path.join(args.out, "demo.mp4")
    shots = [
        SynthShot(target_x_in=-30, target_y_in=42, flight_time_s=0.20, start_time_s=0.30),
        SynthShot(target_x_in=30, target_y_in=6, flight_time_s=0.26, start_time_s=0.95),
        SynthShot(target_x_in=0, target_y_in=8, flight_time_s=0.30, start_time_s=1.60),
        SynthShot(target_x_in=48, target_y_in=30, flight_time_s=0.24, start_time_s=2.25),
    ]
    truth = render_session(video, shots, camera=args.camera, fps=args.fps)
    print(f"rendered {video} ({truth['frames']} frames, {args.camera} camera)")
    print("ground truth:")
    for i, s in enumerate(truth["shots"]):
        print(f"  shot {i+1}: {s['release_speed_mph']:5.1f} mph at "
              f"({s['impact_x_in']:+.0f}\", {s['impact_y_in']:.0f}\")")
    print()

    cfg = Config()
    cfg.speed.shot_distance_ft = 20.0
    result = analyze(video, cfg)
    print(format_text_report(result))

    chart = os.path.join(args.out, "demo-chart.svg")
    with open(chart, "w") as fh:
        fh.write(shot_chart_svg(result))
    with open(os.path.join(args.out, "demo.json"), "w") as fh:
        json.dump({"truth": truth, "measured": result.to_dict()}, fh, indent=2)
    print(f"\nwrote {chart} and {os.path.join(args.out, 'demo.json')}")

    if args.overlay:
        from .overlay import render_overlay

        path = os.path.join(args.out, "demo-overlay.mp4")
        render_overlay(result, path, cfg=cfg)
        print(f"wrote {path}")
    return 0


def cmd_benchmark(args) -> int:
    """Grade the tracker against clips whose answers are known exactly."""
    from .benchmark import format_table, run

    print("rendering and analyzing synthetic clips (this takes a minute)...\n")
    scores = run(cameras=args.cameras, workdir=args.keep)
    print(format_table(scores))
    print(
        "\non-net / off-net are mean impact errors in inches on a 72x48 in goal mouth; "
        "\nspeed errors assume the shooting distance was given."
    )
    return 0


def cmd_fullspeed(args) -> int:
    """How much would be lost filming this shot at normal speed?"""
    from .fullspeed import compare, format_table

    cfg = _build_config(args)
    print(format_table(compare(args.video, cfg, rates=tuple(args.rates))))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shottracker",
        description="Track hockey shots: find the net, follow the puck, report speed and accuracy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze one or more videos of someone shooting")
    a.add_argument("video", nargs="+",
                   help="one clip, or several of the same goal to read as one session "
                        "(slow motion often comes one shot per clip)")
    a.add_argument("--distance", type=float, metavar="FT",
                   help="distance from the shooting spot to the goal line, in feet. "
                        "The single number that most improves speed accuracy.")
    a.add_argument("--offset", type=float, metavar="FT",
                   help="how far the shooter stands to the side of net centre, in feet "
                        "(negative = the shooter's left)")
    a.add_argument("--fps", type=float, help="override the frame rate the clip was filmed at. Slow motion with the "
                   "slowdown baked in is usually detected from the sound; use this when it is not")
    a.add_argument("--speed-method", choices=["auto", "time_of_flight", "ballistic_3d", "goal_plane"],
                   default="auto")
    a.add_argument("--hfov", type=float, metavar="DEG",
                   help="your camera's horizontal field of view in degrees. Only used when the "
                        "view is too square-on for the goal to reveal it; phones are 60-70.")
    a.add_argument("--target", choices=TARGET_CHOICES,
                   help="what the player was aiming at; each shot is scored against it. "
                        "Groups (top_shelf, any_corner, low_corners) score each shot against "
                        "whichever member it was nearest.")
    a.add_argument("--target-at", type=_parse_point, metavar="X,Y",
                   help="a custom aim point in goal inches: x from net centre (right is +), "
                        "y up from the ice")
    a.add_argument("--target-radius", type=float, metavar="IN",
                   help="hit radius around the aim point, in inches (default 6)")
    a.add_argument("--net", type=_parse_quad, metavar="X1,Y1,...",
                   help="skip net detection and use these four corners (top-left, top-right, "
                        "bottom-right, bottom-left) of the outer pipe")
    a.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    a.add_argument("--chart", metavar="PATH", help="write the shot chart as SVG")
    a.add_argument("--overlay", metavar="PATH", help="write an annotated copy of the video")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=cmd_analyze)

    d = sub.add_parser("demo", help="render a synthetic clip with known answers and analyze it")
    d.add_argument("--out", default="out", metavar="DIR")
    d.add_argument("--camera", default="angled", choices=["side", "angled", "head_on", "extreme_side"])
    d.add_argument("--fps", type=float, default=120.0)
    d.add_argument("--overlay", action="store_true", help="also render the annotated video")
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fullspeed", help="read a slow-motion shot as if filmed at normal speed, and compare")
    f.add_argument("video", help="a slow-motion clip of one shot")
    f.add_argument("--distance", type=float, metavar="FT", help="shooting distance, in feet")
    f.add_argument("--offset", type=float, metavar="FT", help="shooter's offset from net centre, in feet")
    f.add_argument("--fps", type=float, help="the rate the clip was filmed at, if not detected")
    f.add_argument("--rates", type=float, nargs="+", default=[30, 60], help="normal-speed rates to try")
    f.add_argument("--speed-method", default=None)
    f.set_defaults(func=cmd_fullspeed)

    b = sub.add_parser("benchmark", help="measure accuracy against synthetic clips with known answers")
    b.add_argument("--cameras", nargs="+", default=["side", "angled", "head_on"],
                   choices=["side", "angled", "head_on", "extreme_side"])
    b.add_argument("--keep", metavar="DIR", help="keep the rendered clips in this directory")
    b.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
