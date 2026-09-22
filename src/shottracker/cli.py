"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .config import CameraConfig, Config
from .pipeline import analyze
from .report import format_text_report, shot_chart_svg


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
    return cfg


def cmd_analyze(args) -> int:
    cfg = _build_config(args)

    def progress(stage: str, frac: float) -> None:
        if not args.quiet:
            print(f"\r  {stage:<22} {frac*100:5.1f}%", end="", file=sys.stderr, flush=True)

    result = analyze(args.video, cfg, net_quad=args.net, progress=None if args.quiet else progress)
    if not args.quiet:
        print("\r" + " " * 40 + "\r", end="", file=sys.stderr)

    print(format_text_report(result))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\nwrote {args.json}")
    if args.chart:
        with open(args.chart, "w") as fh:
            fh.write(shot_chart_svg(result))
        print(f"wrote {args.chart}")
    if args.overlay:
        from .overlay import render_overlay

        render_overlay(result, args.overlay, cfg=cfg)
        print(f"wrote {args.overlay}")

    return 0 if result.net is not None else 2


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shottracker",
        description="Track hockey shots: find the net, follow the puck, report speed and accuracy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyze a video of someone shooting")
    a.add_argument("video")
    a.add_argument("--distance", type=float, metavar="FT",
                   help="distance from the shooting spot to the goal line, in feet. "
                        "The single number that most improves speed accuracy.")
    a.add_argument("--offset", type=float, metavar="FT",
                   help="how far the shooter stands to the side of net centre, in feet "
                        "(negative = the shooter's left)")
    a.add_argument("--fps", type=float, help="override the clip's frame rate (slow-motion files often lie)")
    a.add_argument("--speed-method", choices=["auto", "time_of_flight", "ballistic_3d", "goal_plane"],
                   default="auto")
    a.add_argument("--hfov", type=float, metavar="DEG",
                   help="your camera's horizontal field of view in degrees. Only used when the "
                        "view is too square-on for the goal to reveal it; phones are 60-70.")
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

    b = sub.add_parser("benchmark", help="measure accuracy against synthetic clips with known answers")
    b.add_argument("--cameras", nargs="+", default=["side", "angled", "head_on"],
                   choices=["side", "angled", "head_on", "extreme_side"])
    b.add_argument("--keep", metavar="DIR", help="keep the rendered clips in this directory")
    b.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
