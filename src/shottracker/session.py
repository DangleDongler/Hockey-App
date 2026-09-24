"""Several clips of the same goal, read as one shooting session.

Slow motion is recorded in short bursts -- often one shot per clip -- so a
session of ten shots can arrive as ten files.  Each clip is analyzed on its
own, with its own goal outline and frame rate, because the phone may have
been nudged between them.  What the clips share is the goal itself: every
impact is already in inches on the same net, so the shots simply pool into one
chart, one set of statistics, and one score against the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import Config
from .pipeline import SessionResult
from .shots import Shot


@dataclass
class Clip:
    name: str
    result: SessionResult


@dataclass
class ShotSession:
    clips: list[Clip]
    # The goal and target every clip was scored against.  Shared, so that
    # re-scoring against a different target updates the whole session.
    config: Config = field(default_factory=Config)

    @classmethod
    def from_results(cls, named: list[tuple[str, SessionResult]], config: Config | None = None) -> "ShotSession":
        if not named:
            raise ValueError("a session needs at least one clip")
        return cls([Clip(n, r) for n, r in named], config or named[0][1].config)

    @property
    def placed(self) -> list[tuple[int, Shot]]:
        """Every shot with the clip it came from, in the order they were taken."""
        return [(ci, s) for ci, c in enumerate(self.clips) for s in c.result.shots]

    @property
    def shots(self) -> list[Shot]:
        """All shots, numbered through the session rather than within each clip."""
        return [replace(s, index=i) for i, (_, s) in enumerate(self.placed)]

    @property
    def warnings(self) -> list[str]:
        """Each clip's notes, said once however many clips share them."""
        if len(self.clips) == 1:
            return list(self.clips[0].result.warnings)
        by_note: dict[str, list[str]] = {}
        for c in self.clips:
            for w in c.result.warnings:
                names = by_note.setdefault(w, [])
                if c.name not in names:
                    names.append(c.name)
        out = []
        for w, names in by_note.items():
            who = "every clip" if len(names) == len(self.clips) else ", ".join(names)
            out.append(f"{who}: {w}")
        return out

    def targeting(self) -> dict | None:
        from .targets import targeting_block

        # No image outline: each clip sees the goal from its own position.
        return targeting_block(
            [(s.impact_goal_in[0], s.impact_goal_in[1], s.outcome) for s in self.shots],
            self.config.goal,
            self.config.target,
        )

    def to_dict(self) -> dict:
        from .report import summarize

        targeting = self.targeting()
        shots = []
        for i, ((ci, s), sd) in enumerate(zip(self.placed, (s.to_dict() for s in self.shots))):
            sd["clip"] = ci
            sd["clip_shot"] = s.index
            if targeting:
                sd["vs_target"] = targeting["per_shot"][i]
            shots.append(sd)
        return {
            "clips": [dict(c.result.to_dict(), name=c.name) for c in self.clips],
            "shots": shots,
            "targeting": targeting,
            "goal": {
                "mouth_width_in": self.config.goal.mouth_width_in,
                "mouth_height_in": self.config.goal.mouth_height_in,
                "post_diameter_in": self.config.goal.post_diameter_in,
                "corner_radius_in": self.config.goal.corner_radius_in,
            },
            "summary": summarize(self),
            "warnings": self.warnings,
            "elapsed_s": round(sum(c.result.elapsed_s for c in self.clips), 2),
        }
