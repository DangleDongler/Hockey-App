# Shot Tracker

Film yourself shooting at a net. Get back how hard each shot was and exactly
where on the net it landed.

The tracker finds the goal by itself, follows the puck through the air, and
marks every impact on a diagram of the net, so a session ends with a picture of
where your shots actually went — which is usually not where you thought.

```
Clip      1280x720 @ 120.0 fps (container), 353 frames
Net       found by pipe_edges, confidence 99% (24/24 frames agreed)
Camera    25.6 ft out, +14.5 ft across, 4.2 ft up (reprojection 1.09 px)

Shots     4   on net 3   posts 0   missed 1
Accuracy  75% on net
Speed     max 70 mph, average 57 mph
Grouping  centred +11", 21" up, spread 32"
Corners   nearest-corner distance: best 9", average 19"

  #   frame    speed          where
  1   60       70.1 mph ±4.1  Top Shelf Left
  2   146      53.3 mph ±3.1  Bottom Right Corner
  3   228      45.7 mph ±2.7  Five Hole
  4   298      60.1 mph ±3.5  missed - 0.7 ft wide right
```

## Getting started

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[server,dev]"
```

See it work on a clip with known answers:

```bash
.venv/bin/shottracker demo --out out --overlay
```

Analyze your own footage:

```bash
.venv/bin/shottracker analyze shots.mp4 --distance 20 \
    --chart chart.svg --overlay annotated.mp4 --json result.json
```

Or run the web app and drag a clip in:

```bash
PYTHONPATH=src:server .venv/bin/python -m uvicorn app:app --reload
# open http://127.0.0.1:8000
```

The browser plays your original clip and draws the analysis over it on a
canvas — nothing is re-encoded, so playback stays smooth and you can scrub to
any shot by clicking its mark on the chart.

## Filming so the numbers are good

These are ordered by how much they actually decide whether a clip can be read
at all. The first one is not a nicety — at 30 fps most shots are simply not in
the footage.

1. **Record in slow motion — 120 or 240 fps.** A puck only exists on camera for
   `distance / speed` seconds. A 45 mph shot from 10 ft is in the air for about
   a seventh of a second, which at 30 fps is *four frames*, of which the puck is
   cleanly visible in one or two. The tracker needs at least four sightings and
   wants eight. Every recent phone shoots 240 fps slow-motion; switching to it
   turns four frames into thirty.

   | distance | 30 fps | 60 fps | 120 fps | 240 fps |
   | --- | --- | --- | --- | --- |
   | 10 ft | 4.5 | 9.1 | 18.2 | 36.4 |
   | 20 ft | 9.1 | 18.2 | 36.4 | 72.7 |

   *(frames of flight for a 45 mph shot)*

2. **Shoot in even light.** Flat overcast, open shade, or indoors is ideal.
   Direct sun filtered through trees is the worst case there is: the dappled
   pattern crawls across the grass frame to frame, so the detector sees
   hundreds of moving dark shapes and the puck is one of them. On real backyard
   footage in dappled sun this produced ~320 false candidates per frame against
   a puck about 17 pixels in area.

3. **Prop the phone against something.** A fence post, a bag, a chair. The
   background model assumes the camera holds still.

4. **Tell it how far out you were shooting** (`--distance`, in feet). Speed is
   directly proportional to it. Pacing it out is fine.

5. **Put the camera off to one side**, not directly behind you. Square-on to the
   net, the goal's outline reveals nothing about your lens and the tracker has
   to assume one. Ten feet to the side is plenty.

6. **Frame it tight** — the net and the shooting lane, nothing else. At 540 px
   wide a puck is about three pixels across; filling the frame with the part
   that matters is free resolution.

If you shoot slow-motion, check that the file reports the real rate — many
phones write 30 fps into a 240 fps file — and pass `--fps` if it does not.
Every speed scales directly with it.

## How accurate is it

Measured against rendered clips where the shot speed and impact point are known
exactly. Reproduce with `shottracker benchmark`:

```
camera         net   shots   on-net  off-net   zones    speed   worst  in bars
           % width   found   inches   inches   right   mean %       %
------------------------------------------------------------------------------
side          0.66     4/4      1.3      6.6     4/4      1.4     2.8      4/4
angled        0.60     4/4      0.7      3.6     4/4      0.7     2.3      4/4
head_on       0.40     4/4      0.4      1.1     4/4      6.6     8.3      4/4   (lens assumed)
```

So on a 72" × 48" goal mouth, impacts land within about an inch, every shot's
target zone is identified correctly, and speed is within a couple of percent
when the camera is off to one side. Square-on to the net, speed degrades to
under 10% because the lens has to be assumed — and the reported error bar grows
to match. Every shot's true speed fell inside its stated ± in all three cases,
which is the property that actually matters: a number without an honest error
bar is worse than no number.

These are synthetic clips. They model perspective, motion blur, sensor noise,
ballistic flight and the red rink lines that trip up a naive detector, but they
are not a substitute for real footage — see *Where it needs real video* below.

## How it works

Five stages, in `src/shottracker/`:

| Stage | File | What it does |
| --- | --- | --- |
| Find the goal | `net_detect.py` | Segments the red pipe in HSV, fits the outer edge of each post and the top of the crossbar with RANSAC, and intersects them for the corners. Repeated over frames sampled across the clip; the median of the frames that agree wins. |
| Calibrate | `camera.py` | Four corners of a rectangle of known size determine the camera. Solves focal length from the orthonormality of the rotation's columns, then pose. |
| Steady the view | `stabilize.py` | Optional per-frame translation for hand-held clips. Off by default. |
| Find the puck | `puck_detect.py` | A per-pixel median over the clip is a clean, puck-free plate of the rink. Anything differing from it is a scored candidate. |
| Follow it | `tracking.py` | Grows trajectories best-first with velocity gating, then keeps only those that travel fast, in one direction, along a straight line in the image. |
| Score the shot | `shots.py`, `speed.py` | Maps the impact through the goal-plane homography into inches on the net, and estimates speed. |

The idea holding it together is the **goal plane**: a coordinate system in
inches with its origin on the ice at the centre of the mouth. A regulation goal
is 72" × 48" of known geometry, so finding it in the image fixes the scale for
everything else. Impacts are reported in that frame, which is why they can be
quoted in inches rather than pixels.

### Speed, and why it is the hard part

A single camera cannot see depth, and a puck flying at a net is mostly moving in
depth. No amount of tracking precision fixes that, so the tracker runs three
estimators and reports which one it trusted:

- **`time_of_flight`** — needs your shooting distance. Given it, the puck's
  flight line is known end to end: the release point is (offset, stick height,
  distance) and the impact point comes from the homography. Each detection
  back-projects to a ray, and where that ray crosses the flight line says how
  far along the flight the puck was. Those fractions advance linearly with
  time, and the slope of that line is the flight time. Nothing is extrapolated
  beyond what was actually seen. **This is the accurate one.**
- **`ballistic_3d`** — needs nothing. Fits a 3-D parabola whose projection
  matches the track. Uniform motion along a ray bundle is ambiguous (a slow
  near puck and a fast far puck look identical); only gravity breaks the tie,
  and over a two-tenths-of-a-second flight that is a faint signal. Carries a
  real uncertainty and says so.
- **`goal_plane`** — the roughest. Measures motion in the last frames before
  impact, where the puck is close enough to the goal plane that the plane's
  scale is nearly the puck's own. Only meaningful from well off to the side.

Reported error bars combine detection scatter with the things that actually
dominate: how well you know the shooting distance, and how well the goal
outline pinned down the camera.

## Layout

```
src/shottracker/    the tracker: config, geometry, camera, detection, tracking, speed, reporting
  synth.py          renders clips with exact ground truth (a pinhole camera and a ballistic puck)
  benchmark.py      grades the tracker against them
server/app.py       upload a clip, poll a job, fetch the result
web/                the browser app: canvas overlay on the original video, interactive shot chart
  stabilize.py      optional motion compensation (off by default, see below)
tests/              71 tests, including end-to-end accuracy against ground truth
```

Run the tests with `.venv/bin/python -m pytest` (about two minutes — most of it
is rendering video).

## What real footage showed

The accuracy table above is against rendered clips. The first real clip — a kid
shooting into a backyard net, 540x960 at 30 fps, handheld, in direct afternoon
sun through trees — could not be read at all. That is worth writing down
honestly, because the reasons are specific and mostly fixable at the camera:

| Problem | Measured | Fixable by |
| --- | --- | --- |
| Frame rate too low | ~5 frames of flight; puck visible in 1–2 | Slow-motion capture |
| Dappled sun through leaves | ~320 false candidates per frame | Even light |
| Hand-held camera | up to 80 px of drift | Propping the phone |
| Maroon goal pipe | posts at H≈159–175, **overlapping bare skin at H=177** | Marking the net by hand |

The last one is the interesting failure. Colour thresholding works on a
saturated red rink goal and cannot work here: this net's pipe sits in the same
HSV neighbourhood as the shooter's legs, and its crossbar was washed out to
S=26 by glare. No threshold separates those. Hence hand-marking, which is now a
first-class path in both the CLI (`--net`) and the web app.

What that clip changed in the code:

- **The detector refuses rather than guesses.** It had confidently reported a
  flowering shrub as the goal at 19% confidence. Below `min_confidence` it now
  returns nothing and says why, because a wrong outline poisons every number
  downstream.
- **It cannot hang any more.** Seed pairs grow with the square of the
  candidates per frame, so a failing foreground model took the search from
  seconds to over fifteen minutes. Seeds are capped and the saturation is
  reported.
- **It checks the capture before doing any work** and says plainly when the
  frame rate cannot resolve a shot at the stated distance.
- **Hand-marking the net in the browser**, on a server-decoded frame, so it
  works even for codecs the browser cannot play (that clip was HEVC).

A stabilization pass is included (`stabilize.py`) but **off by default**: on
that footage it made things 71% worse, because the noise was moving light
rather than a moving camera, and resampling blur pushed more pixels over the
difference threshold. It stays available, unproven, until there is footage that
shows it helping.

Still untested on real video, and next in line:

- **The shooter in frame.** A stick blade travelling with the puck at release
  is the case to watch.
- **Rebounds and multiple pucks.** A second trajectory overlapping the first is
  merged as a rebound; two pucks genuinely in the air need real multi-target
  tracking.
- **Shooter tutors.** A tarp over the net hides the pipe entirely.
- **Non-regulation nets.** Backyard goals are often not 72x48. `GoalSpec` takes
  the real dimensions, and getting them right matters — every impact position
  scales with them.

## Next

- Calibrate the puck size and blur against real clips to widen the detector's
  size gates without letting sticks in.
- Aim points: declare a target zone before a session and score against it.
- Session history and trends — is the grouping tightening, is the release
  getting quicker.
- On-device capture, so the phone records and analyzes without an upload.
