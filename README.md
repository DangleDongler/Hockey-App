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

Say what you were aiming at and every shot is scored against it:

```bash
.venv/bin/shottracker analyze shots.mp4 --distance 20 --target any_corner
```

```
Target    Any Corner (6" radius): 2 of 4 on target, 13" off on average
          no consistent lean -- the misses scatter rather than drift one way

  #   frame    speed          where
  1   60       71.3 mph ±4.2 Top Shelf Left   [on target]
  2   146      53.5 mph ±3.1 Bottom Right Corner   [on target]
  3   228      46.0 mph ±2.7 Five Hole   [30" off Bottom Right Corner]
  4   298      60.5 mph ±3.5 missed - 0.7 ft wide right   [20" off Top Shelf Right]
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

### If your net sits inside a backstop frame

Mark **the goal**, not the backstop. The two are easy to confuse: both are
rectangles, both are often the same colour, and the backstop is the more
obvious shape in frame. But the scale of every measurement comes from the
marked rectangle being the size you said it was. On one real setup the
backstop was 1.67x the goal's width — marking it would have turned a 60 mph
shot into 36 mph, with no other sign anything was wrong.

The results panel reports the camera distance it derived from your marking.
That number is the check: if it says the camera was 40 ft away and you know it
was 25, the wrong rectangle was marked.

Slow motion often arrives with the slowdown baked in: when an iPhone shares a
240 fps clip it can turn one real second into eight seconds of ordinary 30 fps
video. The app catches this from the sound, which is not slowed: eight seconds
of picture over one second of sound means it was filmed at 240. It then times
everything at 240 and says so in the notes. If only part of the clip is slowed
(a normal-speed start or end), the numbers match no filming rate and it says
that instead of guessing; trim the clip to the slow part, or pass `--fps`.
Every speed scales directly with the rate, so it is worth a glance at the
first line of the report.

## How accurate is it

Measured against rendered clips where the shot speed and impact point are known
exactly. Reproduce with `shottracker benchmark`:

```
camera         net   shots   on-net  off-net   zones    speed   worst  in bars
           % width   found   inches   inches   right   mean %       %
------------------------------------------------------------------------------
side          0.57     4/4      1.3      6.6     4/4      1.2     3.1      4/4
angled        0.75     4/4      0.6      3.5     4/4      1.5     2.9      4/4
head_on       0.39     4/4      0.3      1.1     4/4      1.0     1.5      4/4   (lens assumed)
```

The synthetic goal has bent corners like a real frame, so the corner the
detector has to recover is the point where the two straight sections *would*
meet — which no pixel sits on. It finds it to within 0.8% of the goal's width.

So on a 72" × 48" goal mouth, impacts land within about an inch, every shot's
target zone is identified correctly, and speed is within a couple of percent
from any of the three camera positions. Every shot's true speed fell inside its
stated ± in all three cases, which is the property that actually matters: a
number without an honest error bar is worse than no number.

These are synthetic clips. They model perspective, motion blur, sensor noise,
ballistic flight and the red rink lines that trip up a naive detector, but they
are not a substitute for real footage — see *Where it needs real video* below.

## How it works

Five stages, in `src/shottracker/`:

| Stage | File | What it does |
| --- | --- | --- |
| Find the goal | `net_detect.py` | Two methods. On bright red pipe: segment it, fit the outer edge of each post and the top of the crossbar, intersect. On dark, faded or washed-out pipe: find it by *shape* — pairs of thin upright bars with level feet, joined by a straight crossbar. Repeated over frames sampled across the clip; most must agree. |
| Calibrate | `camera.py` | Four corners of a rectangle of known size determine the camera. Solves focal length from the orthonormality of the rotation's columns, then pose — keeping the estimate that survives jittering the corners by the pixel they are known to. |
| Steady the view | `stabilize.py` | Optional per-frame translation for hand-held clips. Off by default. |
| Find the puck | `puck_detect.py` | A per-pixel median over the clip is a clean, puck-free plate of the rink. Anything differing from it is a scored candidate. |
| Follow it | `tracking.py` | Grows trajectories best-first with velocity gating, then keeps only those that travel fast, in one direction, along a straight line in the image. |
| Score the shot | `shots.py`, `speed.py` | Maps the impact through the goal-plane homography into inches on the net, and estimates speed. |

The idea holding it together is the **goal plane**: a coordinate system in
inches with its origin on the ice at the centre of the mouth. A regulation goal
is 72" × 48" of known geometry, so finding it in the image fixes the scale for
everything else. Impacts are reported in that frame, which is why they can be
quoted in inches rather than pixels.

### Finding a goal that isn't bright red

The first method keys on saturated red pipe, which is what a rink goal looks
like. The first real backyard net broke it: shaded maroon posts (hue ~159–175),
a crossbar washed out to near-white by sun, and — the real problem — the
shooter's bare legs at hue 177, squarely in the same band. No colour threshold
separates those, and it reported a flowering shrub as the goal.

So there is a second method that finds the goal the way a person does, by its
shape:

- **Posts** are bars many times taller than they are wide. The colour band can
  afford to be loose, because legs (≈4× taller than wide) and shrubs (≈3×)
  fail on shape. Thinness is judged row by row along a straight run, so red
  flowers resting on top of a post don't swallow it.
- **On the orange side of red**, only strongly coloured pixels count: stained
  fence boards measured saturation 56–89, sunlit pipe 145.
- **The crossbar** is found as a straight bar spanning post to post — bright on
  dark mesh or dark on bright ice, either way — and taken as the *first* such
  bar going up. The top of a backstop frame above it can be straighter and
  brighter, which is why "strongest" is the wrong rule. Two upright bars with
  no bar across them are not reported as a goal.
- **A goal inside a backstop frame** is taken over the backstop, never the
  reverse.

On the backyard clip it finds the goal in every frame where a post isn't
blocked, within ~3 px (2% of the goal's width) of a careful hand measurement.
On the synthetic goals it works from every camera angle, at 1.6–4.7%. It is
less precise than the colour method on clean red pipe (0.4–1.4%), which is why
its confidence is capped lower, and why the colour method goes first.

Most sampled frames must put the goal in the same place. When they don't —
hand-held footage, where the net wanders around the frame — no single outline
fits the clip and none is given; the marking screen opens instead, already
outlined on the frame being looked at, to confirm or nudge.

### The goal is not a box

Posts and crossbar are joined by a bend, so the mouth's top corners are
rounded and the "corner" is a point that does not physically exist — it is
where the two straight sections would meet. That matters more than it sounds:
the goal's known size is what sets the scale for every measurement, so anyone
aiming at the *visible* corner under-sizes the goal by roughly the bend radius
and pulls every speed down with it.

So the marking screen extends each edge into a long dashed line and asks you to
lay those along the straight lengths of pipe. Lining a line up with an edge is
something the eye does well; locating a corner that is not there is not. The
automatic detector already worked this way — it fits the post and crossbar
edges and intersects them — and on synthetic footage with bent corners it
recovers the true intersection to within 0.8% of the goal's width.

The bend is modelled rather than ignored: shots arriving in it are reported as
"left corner bend" rather than counted as goals, and the drawn outline follows
the pipe instead of cutting the corner. `GoalSpec.corner_radius_in` defaults to
6 inches, measured on the inside of both top bends of a Bauer steel goal (5.8"
and 6.4") from a square-on photo flattened onto the goal plane. The same photo
confirmed the 2⅜" pipe: the posts measure 2.39" against the 72" mouth. Other
makes may differ; the radius only affects the corner call and the drawing,
never the scale.

### Aiming

A target is an aim point with a hit radius. The named ones — `top_left`,
`top_right`, `bottom_left`, `bottom_right`, `five_hole` — sit six inches in
from the pipe and scale with the goal's real size. The groups — `top_shelf`,
`any_corner`, `low_corners` — score each shot against whichever member it was
plainly going for. `--target-at X,Y` sets a custom one in goal inches. The
default six-inch radius is about the size of the hanging target discs sold for
backyard nets.

A hit has to go in: a puck that clips the pipe inside the target circle rang
off, and is not counted.

The useful output is not the hit rate but the **lean**. Scatter is noise; a
consistent offset — every shot landing four inches under where it was aimed —
is a habit, and the one thing a shooter can fix between sessions. It is only
reported when it clears twice its own standard error *and* is at least two
inches, so nobody gets told they miss low on the strength of two shots.

Scoring only needs the impact points, so a finished session can be re-scored
against a different target instantly — in the web app, change "Score against"
on the results page; nothing is re-processed.

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
  targets.py        scoring shots against what the player was aiming at
  container.py      reads a video file's own track timing, to catch baked-in slow motion
tests/              124 tests, including end-to-end accuracy against ground truth
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
| Maroon goal pipe | posts at H≈159–175, **overlapping bare skin at H=177** | Fixed: found by shape instead of colour |
| Detection too slow for slow-motion | 4 frames/s → 33 min for a 30 s 240 fps clip | Fixed: now 42 frames/s |

The last one is the interesting failure. Colour thresholding works on a
saturated red rink goal and cannot work here: this net's pipe sits in the same
HSV neighbourhood as the shooter's legs, and its crossbar was washed out to
S=26 by glare. No threshold separates those. Hence hand-marking, which is now a
first-class path in both the CLI (`--net`) and the web app.

It also exposed a performance bug that would have bitten the very fix being
recommended. The detector compared the whole label image against each
component in turn, so a frame with a few hundred movers scanned half a
megapixel a few hundred times. Working inside each component's bounding box
instead took throughput from 4 to 42 frames per second — the difference
between 33 minutes and under 3 for a 30-second slow-motion clip. There is a
test pinning it.

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

### The first slow-motion clip

One wrist shot, filmed at 240 fps by a phone propped on the ground about 27 ft
from the net and 12 ft to the side, looking into the sun past the shooter. The
shooter stood 225 in (18 ft 9 in) from the goal line.

| | By hand, frame by frame | The app |
| --- | --- | --- |
| Release → impact | frame 130 → 209, 79 frames = 0.33 s | — |
| Speed | 18.75 ft / 0.33 s = 39 mph | 39.0 mph ± 4.1 |
| Where | just outside the right post, about 2 ft up | 0.6 ft wide right, 24 in up |
| Net | — | found on its own, 24 of 24 sampled frames agreeing |

The puck was dark and sharp against bright concrete in every frame of its
flight, even with the sun in shot, and the track began with the blade dragging
it -- so a stick travelling with the puck at release did not throw the speed.

What it changed in the code:

- **Slow motion is recognised from the sound.** The phone shared the clip as
  8.4 s of 30 fps video with 1.1 s of sound. Read naively, every speed comes
  out 8x too slow. See *Filming* above.
- **A shot has to come from somewhere.** After the impact the netting kept
  moving, and two bits of that motion were tracked and reported as shots, one
  at 223 mph. Tracks must now first appear at least a quarter of a goal width
  outside the goal and move toward it. On the synthetic benchmark this changes
  nothing; on this clip it set aside 337 bits of movement on the goal itself.
- **Shots are spaced in seconds, not frames.** The rule merging an impact with
  whatever follows it was eight frames: a quarter second at 30 fps, a
  thirtieth at 240. It is now a quarter second at any rate, and what follows
  an impact never replaces the shot however cleanly it was tracked.
- **"Lots of motion" is diagnosed, not assumed.** The report used to blame a
  hand-held camera whenever most frames were busy. It now measures camera
  drift on the frames already sampled, and here says correctly that the phone
  held still and the scene itself was moving.

The same rules on a 2½-minute session from the same spot at ordinary 30 fps
cut the reported shots from 71 to 27, but most of those 27 are still not
shots. At that rate and distance the puck is a few pixels for a handful of
frames, and the scene never kept still (the report now says exactly that).
Footage like it needs slow motion, or a camera nearer the net and off to the
side with the sun behind it.

Still untested on real video, and next in line:

- **Rebounds and multiple pucks.** A second trajectory overlapping the first is
  merged as a rebound; two pucks genuinely in the air need real multi-target
  tracking.
- **Shooter tutors.** A tarp over the net hides the pipe entirely.
- **Non-regulation nets.** Backyard goals are often not 72x48. The goal mouth
  is an input in both the CLI and the web app, and getting it right matters:
  every impact position and speed scales with it.

## Next

- Calibrate the puck size and blur against real clips to widen the detector's
  size gates without letting sticks in.
- Session history and trends — is the grouping tightening, is the lean going
  away, is the release getting quicker.
- Drills with a sequence of targets (top left, then top right, …). Held back
  for now because a single missed detection would misalign the sequence and
  score every later shot against the wrong target.
- On-device capture, so the phone records and analyzes without an upload.
