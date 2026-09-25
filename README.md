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

Slow motion usually comes a shot or two per clip. Give several clips of the
same net and they are read as one session: each clip gets its own net outline
and frame rate, and the shots pool into one chart, one set of numbers and one
target score.

```bash
.venv/bin/shottracker analyze IMG_0201.MOV IMG_0202.MOV IMG_0203.MOV --distance 18.75 \
    --target any_corner --chart session.svg
```

```
[1] IMG_0201.MOV
    Clip      1080x1920 @ 240 fps (slow motion, plays at 30), 251 frames
    ...
Shots     3   on net 2   posts 0   missed 1
  #   clip frame    speed          where
  1   1    208      39.0 mph ±4.1 missed - 0.6 ft wide right
  ...
```

Or run the web app and drag a clip in — or several, for a session:

```bash
PYTHONPATH=src:server .venv/bin/python -m uvicorn app:app --reload
# open http://127.0.0.1:8000
```

The browser plays your original clip and draws the analysis over it on a
canvas — nothing is re-encoded, so playback stays smooth and you can scrub to
any shot by clicking its mark on the chart. With several clips, a switcher
above the footage picks which one plays, and clicking a shot opens its clip.
If the net cannot be found in one clip you are asked to mark it in that clip
only, or to leave the clip out.

Every finished session is saved next to its clips (in `$SHOTTRACKER_DATA`,
the system temp folder by default; point it somewhere permanent to keep a
season's worth). The upload screen then shows **your progress**: the latest
session's average speed, on-net rate and on-target rate, each against the
average of the few sessions before it, with a trend line per number, and a
list of past sessions to reopen, re-score against a different target, or
delete. A handful of shots makes a noisy average, so the comparison is always
against several earlier sessions, never just the last one.

## Filming so the numbers are good

These are ordered by how much they actually decide whether a clip can be read
at all. The first one is not a nicety — at 30 fps most shots are simply not in
the footage.

1. **Record at 60 fps at least; slow motion (120 or 240 fps) is better
   still.** A puck only exists on camera for `distance / speed` seconds. A 45
   mph shot from 10 ft is in the air for about a seventh of a second, which at
   30 fps is *four frames*, of which the puck is cleanly visible in one or two.
   The tracker needs at least four sightings and wants eight. Ordinary 60 fps
   video (Settings → Camera → Record Video) doubles that and records which lens
   filmed it; slow motion multiplies it by eight but does not record the lens.

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

   For slow motion, also give the lens's field of view (`--hfov`, or *Camera
   details* in the web app): about 74° for the 0.5x lens held upright, 42° for
   1x. Ordinary video records its lens and needs nothing.

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
side          0.65     4/4      1.3      6.5     4/4      1.1     3.0      4/4
angled        0.87     4/4      0.6      3.5     4/4      1.9     3.2      4/4
head_on       0.33     4/4      0.3      1.1     4/4      1.8     3.6      4/4   (lens assumed)
```

The synthetic goal has bent corners like a real frame, so the corner the
detector has to recover is the point where the two straight sections *would*
meet — which no pixel sits on. It finds it to within 0.8% of the goal's width.

So on a 72" × 48" goal mouth, impacts land within about an inch, every shot's
target zone is identified correctly, and speed is within a couple of percent
from any of the three camera positions. Every shot's true speed fell inside its
stated ± in all three cases, which is the property that actually matters: a
number without an honest error bar is worse than no number.

### From a backyard camera

`tests/test_backyard.py` renders the setup the first real slow-motion clip came
from: a phone on the ground 27 ft out and 12 ft to the side, portrait, 240 fps,
the goal 75 px wide, with a bush beside it whose leaves keep moving and netting
that shakes after every hit. Three shots:

| | Result |
| --- | --- |
| Shots found | 3 of 3, nothing else (without the rules from real footage, the bush and netting add fakes) |
| Where they hit | 0.6–2.2 in off |
| Speed, lens guessed | about 14% low, inside the stated ± (it says ±3.9 on 37.6 against a true 44.0) |
| Speed, lens known | the gravity-based estimate is within 1% |

The speed is the open problem for this spot. A camera at ground level, square
to the net, cannot measure its own lens from the goal, and the pose it works
out from a 75 px goal is off by a few feet, which the time-of-flight estimate
inherits. The gravity-based estimate is exact once the lens is right.

On the real clip the same two estimates disagree the same way, and the reason
the gravity one looked absurd there (103 mph) turned out not to be the lens:

| Real shot, 240 fps | mph |
| --- | --- |
| Time of flight (what the app reports) | 39.0 ± 4.1 |
| By hand: release frame 130 → impact frame 209 over 18.75 ft | 39 |
| Gravity fit, whole track, 0.5× lens | 96 |
| Gravity fit, from frame 128 / 131 / 140 on, 0.5× lens | 46 / 45 / 44 |

The track begins while the puck is still being dragged on the blade, which is
not free flight, and fitting a parabola through that ruins the fit. From the
release on, it gives 44–46 mph, about 15% above the time-of-flight figure,
which is the gap the synthetic backyard shows between them with the true speed
on the gravity side. The lens behind that number is inferred, not known: the
net photo from the same phone was taken on the 0.5× ultra-wide (14 mm
equivalent), and the net's size in the video fits it; the 1× lens would make
the time-of-flight speed 33 mph, well away from the hand timing.

So the next steps for speed are: find the release in the track and fit only
the free flight; let the player say which lens they film with; and check both
estimates against one independent measurement (a radar reading, or the same
shot filmed from the side) before changing which one is reported.

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
  session.py        several clips of one goal read as one session
  history.py        sessions over time: one line per session, and progress against earlier ones
  fullspeed.py      reads a slow-motion shot as the phone would have recorded it at 30/60 fps
tests/              155 tests, including end-to-end accuracy against ground truth
```

Run the tests with `.venv/bin/python -m pytest` (about two minutes — most of it
is rendering video).

## At normal speed (30 and 60 fps)

Slow motion reads a shot well; the goal is to read it just as well from
ordinary video. Real normal-speed footage with known answers is scarce, so it
is made from slow motion: keeping every 8th frame of a 240 fps clip is exactly
what the phone records at 30 fps, and every 4th is 60 fps. Each starting
frame is a different real clip, so one slow-motion shot gives twelve
normal-speed test cases, judged against its own slow-motion reading:

```bash
.venv/bin/shottracker fullspeed IMG_0198.MOV --distance 18.75
```

On four real slow-motion shots (backyard, phone lying on the ground about
26 ft out and to the side, 0.5x lens), read at 240 fps and at every 60 and
30 fps version of each:

| | Speed vs the slow-motion reading | Mark vs the slow-motion reading |
| --- | --- | --- |
| 60 fps, shot 1 (just wide) | within 2% | within 2.4 in |
| 60 fps, shot 2 (into the net) | −5% to +2% | within 2.2 in |
| 60 fps, shot 3 (over the bar) | −2% to +11% | within 2.5 in |
| 60 fps, shot 4 (off the crossbar) | +5% to +24% | 2–4 in |
| 30 fps, all four | −13% to +45% | median 3 in, up to 52 in; 4 of 32 versions read nothing |

(Against each shot's slow-motion reading from before the second pass below,
so that a change cannot move its own yardstick. Across all sixteen 60 fps
versions: speed within 3.8% at the median, marks within 1.7 in at the median
and 3.8 in at worst.)

The copies are written losslessly: an earlier table here came from re-encoded
copies, and the re-encoding alone blurred the small net's pipes and some
pucks enough to change the result.

What is left at 60 fps comes from the camera spot, not the frame rate. The
two shots filmed from the ground 26 ft out on the far side read within a few
inches but swing ±10-25% on speed between copies: one or two pixels of
difference in where the net is outlined moves the worked-out camera by feet
at that distance. A puck that misses wide also carries on into the backstop,
and from far out and low that looks almost exactly like a puck still short
of the goal line. See *Where to put the phone* below.

30 fps is not good enough to be trusted, and no amount of processing will make
it so: a 40 mph shot moves about two feet between frames, and from one camera
*where* along its path the puck is has to come from the timing of those few
frames. 60 fps (normal video mode, not slow motion: Settings → Camera →
Record Video → 1080p at 60 fps) is where accuracy starts to hold.

What decides accuracy at normal speed was measured on synthetic backyard
scenes with exact answers, six shots per setup:

| Phone placement | 30 fps speed / mark (mean) | 60 fps speed / mark (mean) |
| --- | --- | --- |
| On the ground behind and beside the shooter (the real clips) | 14% / 7 in | 8% / 8 in |
| Chest height, behind the shooter at about 45° | 7% / 10 in | 4% / 6 in |
| Chest height, to the side of the shooting lane | 8% / 42 in | 5% / 22 in |

**Speed at normal frame rates is timed at the goal line.** The speed estimate
needs to know where the flight ends. It used to be where the puck was last
seen, which at 30 fps can be a yard short of the goal line, or, for a puck
that goes in, at the back of the net. Now the fractions-along-the-flight
fit works out *when* the puck reached the goal line and uses where it was in
the picture at that moment, where the goal-plane mapping is exact. From
straight behind the shooter that timing is too weakly measured to help, so it
is used only when the camera sees the flight at least 25° off its line. On the
benchmark it takes the side and angled cameras from 1.1% and 1.9% to 0.7% and
1.2%; the marks themselves still come from the last sighting, because moving
them by the same timing made them worse at 30 fps.

Two approaches were tried and set aside, for reasons worth keeping:

- **A full 3-D fit of the flight** (start on the ice at the shooting distance,
  gravity, every sighting): the camera position worked out from the goal is
  accurate near the goal but not 20 ft away at the shooter, where the true
  start of the flight lands 75 px from where that camera puts it. The fit
  followed the camera's error, not the puck.
- **A straight path from the shooting spot through the late sightings**: every
  sighting lies in one sheet through the camera, so any path in that sheet
  fits them all. Only timing says where along it the puck crossed the line.

### Where to put the phone

The phone's position decides more than the frame rate does. From the ground
26 ft out, the net is about a fifth of the frame wide, and its outline says
little about where the phone is: a pixel or two of error in its corners moves
the worked-out camera by feet, and the speed with it. Most of what looked like
that error on the real clips turned out to be the lawn hiding the bottom of
the posts, which the app now allows for (see *A third pass* below), but a
higher, closer phone is still better, in order:

1. **Record at 60 fps** (normal video, not slow motion). Normal iPhone video
   records which lens filmed it; the app reads that and no longer has to guess
   the field of view. On the 0.5x lens this was worth 40% (see below).
2. **Raise the phone** to chest height (a chair, a bag on a bench, a tripod)
   so the net is seen from above rather than edge-on.
3. **Bring it closer to the net and off to the side, about 45° from the
   shooting lane**, so the net fills a third or more of the frame width.

## What real footage showed

### Four clips from the ground at 60 fps and in slow motion

Two 4K clips at 60 fps and two 240 fps slow-motion clips, all from a phone on
the ground about 26 ft from the net, on the 0.5x lens, with sunlit trees and
backstop netting behind the goal. At first the app misread every one of them:
the wrong frame rate on both 60 fps clips, no net or a refused net on three,
a speed of 85 mph on the one it did read, and three to four made-up shots per
clip. Now each reads as exactly one shot, and the speeds agree with timing the
flight frame by frame:

| Clip | Hand-timed (release → net, 18.75 ft) | The app |
| --- | --- | --- |
| 60 fps #1, top right corner | frames 83½ → 100: 47 mph (45–50 within a frame) | 46.9 mph |
| 60 fps #2, top right | frames 58½ → 74: 50 mph (47–54 within a frame) | 50.7 mph |
| First slow-motion clip (below) | 39 mph | 39.7 mph |

What each clip broke, and what changed:

- **The 60 fps clips said they were 58.5 and 52.1 fps.** An iPhone's 60 fps
  file starts with a few frames at 30 fps, which drags the file's average rate
  down. The rate is now read from the frames' own timestamps.
- **The net was refused as "the camera moved".** The posts' feet were found in
  exactly the same place in every frame, but the top edge of the crossbar was
  read two ways -- the bar, or a band of netting just under it. Now the feet
  decide whether the camera held still, the crossbar is whatever most frames
  agree on, and frames that simply failed to find the net (glare, the shooter
  in the way) no longer count as the camera moving. Posts as short as 2.5% of
  the frame are considered, so the small net in glare is found.
- **The puck was found in every frame and still lost.** It ranked 7th to 25th
  among ~90 moving things a frame -- sunlit leaves, rippling netting -- and
  only the top 14 are followed. Anything seen in the same spot both just
  before and just after is now ranked last: a puck passes any spot once. On
  the 240 fps clip that let through all 60 puck sightings and demoted three
  quarters of the clutter.
- **Tracks bent by the drag before the release, or the drop into the net,
  were thrown away whole.** They are now trimmed, from whichever end is
  further off the line, down to the clean flight.
- **Short lines of flicker were reported as shots**, 2–4 per clip. A shot now
  needs 0.075 s worth of sightings (at least 3) in clear background. Measured
  on every real clip at 30, 60 and 240 fps: every made-up shot had at most 3
  such sightings, every real shot at 60 fps at least 7.
- **The lens was 40% off.** Solved from the net's outline, the 0.5x lens came
  out as 885 px instead of the 1426 px the file records; that put the camera
  two feet up and nearly behind the shooter, and read the shot at 85 mph. With
  the lens from the file it reads 47 mph. Slow motion does not record the lens,
  so for slow motion give the field of view (`--hfov 74` for the 0.5x lens
  held upright; about 42 for 1x). On the first slow-motion clip that takes the
  speed from 42.6 to 38.2 mph against a hand-timed 39.

Still open from these clips: on the two new slow-motion shots, the puck's
release sits 12% and 23% behind where an 18.75 ft shot would start it, as if
they were taken from 21–23 ft. Either the shooter stood further back, or
something in that setup differs; until that is known their speeds (60 and
56 mph) are not trusted.

### A second pass: long clips, the stick, and the net

Going back over the same clips, and the 2.6-minute session, turned up more:

- **4K HDR frames held ~1,600 moving specks each.** Gravel and concrete just
  in front of the lens shimmer by a few grey levels from sensor noise, and
  one threshold for the whole picture turned that into puck-sized specks by
  the thousand; the puck was cut from the list before anything could tell it
  apart, and whether a shot was found at all came down to rounding. Each
  pixel now has its own threshold: the larger of 20 grey levels and five
  times how much that pixel normally strays from the background. 200-400
  candidates a frame instead, and every real clip finds its shot.
- **Reading 4K video was the slow part.** OpenCV decoded the iPhone's 4K HDR
  video at 7 frames a second, most of it converting colour on one core; a
  two-minute clip took over twenty minutes just to read. Frames now come
  from PyAV (FFmpeg on every core), turned upright to match, with HDR's
  colour matched to OpenCV's by a curve learned from the first frame; the
  frames the net is found on still come from OpenCV, because the outline
  decides where the camera is and even a few grey levels moved it.
- **The blade carrying the puck counted as flight.** A detector sensitive
  enough to see the puck in flight sees it being carried too, slowly, before
  the release: 40 frames of it at 240 fps pulled one shot from 39 to 31 mph.
  Speed is now fitted from the first step at 65% of the pace the flight
  settles at.
- **Tracks ran on past the impact.** A stray point after a gap, off the
  line the puck was on (the netting springing back), is dropped at 60 fps
  and up, and near the net a sharp turn or a jump in speed (a drop into the
  mesh, a bounce off the bar) ends the track.
- **Skating counted as shots.** Tracks timed at under 12 mph are not shots;
  on the 2.6-minute session that set aside three slow movers of 5-11 mph.
- **A distance check.** When the flights start well behind the shooting spot
  given, the report says where they seem to start. On the two new
  slow-motion clips that is about 21-23 ft rather than 18.75.

Tried and dropped: sharpening the net outline to a fraction of a pixel on
the clip's median frame (worse on synthetic clips with exact outlines), and
decoding straight to grey at working size (faster, but a grey level of
difference at the net, where the puck overlaps the red post, moved a
synthetic shot's last sighting 5 px and its speed 7%; it stays available as
``grey_decode``).

### A third pass: many shots in one clip, and where the phone really was

A real practice session is one long clip with a shot every few seconds, so
the synthetic backyard was extended to that: 20 shots in 40 s at 60 fps,
1080p, a bush beside the net whose leaves never keep still, the netting
shaking after every hit, and a stick-sized blob sweeping past every few
seconds. Two phone spots: on the ground behind and beside the shooter (the
real clips), and chest height at about 45°. With the lens known, as it is
for normal phone video:

| | Before | After |
| --- | --- | --- |
| Ground: shots found | 17 of 20 | 19 of 20, nothing made up |
| Ground: speed, median / worst | 2.0% / 5.1% | 0.8% / 4.0% |
| Ground: mark, median / worst | 1.8 / 6.3 in | 2.4 / 6.3 in (two more, harder, shots found) |
| Chest height: shots found | 20 of 20 | 20 of 20, nothing made up |
| Chest height: speed, median / worst | 4.8% / 17.2% | 4.1% / 6.4% |
| Chest height: mark, median / worst | 4.8 / 25.5 in | 2.5 / 7.1 in |

At half that resolution (540 px wide) the ground spot found nothing at all:
the puck is a few pixels across there and loses to the leaves. Film at 1080p
or more. At 30 fps the same session finds only 6-9 of the 20 shots and makes
up 2-3: film at 60.

What changed:

- **The camera is fitted to the goal's corners.** The pose used to be read
  straight off the homography, which spreads any error in the outline into
  where the camera is: a pixel of error moved it 17-35 in. With the lens
  known, the pose is now fitted to the four corners themselves (OpenCV's
  planar solver), 3-13 in from the same corners. That alone took the chest
  camera's speed error from +4.8% to -2.2% when timed against the true
  impact. With the lens solved from the outline instead, the homography's
  pose stays: it is the one consistent with that solve, and fitting on top
  of it made the benchmark worse. It also stays where the view is far and
  low: there a pixel's error in the crossbar's height -- the outline's
  commonest error on real clips -- swings the fitted camera by 8-15 in, while
  the homography's pose, scaled by the goal's width, moves 1-2 in. Fitting
  the real clips (entered at the size their outlines fit) spread their 60 fps
  copies 7.1% on speed at the median instead of 5.4%, and 43% at worst
  instead of 25%. So the fit is used only where a crossbar pixel moves it
  less than 6 in.
- **The net's size is checked, and posts hidden in grass are put back.** A
  goal's proportions can be read from its outline once the lens is known.
  Every real clip so far, entered as 72 x 48, fitted a goal about 72 x 40-42
  in to a pixel or less, against 2.5-5 px for 72 x 48 -- and that fit put the
  phone where it was, on the ground 24-26 ft out, where 72 x 48 put it nine
  feet underground. The net is a regulation NHL net, so the outline was
  short, and looking at the frames says why: from a phone on the ground, the
  lawn in front of the net hides the bottom of the posts, and the outline
  stopped where the red did, 6-8 in up. With the lens known, the app now
  measures that hidden strip (the cut-off goal that fits the outline best)
  and puts the feet back where a goal of the size entered must have them;
  the notes say how much was hidden. On the three clips it applies to, the
  speeds now read 39.7, 46.9 and 50.7 mph against 39, 47 and 50 timed by
  hand, and marks sit a few inches higher, where the bottom of the net
  really is. On a synthetic lawn with 7 in of grass (`grass_in`), the feet
  come out 15-16 px high without this and within 3 px with it, and the marks
  go from 2.7-6.4 in off to 1.3-4.8. It needs the lens: without it the lens
  is worked out from the same short outline, speeds there read 7-10% high,
  and the notes now ask for the lens. The pose is only fitted to corners
  that fit the size entered.
- **Crowded clutter ranks last.** A bush in the wind is dozens of leaves that
  each move too far to count as recurring, and they took every place under
  the per-frame cap from a puck crossing open ground. Candidates in a crowd
  -- more than 10 others within half a goal width -- now rank after the ones
  standing alone. On the real clips' 60 fps versions that took marks from
  2.1 to 1.7 in at the median and 7.9 to 3.8 in at worst; at 30 fps, 4.4 to
  2.9 in and 85 to 52 in. The first slow-motion clip now reads 39.5 mph
  against a hand-timed 39.
- **A sighting after the impact is dropped.** The frame after a hit, a track
  can pick up the netting springing back or a leaf at the same pace, and end
  on it: 6 of the 20 chest-height shots did, 50-80 px past where the puck
  stopped. Timed from the sightings before it, such a last sighting comes
  after the puck had reached the goal line, and is dropped (60 fps and up,
  lens known, only after missing frames).
- **Long clips are faster.** Frames are searched for the puck on every core,
  and salvaging the straight part of a bent track no longer tries every
  stretch: 2.6 minutes of 1080p30 in about 3 minutes.
- **"The camera moved" needs a quarter of the frames to agree.** One frame
  thrown off by something big crossing the view read as 12 px of drift on a
  camera that never moved.

The synthetic chest-height camera with the lens *solved from the outline*
still reads speeds 14% high: the lens comes out 15% long from a clean
outline. Pick the lens in the form for slow motion, which does not record it.

Tried and dropped: following a leaning post's whole width row by row. Posts
are found by their long upright runs of colour, which shaves a post leaning
6-7 deg (a phone held off level) to an upright sliver, and the outline's side
with it; taking each row's full run of colour fixed the lean but picked up
clutter touching the posts, and on the newer slow-motion clips the outline
then changed so much from sample to sample that their 60 fps copies read
54-82% fast. Those two clips also fit a view of about 66 deg best rather than
the 0.5x lens's 74, as if slow motion cropped the sensor; until that is known
they are left as they were.

### The first real clip

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

- **A shot is on camera for a tenth of a second or more.** The minimum track
  was a count of detections, 4: a fair 0.13 s at 30 fps, but 0.017 s at 240,
  where leaves flickering beside the goal strung together into "shots".
- **Speed error bars include the camera.** At 240 fps the gravity-based fit
  has so many points that its curve-fitting error alone said ±1%, so it won
  every comparison while ignoring that the lens was guessed. It now carries
  the same camera uncertainty as the time-of-flight estimate.
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

- The newer slow-motion clips, filmed further to the right, still have an
  outline that fits no goal: one post leans in the picture and the outline
  runs down it at the wrong angle. Fix the post edges in oblique views.
- With the camera fitted to the corners, try the full 3-D flight fit again
  (set aside because the camera was only accurate near the goal).
- Calibrate the puck size and blur against real clips to widen the detector's
  size gates without letting sticks in.
- Drills with a sequence of targets (top left, then top right, …). Held back
  for now because a single missed detection would misalign the sequence and
  score every later shot against the wrong target.
- On-device capture, so the phone records and analyzes without an upload.
