/* Shot Tracker front end.
 *
 * The server returns geometry in source-video pixels and impacts in inches on
 * the goal plane.  Nothing is re-encoded: the browser plays the original clip
 * and draws the analysis over it on a canvas, which keeps playback smooth and
 * lets the overlay be toggled without another round trip.
 */

const $ = (id) => document.getElementById(id);
const POLL_MS = 700;

const state = {
  jobId: null,
  result: null,    // the whole session: every clip's shots pooled
  clip: 0,         // which clip's footage is showing
  chartHits: [],   // click targets on the shot chart
  mark: { corners: [], img: null, info: null, frame: 0, hover: null, dragging: null, suggested: false, clip: 0 },
};

// The footage panel shows one clip at a time; everything else is the session.
const clipR = () => state.result?.clips?.[state.clip] ?? state.result;
// Clip endpoints take the clip's place in the upload, which differs from its
// place in the session once a clip has been left out.
const uploadIndex = (i = state.clip) => state.result?.clips?.[i]?.upload_index ?? i;
// Shots are numbered through the session, not within each clip.
function sessionNo(clip, localIndex) {
  const s = state.result?.shots?.find((x) => (x.clip ?? 0) === clip && (x.clip_shot ?? x.index) === localIndex);
  return s ? s.index + 1 : localIndex + 1;
}

function describeFiles(files) {
  const list = [...files];
  if (list.length <= 1) return list[0]?.name ?? "";
  return `${list.length} clips: ${list.map((f) => f.name).join(", ")}`;
}

/* ---------------------------------------------------------------- upload */

// The form remembers what was last used -- the same backyard, net and
// shooting spot, session after session.  Browser storage can be missing or
// refuse (private browsing), so every access is guarded.
const REMEMBERED = ["distance", "offset", "lens", "phone", "target", "target-radius", "goal-w", "goal-h"];
const SETTINGS_KEY = "shottracker.form";

function restoreForm() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch { saved = null; }
  if (!saved || typeof saved !== "object") return;
  for (const id of REMEMBERED) {
    const el = $(id);
    if (!el || saved[id] === undefined || saved[id] === null) continue;
    if (el.tagName === "SELECT" && ![...el.options].some((o) => o.value === saved[id])) continue;
    el.value = saved[id];
  }
}

function rememberForm() {
  const values = {};
  for (const id of REMEMBERED) if ($(id)) values[id] = $(id).value;
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(values)); } catch { /* not kept; fine */ }
}

restoreForm();

const fileInput = $("video-input");
const fileDrop = $("file-drop");

fileInput.addEventListener("change", () => {
  $("file-name").textContent = describeFiles(fileInput.files);
});

["dragenter", "dragover"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => { e.preventDefault(); fileDrop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => { e.preventDefault(); fileDrop.classList.remove("drag"); })
);
fileDrop.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    $("file-name").textContent = describeFiles(fileInput.files);
  }
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = [...fileInput.files];
  if (!files.length) return;

  showError(null);
  $("submit-btn").disabled = true;
  $("progress").classList.remove("hidden");
  setProgress(files.length > 1 ? `Uploading ${files.length} clips` : "Uploading", 0.02);

  const body = new FormData();
  for (const f of files) body.append("videos", f);
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  if (num("distance") !== null) body.append("shot_distance_ft", num("distance"));
  body.append("shooter_offset_ft", num("offset") ?? 0);
  if (num("hfov") !== null) body.append("hfov_deg", num("hfov"));
  if ($("lens").value) body.append("lens", $("lens").value);
  if ($("phone").value) body.append("phone", $("phone").value);
  rememberForm();
  if (num("fps") !== null) body.append("fps_override", num("fps"));
  if ($("target").value) body.append("target", $("target").value);
  if (num("target-radius") !== null) body.append("target_radius_in", num("target-radius"));
  if (num("goal-w") !== null) body.append("goal_width_in", num("goal-w"));
  if (num("goal-h") !== null) body.append("goal_height_in", num("goal-h"));

  try {
    const res = await fetch("/api/analyze", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail ?? `upload failed (${res.status})`);
    state.jobId = (await res.json()).id;
    poll();
  } catch (err) {
    showError(err.message);
    $("submit-btn").disabled = false;
    $("progress").classList.add("hidden");
  }
});

$("again").addEventListener("click", () => {
  $("results").classList.add("hidden");
  $("mark-panel").classList.add("hidden");
  state.mark = { corners: [], img: null, info: null, frame: 0, hover: null, dragging: null, suggested: false, clip: 0 };
  state.clip = 0;
  $("upload-panel").classList.remove("hidden");
  loadHistory();
  $("submit-btn").disabled = false;
  $("progress").classList.add("hidden");
});

async function poll() {
  try {
    const job = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    setProgress(job.stage, job.progress);
    if (job.status === "done") {
      // No outline means every downstream number is unavailable, so offer the
      // one thing that always works: let the player point at the net -- one
      // clip at a time, for each clip where it could not be found.
      const pending = (job.clips ?? []).findIndex((c) => !c.has_net && !c.skipped);
      if (pending >= 0) return offerManualMarking(pending, job);
      if (!job.result) throw new Error("Every clip was left out, so there is nothing to show.");
      return render(job.result);
    }
    if (job.status === "error") throw new Error(job.error);
    setTimeout(poll, POLL_MS);
  } catch (err) {
    showError(err.message);
    $("submit-btn").disabled = false;
    $("progress").classList.add("hidden");
  }
}

function setProgress(stage, frac) {
  $("bar-fill").style.width = `${Math.round(frac * 100)}%`;
  $("progress-text").textContent = `${stage}…`;
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg ?? "";
  el.classList.toggle("hidden", !msg);
}

/* --------------------------------------------------------------- results */

let overlayRunning = false;

function render(result) {
  state.result = result;
  state.clip = Math.min(state.clip, Math.max((result.clips?.length ?? 1) - 1, 0));
  $("upload-panel").classList.add("hidden");
  $("mark-panel").classList.add("hidden");
  $("history").classList.add("hidden");
  $("results").classList.remove("hidden");
  $("marked-status").classList.add("hidden");

  renderStats(result);
  renderTargeting(result);
  renderTable(result);
  renderNotes(result);
  renderClipTabs();
  drawChart();
  loadClipVideo();
  if (!overlayRunning) {
    overlayRunning = true;
    requestAnimationFrame(drawOverlay);
  }
}

function loadClipVideo(then) {
  const video = $("video");
  $("video-hint").textContent = "";
  sizeOverlay();                     // the analysis already told us the dimensions
  video.addEventListener("loadedmetadata", () => { sizeOverlay(); then?.(); }, { once: true });
  video.addEventListener("error", () => {
    $("video-hint").textContent =
      "This browser cannot play the clip's codec, so the overlay is unavailable. " +
      "The shot chart and numbers below are unaffected.";
  }, { once: true });
  video.src = `/api/jobs/${state.jobId}/video?clip=${uploadIndex()}`;
}

function selectClip(i, then) {
  if (i === state.clip) return then?.();
  state.clip = i;
  $("marked-status").classList.add("hidden");
  renderClipTabs();
  renderStats(state.result);
  loadClipVideo(then);
}

function renderClipTabs() {
  const clips = state.result?.clips ?? [];
  const el = $("clip-tabs");
  el.classList.toggle("hidden", clips.length < 2);
  if (clips.length < 2) return;
  el.innerHTML = clips.map((c, i) => {
    const n = state.result.shots.filter((s) => (s.clip ?? 0) === i).length;
    const count = c.net ? `${n} shot${n === 1 ? "" : "s"}` : "net not found";
    return `<button type="button" class="clip-tab${c.net ? "" : " no-net"}" role="tab"
      aria-selected="${i === state.clip}" data-clip="${i}" title="${escapeHtml(c.name ?? "")}">
      Clip ${i + 1}<span class="count">\u00b7 ${count}</span></button>`;
  }).join("");
  el.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => selectClip(Number(b.dataset.clip))));
}

function renderStats(r) {
  const s = r.summary;
  const speed = s.speed_mph ?? {};
  const cards = [
    { value: s.shots, label: "shots" },
    { value: s.accuracy_pct != null ? `${Math.round(s.accuracy_pct)}%` : "–", label: "on net",
      sub: `${s.on_net} of ${s.shots}` },
    { value: speed.max != null ? Math.round(speed.max) : "–", label: "top speed", sub: "mph" },
    { value: speed.mean != null ? Math.round(speed.mean) : "–", label: "average", sub: "mph" },
  ];
  if (s.grouping) {
    cards.push({ value: `${Math.round(s.grouping.spread_in)}"`, label: "grouping", sub: "spread from centre" });
  }
  if (s.corner_precision) {
    cards.push({ value: `${Math.round(s.corner_precision.best_distance_in)}"`, label: "best corner",
                 sub: "closest to a corner" });
  }
  if (r.targeting) {
    const ts = r.targeting.summary;
    cards.splice(2, 0, { value: `${ts.hits}/${ts.shots}`, label: "on target",
                         sub: r.targeting.label });
  }

  // The camera distance follows from the marked net being the size it was said
  // to be. If the wrong rectangle was marked -- a backstop frame rather than
  // the goal -- this is the number that gives it away, so it sits with the
  // footage as a check rather than among the scores.
  const cam = $("camera-check");
  const camera = clipR()?.camera;
  if (camera?.position_in) {
    const ft = camera.position_in[2] / 12;
    cam.textContent = `The camera works out to about ${ft.toFixed(0)} ft from the net. ` +
      "If that's clearly wrong, the net was marked on the wrong rectangle.";
    cam.classList.remove("hidden");
  } else {
    cam.classList.add("hidden");
  }

  $("stats").innerHTML = cards.map((c) => `
    <div>
      <div class="stat-value">${c.value}</div>
      <div class="stat-label">${c.label}</div>
      ${c.sub ? `<div class="stat-sub">${c.sub}</div>` : ""}
    </div>`).join("");
}

function renderTargeting(r) {
  const t = r.targeting;
  $("target-live").value = t?.kind ?? "";
  if (!t) {
    $("target-verdict").textContent =
      "Pick a target to see how many shots found it and whether the misses lean one way.";
    $("target-bias").textContent = "";
    $("target-bias").className = "target-bias";
    return;
  }
  const s = t.summary;
  const off = s.mean_distance_in != null ? ` \u00b7 ${Math.round(s.mean_distance_in)}" off on average` : "";
  $("target-verdict").innerHTML =
    `<strong>${s.hits}/${s.shots}</strong> on target (${escapeHtml(t.label)}, ${t.radius_in}" radius)${off}`;
  const bias = s.bias?.description ?? "";
  $("target-bias").textContent = bias ? bias.charAt(0).toUpperCase() + bias.slice(1) + "." : "";
  $("target-bias").className = "target-bias" + (s.bias?.significant ? " lean" : "");
}

// Re-scoring is pure geometry on the impact points, so the server answers
// instantly and the player can try "what if I'd been aiming top shelf?".
$("target-live").addEventListener("change", async (e) => {
  const body = new FormData();
  body.append("target", e.target.value);
  const radius = $("target-radius").value.trim();
  if (radius) body.append("target_radius_in", radius);
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/target`, { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail ?? "could not re-score");
    const job = await res.json();
    state.result = job.result;
    renderStats(job.result);
    renderTargeting(job.result);
    renderTable(job.result);
    drawChart();
  } catch (err) {
    showError(err.message);
  }
});

function renderTable(r) {
  const multi = (r.clips?.length ?? 1) > 1;
  const rows = r.shots.map((s) => {
    const speed = s.speed
      ? `${s.speed.mph.toFixed(1)} <span class="unc">${s.speed.uncertainty_mph != null ? `±${s.speed.uncertainty_mph.toFixed(1)}` : ""}</span>`
      : "–";
    const where = s.zone_label ?? s.miss_detail ?? s.outcome;
    const label = { on_net: "on net", post: "post", miss: "missed" }[s.outcome] ?? s.outcome;
    const vt = s.vs_target;
    const aim = !vt ? "" : vt.hit
      ? `<span class="vs-hit">on target</span>`
      : `${Math.round(vt.distance_in)}" off <span class="unc">${escapeHtml(vt.target_label)}</span>`;
    return `<tr data-frame="${s.impact_frame}" data-clip="${s.clip ?? 0}">
      <td>${s.index + 1}</td>
      ${multi ? `<td class="unc">${(s.clip ?? 0) + 1}</td>` : ""}
      <td>${speed}</td>
      <td><span class="pill ${s.outcome}">${label}</span></td>
      <td>${where}</td>
      ${r.targeting ? `<td>${aim}</td>` : ""}
      <td class="unc">${s.impact_goal_in[0] >= 0 ? "+" : ""}${s.impact_goal_in[0].toFixed(0)}", ${s.impact_goal_in[1].toFixed(0)}" up</td>
    </tr>`;
  }).join("");
  $("shot-table").innerHTML =
    `<thead><tr><th>#</th>${multi ? "<th>Clip</th>" : ""}<th>Speed (mph)</th><th>Result</th><th>Where</th>
       ${r.targeting ? "<th>Vs target</th>" : ""}<th>Position</th></tr></thead>
     <tbody>${rows}</tbody>`;
  $("shot-table").querySelectorAll("tbody tr").forEach((tr) =>
    tr.addEventListener("click", () => seekToFrame(Number(tr.dataset.frame), Number(tr.dataset.clip)))
  );
}

function renderNotes(r) {
  const notes = [...(r.warnings ?? [])];
  r.shots.forEach((s) => (s.speed?.notes ?? []).forEach((n) => notes.push(`Shot ${s.index + 1}: ${n}`)));
  $("notes-card").classList.toggle("hidden", notes.length === 0);
  $("notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function seekToFrame(frame, clip = state.clip) {
  if (clip !== state.clip) return selectClip(clip, () => seekToFrame(frame, clip));
  const video = $("video");
  // Seeking goes by the rate the file plays at, which slow motion makes lower than the rate it was filmed at.
  const fps = clipR()?.video?.playback_fps || clipR()?.video?.fps;
  if (!fps || !Number.isFinite(video.duration)) return;
  video.currentTime = Math.max(0, frame / fps - 0.15);
  video.pause();
  drawOverlay();
}

/* ------------------------------------------------------- marking the net */

async function offerManualMarking(clip, job) {
  $("progress").classList.add("hidden");
  $("upload-panel").classList.add("hidden");
  $("results").classList.add("hidden");
  $("history").classList.add("hidden");
  $("mark-panel").classList.remove("hidden");

  state.mark = { corners: [], img: null, info: null, frame: 0, hover: null, dragging: null, suggested: false, clip };
  const clipResult = (job.result?.clips ?? []).find((c) => (c.upload_index ?? 0) === clip);
  const why = (clipResult?.warnings || []).find((w) => w.includes("goal")) ||
    "The goal could not be found automatically.";
  const many = (job.clips?.length ?? 1) > 1;
  const which = many ? `Clip ${clip + 1} (${job.clips[clip].name}): ` : "";
  $("mark-intro").textContent = which + why + " Mark it once here and the clip will be re-read.";
  $("mark-skip").classList.toggle("hidden", !many);
  $("mark-go").disabled = true;

  state.mark.info = await (await fetch(`/api/jobs/${state.jobId}/info?clip=${clip}`)).json();
  const slider = $("mark-frame");
  slider.max = Math.max(0, (state.mark.info.frame_count || 1) - 1);
  slider.value = Math.floor((state.mark.info.frame_count || 1) / 2);
  state.mark.frame = Number(slider.value);
  await loadMarkFrame();
}

function loadMarkFrame() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = async () => {
      state.mark.img = img;
      const canvas = $("mark-canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      await suggestCorners();
      drawMark();
      resolve();
    };
    img.src = `/api/jobs/${state.jobId}/frame.png?frame=${state.mark.frame}&clip=${state.mark.clip}`;
  });
}

// Ask the server to find the goal in this frame and start from that. The
// player only has to check the dashed lines and nudge a corner if it is off.
async function suggestCorners() {
  // Never overwrite corners the player has started placing themselves.
  if (state.mark.corners.length && !state.mark.suggested) return;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/suggest-net?frame=${state.mark.frame}&clip=${state.mark.clip}`);
    const s = await res.json();
    if (s.quad) {
      state.mark.corners = s.quad.map(([x, y]) => [x, y]);
      state.mark.suggested = true;
      $("mark-go").disabled = false;
      $("mark-found").textContent =
        "Found the goal in this frame. Check the dashed lines sit on the outside of the pipe, " +
        "drag any corner that is off, then analyze.";
      $("mark-found").classList.remove("hidden");
    } else {
      if (state.mark.suggested) state.mark.corners = [];
      state.mark.suggested = false;
      $("mark-go").disabled = state.mark.corners.length !== 4;
      $("mark-found").classList.add("hidden");
    }
  } catch {
    // A failed suggestion just means marking by hand, which always works.
  }
}

$("mark-frame").addEventListener("change", async (e) => {
  state.mark.frame = Number(e.target.value);
  await loadMarkFrame();
});

function drawMark() {
  const canvas = $("mark-canvas");
  const ctx = canvas.getContext("2d");
  if (state.mark.img) ctx.drawImage(state.mark.img, 0, 0);

  const pts = state.mark.corners;
  const r = Math.max(6, canvas.width * 0.012);
  ctx.lineWidth = Math.max(2, canvas.width * 0.005);

  if (pts.length > 1) {
    // Extend every edge right across the frame. Lining a long line up with a
    // straight length of pipe is something the eye does well; locating the
    // corner itself is not, because the pipe bends and the corner is a point
    // that only exists where the two straight sections would have met.
    ctx.strokeStyle = "rgba(56,189,248,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([7, 7]);
    const edges = pts.length === 4 ? [[0, 1], [1, 2], [2, 3], [3, 0]] : [[0, 1]];
    for (const [a, b] of edges) {
      if (!pts[a] || !pts[b]) continue;
      const dx = pts[b][0] - pts[a][0];
      const dy = pts[b][1] - pts[a][1];
      const n = Math.hypot(dx, dy) || 1;
      const k = (canvas.width + canvas.height) / n;
      line(ctx, pts[a][0] - dx * k, pts[a][1] - dy * k, pts[b][0] + dx * k, pts[b][1] + dy * k);
    }
    ctx.setLineDash([]);

    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = Math.max(2, canvas.width * 0.005);
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (pts.length === 4) ctx.closePath();
    ctx.stroke();
    if (pts.length === 4) drawMouthPreview(ctx, pts);
  }
  const labels = ["TL", "TR", "BR", "BL"];
  pts.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = state.mark.dragging === i ? "#fbbf24" : "#38bdf8";
    ctx.fill();
    ctx.strokeStyle = "#0b1220";
    ctx.stroke();
    // A cross through the middle: the dot alone hides the pixel it marks.
    ctx.beginPath();
    ctx.moveTo(x - r * 1.8, y); ctx.lineTo(x + r * 1.8, y);
    ctx.moveTo(x, y - r * 1.8); ctx.lineTo(x, y + r * 1.8);
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.lineWidth = Math.max(2, canvas.width * 0.005);
    ctx.fillStyle = "#e8eefc";
    ctx.font = `600 ${Math.max(13, canvas.width * 0.032)}px system-ui, sans-serif`;
    ctx.fillText(labels[i], x + r + 5, y - r);
  });

  drawLoupe(ctx, canvas);
  updateMarkStatus();
}

/* Preview of the opening the marked corners imply, bends and all, so what is
   on screen matches the net being looked at. */
function drawMouthPreview(ctx, quad) {
  const gw = Number($("goal-w")?.value) || 72;
  const gh = Number($("goal-h")?.value) || 48;
  const post = 2.375;
  const rad = 4;
  // Bilinear placement inside the marked outer quad is close enough for a
  // preview; the analysis itself uses the full homography.
  const at = (u, v) => {
    const [tl, tr, br, bl] = quad;
    const top = [tl[0] + (tr[0] - tl[0]) * u, tl[1] + (tr[1] - tl[1]) * u];
    const bot = [bl[0] + (br[0] - bl[0]) * u, bl[1] + (br[1] - bl[1]) * u];
    return [bot[0] + (top[0] - bot[0]) * v, bot[1] + (top[1] - bot[1]) * v];
  };
  const ow = gw + 2 * post;
  const oh = gh + post;
  const ux = (xin) => (xin + ow / 2) / ow;
  const uy = (yin) => yin / oh;

  const path = [];
  path.push(at(ux(-gw / 2), uy(0)));
  const steps = 10;
  for (let i = 0; i <= steps; i++) {
    const t = Math.PI - (i / steps) * (Math.PI / 2);
    path.push(at(ux(-gw / 2 + rad + rad * Math.cos(t)), uy(gh - rad + rad * Math.sin(t))));
  }
  for (let i = 0; i <= steps; i++) {
    const t = Math.PI / 2 - (i / steps) * (Math.PI / 2);
    path.push(at(ux(gw / 2 - rad + rad * Math.cos(t)), uy(gh - rad + rad * Math.sin(t))));
  }
  path.push(at(ux(gw / 2), uy(0)));

  ctx.strokeStyle = "rgba(34,197,94,0.9)";
  ctx.lineWidth = Math.max(1.5, ctx.canvas.width * 0.004);
  ctx.beginPath();
  path.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

/* A magnifier under the cursor. Goal pipe is a few pixels wide in phone
   footage, and the scale of every measurement comes from landing on it, so
   placing corners by eye at display size is not good enough. */
const LOUPE_ZOOM = 6;

function drawLoupe(ctx, canvas) {
  const at = state.mark.hover;
  if (!at || !state.mark.img) return;

  const size = Math.round(Math.min(canvas.width, canvas.height) * 0.42);
  const src = size / LOUPE_ZOOM;
  // Keep the loupe away from the point being placed, and inside the frame.
  const pad = Math.round(canvas.width * 0.03);
  const left = at.x < canvas.width / 2 ? canvas.width - size - pad : pad;
  const top = at.y < canvas.height / 2 ? canvas.height - size - pad : pad;

  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, size, size);
  ctx.clip();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    state.mark.img,
    at.x - src / 2, at.y - src / 2, src, src,
    left, top, size, size,
  );
  ctx.restore();

  // Crosshair on the exact pixel under the cursor.
  const cx = left + size / 2;
  const cy = top + size / 2;
  ctx.strokeStyle = "rgba(56,189,248,0.95)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx - size * 0.16, cy); ctx.lineTo(cx - 3, cy);
  ctx.moveTo(cx + 3, cy); ctx.lineTo(cx + size * 0.16, cy);
  ctx.moveTo(cx, cy - size * 0.16); ctx.lineTo(cx, cy - 3);
  ctx.moveTo(cx, cy + 3); ctx.lineTo(cx, cy + size * 0.16);
  ctx.stroke();

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.strokeRect(left, top, size, size);
}

/* Tell the player whether the shape they have drawn could be the goal they
   said it is. A rectangle seen at an angle looks narrower than head-on, never
   wider, so an aspect well above the head-on figure means a mis-placed corner. */
function updateMarkStatus() {
  const el = $("mark-status");
  if (!el) return;
  const pts = state.mark.corners;
  if (pts.length < 4) {
    el.textContent = `${pts.length} of 4 corners placed`;
    el.className = "mark-status";
    return;
  }
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const w = (dist(pts[0], pts[1]) + dist(pts[3], pts[2])) / 2;
  const h = (dist(pts[0], pts[3]) + dist(pts[1], pts[2])) / 2;
  const aspect = h > 0 ? w / h : 0;

  const gw = Number($("goal-w")?.value) || 72;
  const gh = Number($("goal-h")?.value) || 48;
  const post = 2.375;
  const headOn = (gw + 2 * post) / (gh + post);

  if (aspect > headOn * 1.15) {
    el.textContent = `That shape is wider than a ${gw}x${gh}" goal can look (${aspect.toFixed(2)} vs ${headOn.toFixed(2)} head-on). Check the top and bottom corners.`;
    el.className = "mark-status bad";
  } else if (aspect < headOn * 0.45) {
    el.textContent = `Very side-on (${aspect.toFixed(2)} vs ${headOn.toFixed(2)} head-on). Fine if the camera was well off to the side, otherwise check the corners.`;
    el.className = "mark-status warn-text";
  } else {
    el.textContent = `Shape is consistent with a ${gw}x${gh}" goal (${aspect.toFixed(2)} vs ${headOn.toFixed(2)} head-on). Drag any corner to fine-tune.`;
    el.className = "mark-status good";
  }
}

function canvasPoint(e) {
  const canvas = $("mark-canvas");
  const rect = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * (canvas.width / rect.width),
    y: (src.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function nearestCorner(pt) {
  const canvas = $("mark-canvas");
  const grab = canvas.width * 0.05;
  let best = null;
  let bestD = grab;
  state.mark.corners.forEach(([x, y], i) => {
    const d = Math.hypot(x - pt.x, y - pt.y);
    if (d < bestD) { bestD = d; best = i; }
  });
  return best;
}

function onMarkDown(e) {
  e.preventDefault();
  const pt = canvasPoint(e);
  state.mark.hover = pt;
  const hit = nearestCorner(pt);
  if (hit !== null) {
    state.mark.dragging = hit;
    state.mark.suggested = false;
  } else if (state.mark.corners.length < 4) {
    state.mark.corners.push([pt.x, pt.y]);
  }
  $("mark-go").disabled = state.mark.corners.length !== 4;
  drawMark();
}

function onMarkMove(e) {
  if (e.touches) e.preventDefault();
  const pt = canvasPoint(e);
  state.mark.hover = pt;
  if (state.mark.dragging !== null) state.mark.corners[state.mark.dragging] = [pt.x, pt.y];
  drawMark();
}

function onMarkUp() {
  state.mark.dragging = null;
  drawMark();
}

const markCanvas = $("mark-canvas");
markCanvas.addEventListener("mousedown", onMarkDown);
markCanvas.addEventListener("mousemove", onMarkMove);
window.addEventListener("mouseup", onMarkUp);
markCanvas.addEventListener("mouseleave", () => { state.mark.hover = null; drawMark(); });
markCanvas.addEventListener("touchstart", onMarkDown, { passive: false });
markCanvas.addEventListener("touchmove", onMarkMove, { passive: false });
markCanvas.addEventListener("touchend", onMarkUp);

$("mark-undo").addEventListener("click", () => {
  state.mark.suggested = false;
  state.mark.corners.pop();
  $("mark-go").disabled = state.mark.corners.length !== 4;
  drawMark();
});

$("mark-go").addEventListener("click", async () => {
  const body = new FormData();
  body.append("net_quad", state.mark.corners.flat().map((v) => v.toFixed(1)).join(","));
  body.append("clip", state.mark.clip);
  $("mark-go").disabled = true;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/reanalyze`, { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail ?? "could not re-analyze");
    $("mark-panel").classList.add("hidden");
    $("upload-panel").classList.remove("hidden");
    $("progress").classList.remove("hidden");
    setProgress("re-reading with your net", 0.05);
    poll();
  } catch (err) {
    showError(err.message);
    $("mark-go").disabled = false;
  }
});

// A clip whose net cannot be marked -- the goal out of shot, say -- can be
// left out, and the rest of the session goes ahead without it.
$("mark-skip").addEventListener("click", async () => {
  const body = new FormData();
  body.append("clip", state.mark.clip);
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/skip`, { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail ?? "could not leave the clip out");
    poll();
  } catch (err) {
    showError(err.message);
  }
});

/* ------------------------------------------------------------ shot chart */

const OUTCOME_COLOR = { on_net: "#22c55e", post: "#f59e0b", miss: "#ef4444" };

function drawChart() {
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");
  const r = state.result;
  const goal = r.goal ?? {};
  const W = goal.mouth_width_in ?? 72;
  const H = goal.mouth_height_in ?? 48;

  // Show enough ice around the net that every mark fits, misses included.
  let pad = 9;
  for (const shot of r.shots) {
    const [gx, gy] = shot.impact_goal_in;
    pad = Math.max(pad, Math.abs(gx) - W / 2 + 7, gy - H + 7, -gy + 7);
  }
  const sx = canvas.width / (W + 2 * pad);
  const sy = canvas.height / (H + 2 * pad);
  const s = Math.min(sx, sy);
  const ox = (canvas.width - (W + 2 * pad) * s) / 2;
  const oy = (canvas.height - (H + 2 * pad) * s) / 2;
  const X = (xin) => ox + (xin + W / 2 + pad) * s;
  const Y = (yin) => oy + (H + pad - yin) * s;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0e1728";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // The mouth, with mesh.
  ctx.fillStyle = "#16233a";
  ctx.fillRect(X(-W / 2), Y(H), W * s, H * s);
  ctx.strokeStyle = "#24314c";
  ctx.lineWidth = 1;
  for (let gx = -W / 2; gx <= W / 2 + 0.01; gx += 6) line(ctx, X(gx), Y(0), X(gx), Y(H));
  for (let gy = 0; gy <= H + 0.01; gy += 6) line(ctx, X(-W / 2), Y(gy), X(W / 2), Y(gy));

  // Zone grid and counts.
  const counts = state.result.summary.zone_counts ?? {};
  if ($("show-zones").checked) {
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "#334669";
    for (const f of [1 / 3, 2 / 3]) {
      line(ctx, X(-W / 2 + f * W), Y(0), X(-W / 2 + f * W), Y(H));
      line(ctx, X(-W / 2), Y(f * H), X(W / 2), Y(f * H));
    }
    ctx.restore();

    const keys = [["top_left", 0, 2], ["top_mid", 1, 2], ["top_right", 2, 2],
                  ["mid_left", 0, 1], ["mid_mid", 1, 1], ["mid_right", 2, 1],
                  ["low_left", 0, 0], ["five_hole", 1, 0], ["low_right", 2, 0]];
    for (const [key, col, row] of keys) {
      const n = counts[key] ?? 0;
      if (!n) continue;
      const left = X(-W / 2 + col * W / 3);
      const top = Y((row + 1) * H / 3);
      ctx.fillStyle = "rgba(34,197,94,0.10)";
      ctx.fillRect(left, top, (W / 3) * s, (H / 3) * s);
      // The tally sits in the zone's corner so it never fights a shot mark,
      // which lands wherever the puck actually went.
      ctx.textAlign = "left";
      ctx.fillStyle = "#4ade80";
      ctx.font = "700 17px system-ui, sans-serif";
      ctx.fillText(`${n}`, left + 9, top + 22);
    }
  }

  // The pipe, bent where the posts meet the crossbar like the real thing.
  const pipe = (goal.post_diameter_in ?? 2.375) * s;
  const rad = Math.min(goal.corner_radius_in ?? 4, W / 2, H);
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = pipe;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(X(-W / 2), Y(0));
  ctx.lineTo(X(-W / 2), Y(H - rad));
  ctx.quadraticCurveTo(X(-W / 2), Y(H), X(-W / 2 + rad), Y(H));
  ctx.lineTo(X(W / 2 - rad), Y(H));
  ctx.quadraticCurveTo(X(W / 2), Y(H), X(W / 2), Y(H - rad));
  ctx.lineTo(X(W / 2), Y(0));
  ctx.stroke();
  ctx.lineCap = "butt";

  // Targets, under the marks so a hit is visibly inside its circle.
  for (const t of state.result.targeting?.targets ?? []) {
    ctx.beginPath();
    ctx.arc(X(t.x_in), Y(t.y_in), t.radius_in * s, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(59,130,246,0.12)";
    ctx.fill();
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#60a5fa";
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // The marks.
  state.chartHits = [];
  for (const shot of state.result.shots) {
    const [gx, gy] = shot.impact_goal_in;
    const px = X(gx), py = Y(gy);
    const rad = Math.max(7, 0.012 * canvas.width);
    ctx.beginPath();
    ctx.arc(px, py, rad, 0, Math.PI * 2);
    ctx.fillStyle = OUTCOME_COLOR[shot.outcome] ?? "#94a3b8";
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#0b1220";
    ctx.stroke();

    ctx.fillStyle = "#e8eefc";
    ctx.font = "700 15px system-ui, sans-serif";
    ctx.textAlign = "center";
    const tag = shot.speed ? `${shot.index + 1} · ${Math.round(shot.speed.mph)}` : `${shot.index + 1}`;
    ctx.fillText(tag, px, py - rad - 5);

    state.chartHits.push({ x: px, y: py, r: rad + 6, frame: shot.impact_frame, clip: shot.clip ?? 0 });
  }
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

$("chart").addEventListener("click", (e) => {
  const canvas = $("chart");
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const hit = state.chartHits.find((h) => Math.hypot(h.x - x, h.y - y) <= h.r);
  if (hit) seekToFrame(hit.frame, hit.clip);
});

$("show-zones").addEventListener("change", drawChart);

/* ------------------------------------------------------- video overlay */

function sizeOverlay() {
  const video = $("video");
  const canvas = $("overlay");
  const v = clipR()?.video ?? {};
  canvas.width = video.videoWidth || v.width || 1280;
  canvas.height = video.videoHeight || v.height || 720;
}

window.addEventListener("resize", () => { sizeOverlay(); drawOverlay(); });

function drawOverlay() {
  const video = $("video");
  const canvas = $("overlay");
  const r = clipR();
  if (r?.video && canvas.width) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const fps = r.video.playback_fps || r.video.fps || 30;
    const frame = Math.round((video.currentTime || 0) * fps);
    const u = screenPx(canvas);

    if (r.net?.quad) drawNet(ctx, r, $("show-zones").checked, u);
    for (const t of r.targeting?.targets ?? []) {
      if (!t.image_outline) continue;
      ctx.setLineDash([6 * u, 5 * u]);
      ctx.lineWidth = 2 * u;
      ctx.strokeStyle = "rgba(96,165,250,0.9)";
      poly(ctx, t.image_outline);
      ctx.setLineDash([]);
    }
    drawShots(ctx, r, frame, fps, u);
  }
  requestAnimationFrame(drawOverlay);
}

// The canvas is drawn at the video's own resolution -- 2160 px across for 4K
// -- so sizes are given in screen pixels and scaled by this.
function screenPx(canvas) {
  const shown = canvas.getBoundingClientRect().width;
  return shown ? canvas.width / shown : 1;
}

function drawNet(ctx, r, zones, u) {
  const outer = r.goal?.outer_outline ?? r.net.quad;
  const mouth = r.goal?.mouth_outline ?? r.goal?.mouth_quad;
  ctx.lineWidth = 2 * u;
  ctx.strokeStyle = "rgba(56,189,248,0.9)";
  poly(ctx, outer);
  if (mouth) {
    ctx.lineWidth = 1 * u;
    ctx.strokeStyle = "rgba(56,189,248,0.5)";
    poly(ctx, mouth);
    if (zones && r.goal?.mouth_quad) {
      ctx.lineWidth = 1 * u;
      ctx.strokeStyle = "rgba(56,189,248,0.3)";
      const [tl, tr, br, bl] = r.goal.mouth_quad;
      for (const f of [1 / 3, 2 / 3]) {
        line(ctx, ...lerp(tl, tr, f), ...lerp(bl, br, f));
        line(ctx, ...lerp(tl, bl, f), ...lerp(tr, br, f));
      }
    }
  }
  // Corner dots, and a tag over the crossbar.
  for (const [x, y] of r.net.quad) {
    ctx.beginPath();
    ctx.arc(x, y, 3.5 * u, 0, Math.PI * 2);
    ctx.fillStyle = "#ef4444";
    ctx.fill();
  }
  const [tl, tr] = r.net.quad;
  pill(ctx, "NET", (tl[0] + tr[0]) / 2, Math.min(tl[1], tr[1]) - 12 * u, 11 * u, "#ef4444", "#ffffff", "center");
}

const lerp = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

function poly(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
}

function polyline(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

// A rounded label.  ``align`` is where (x, y) sits on it: its "left" edge,
// its "right" edge or its "center"; it is kept inside the picture.
function pill(ctx, text, x, y, size, fill, ink, align = "left") {
  ctx.font = `700 ${size}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + size * 0.9;
  const h = size * 1.55;
  let x0 = align === "center" ? x - w / 2 : align === "right" ? x - w : x;
  x0 = Math.max(2, Math.min(x0, ctx.canvas.width - w - 2));
  y = Math.max(h / 2 + 2, Math.min(y, ctx.canvas.height - h / 2 - 2));
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x0, y - h / 2, w, h, h / 2);
  else ctx.rect(x0, y - h / 2, w, h);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.fillStyle = ink;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x0 + w / 2, y + size * 0.04);
  ctx.textBaseline = "alphabetic";
}

const OUTCOME_MARK = { on_net: "✓", post: "post", miss: "✗" };
const FLASH_S = 0.6;      // the zone it hit lights up this long
const CALLOUT_S = 1.5;    // and the newest shot's label is shown large this long

// Where on the picture a 3x3 zone of the mouth is: top_left .. low_right.
const ZONE_CELL = {
  top_left: [0, 0], top_mid: [0, 1], top_right: [0, 2],
  mid_left: [1, 0], mid_mid: [1, 1], mid_right: [1, 2],
  low_left: [2, 0], five_hole: [2, 1], low_right: [2, 2],
};

function zoneCell(quad, key) {
  const cell = ZONE_CELL[key];
  if (!cell) return null;
  const [row, col] = cell;
  const [tl, tr, br, bl] = quad;
  const at = (fx, fy) => lerp(lerp(tl, tr, fx), lerp(bl, br, fx), fy);
  return [at(col / 3, row / 3), at((col + 1) / 3, row / 3), at((col + 1) / 3, (row + 1) / 3), at(col / 3, (row + 1) / 3)];
}

// Every shot leaves its path on the picture, from where it was first seen to
// where it hit, with its number, result and speed; the newest is drawn
// strongest and the ones before it fade back.
function drawShots(ctx, r, frame, fps, u) {
  const shots = r.shots.map((shot, i) => ({ shot, track: (r.tracks ?? []).find((t) => t.shot_index === i) }));
  const landed = shots.filter(({ shot }) => frame >= shot.impact_frame);
  const latest = landed.length
    ? landed.reduce((a, b) => (b.shot.impact_frame > a.shot.impact_frame ? b : a)).shot
    : null;
  for (const { shot, track } of shots) {
    const from = shot.first_tracked_frame ?? shot.impact_frame;
    if (frame < from) continue;
    const done = frame >= shot.impact_frame;
    const color = done ? OUTCOME_COLOR[shot.outcome] ?? "#94a3b8" : "#ffffff";
    const pts = track ? track.points.filter((_, k) => track.frames[k] <= frame) : [];
    if (done) pts.push(shot.impact_image);
    ctx.globalAlpha = !done || shot === latest ? 1 : 0.4;

    // The path, over a dark edge so it shows on bright concrete too.
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = 5 * u;
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    polyline(ctx, pts);
    ctx.lineWidth = 2.5 * u;
    ctx.strokeStyle = color;
    polyline(ctx, pts);

    if (!done) {
      const head = pts[pts.length - 1];
      if (head) {
        ctx.beginPath();
        ctx.arc(head[0], head[1], 7 * u, 0, Math.PI * 2);
        ctx.lineWidth = 2 * u;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
      }
      continue;
    }

    const age = (frame - shot.impact_frame) / fps;
    if (age < FLASH_S && r.goal?.mouth_quad) {
      const cell = zoneCell(r.goal.mouth_quad, shot.zone_key);
      if (cell) {
        ctx.globalAlpha = 0.45 * (1 - age / FLASH_S);
        ctx.fillStyle = color;
        ctx.beginPath();
        cell.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = shot === latest ? 1 : 0.4;
      }
    }

    const [x, y] = shot.impact_image;
    ctx.beginPath();
    ctx.arc(x, y, 6 * u, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 2 * u;
    ctx.strokeStyle = "#0b1220";
    ctx.stroke();

    const n = sessionNo(state.clip, shot.index);
    const mark = OUTCOME_MARK[shot.outcome] ?? "";
    const mph = shot.speed ? ` ${Math.round(shot.speed.mph)} mph` : "";
    // The newest shot says everything; older ones keep just their number, so
    // a session's worth of marks does not bury the net in labels.
    const newest = shot === latest;
    const size = (newest && age < CALLOUT_S ? 20 : newest ? 13 : 10) * u;
    const text = newest ? `#${n} ${mark}${mph}` : `#${n}`;
    // To the right of the mark, or to its left if it would run off the picture.
    ctx.font = `700 ${size}px system-ui, sans-serif`;
    const fits = x + 10 * u + ctx.measureText(text).width + size <= ctx.canvas.width;
    pill(ctx, text, fits ? x + 10 * u : x - 10 * u, y - 14 * u, size, color, "#0b1220", fits ? "left" : "right");
  }
  ctx.globalAlpha = 1;
}

/* ------------------------------------------- the video with shots drawn on */

// The same marks, burned into a copy of the clip on the server (H.264, which
// any phone plays), to keep or send.
$("save-marked").addEventListener("click", saveMarked);

async function saveMarked() {
  const btn = $("save-marked");
  const status = $("marked-status");
  const clip = state.clip;
  btn.disabled = true;
  status.classList.remove("hidden");
  status.textContent = "Drawing the shots onto the video\u2026";
  try {
    const form = new FormData();
    form.append("clip", String(clip));
    let res = await fetch(`/api/jobs/${state.jobId}/marked`, { method: "POST", body: form });
    let st = await res.json();
    while (res.ok && st.status === "running") {
      status.textContent = `Drawing the shots onto the video\u2026 ${Math.round((st.progress ?? 0) * 100)}%`;
      await new Promise((r) => setTimeout(r, 1000));
      res = await fetch(`/api/jobs/${state.jobId}/marked?clip=${clip}`);
      st = await res.json();
    }
    if (!res.ok || st.status !== "done") throw new Error(st.error || st.detail || "it could not be made");
    const url = `/api/jobs/${state.jobId}/marked.mp4?clip=${clip}`;
    status.innerHTML = `Ready: <a href="${url}" download>download the video with the shots drawn on</a>.`;
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
    status.textContent = `The video with the shots drawn on could not be made: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

/* ------------------------------------------------------------- history */

// Every finished session is kept on the server. The upload screen shows how
// the latest compares with the ones before it, and any of them can be reopened.

const PROGRESS_ORDER = ["speed_mean", "accuracy_pct", "hit_pct"];

async function loadHistory() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    renderHistory(await res.json());
  } catch {
    // No history is not an error; the card just stays hidden.
  }
}

const fmtValue = (v) => `${Math.round(v)}`;

function fmtWhen(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) + " " +
    d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function deltaHtml(p) {
  if (p.delta == null) return "First session with this &mdash; the next one will show a trend.";
  const unit = p.unit === "%" ? " pts" : ` ${p.unit}`;
  const vs = `vs your previous ${p.baseline_sessions === 1 ? "session" : `${p.baseline_sessions} sessions`}`;
  const d = p.delta;
  if (Math.abs(d) < 0.5) return `About the same ${vs}`;
  const good = (d > 0) === p.up_is_good;
  const arrow = d > 0 ? "▲" : "▼";
  return `<span class="arrow ${good ? "good" : "bad"}" aria-hidden="true">${arrow}</span> ` +
    `${d > 0 ? "up" : "down"} ${Math.abs(d).toFixed(p.unit === "%" ? 0 : 1)}${unit} ${vs}`;
}

// One series per tile: the trend in a quiet gray, the latest session in the
// accent. Hover (or tap) a point for its session.
function sparkSvg(key, p, W) {
  const pts = p.series;
  if (pts.length < 2 || !(W > 40)) return "";
  const H = 44, PAD = 6;
  const vals = pts.map((x) => x.value);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  const X = (i) => PAD + (i * (W - 2 * PAD)) / (pts.length - 1);
  const Y = (v) => H - PAD - ((v - lo) * (H - 2 * PAD)) / (hi - lo);
  const line = pts.map((x, i) => `${X(i).toFixed(1)},${Y(x.value).toFixed(1)}`).join(" ");
  const step = (W - 2 * PAD) / (pts.length - 1);
  const dots = pts.map((x, i) =>
    `<circle class="pt${i === pts.length - 1 ? " latest" : ""}" cx="${X(i).toFixed(1)}" cy="${Y(x.value).toFixed(1)}" r="4"></circle>`
  ).join("");
  // Hover bands split the width between points and stay inside the tile, so
  // they never reach into the neighbouring tile's line.
  const hits = pts.map((x, i) => {
    const x0 = Math.max(0, X(i) - step / 2), x1 = Math.min(W, X(i) + step / 2);
    return `<rect class="hit" data-key="${key}" data-i="${i}" x="${x0.toFixed(1)}" y="0" width="${(x1 - x0).toFixed(1)}" height="${H}"></rect>`;
  }).join("");
  // Drawn at the tile's own width, so the dots stay round.
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img"
    aria-label="${escapeHtml(p.label)} over ${pts.length} sessions"><polyline class="line" points="${line}"></polyline>${dots}${hits}</svg>`;
}

function renderHistory(h) {
  const card = $("history");
  const sessions = h.sessions ?? [];
  card.classList.toggle("hidden", sessions.length === 0 || !$("results").classList.contains("hidden"));
  if (!sessions.length) return;
  state.history = h;

  $("progress-tiles").innerHTML = PROGRESS_ORDER.filter((k) => h.progress?.[k]).map((k) => {
    const p = h.progress[k];
    return `<div class="ptile">
      <div class="ptile-label">${escapeHtml(p.label)} &middot; latest session</div>
      <div class="ptile-value">${fmtValue(p.latest)}<span class="unit">${p.unit === "%" ? "%" : p.unit}</span></div>
      <div class="ptile-delta">${deltaHtml(p)}</div>
      <div class="spark-slot" data-key="${k}"></div>
    </div>`;
  }).join("");
  drawSparks();

  renderHistoryTable(sessions);
}

function drawSparks() {
  const h = state.history;
  if (!h) return;
  const tip = $("spark-tip");
  $("progress-tiles").querySelectorAll(".spark-slot").forEach((slot) => {
    const key = slot.dataset.key;
    slot.innerHTML = sparkSvg(key, h.progress[key], Math.floor(slot.getBoundingClientRect().width));
  });
  $("progress-tiles").querySelectorAll(".hit").forEach((r) => {
    const show = (e) => {
      const p = h.progress[r.dataset.key];
      const x = p.series[Number(r.dataset.i)];
      const unit = p.unit === "%" ? "%" : ` ${p.unit}`;
      tip.textContent = `${fmtWhen(x.created_at)} · ${fmtValue(x.value)}${unit} · ${x.shots} shot${x.shots === 1 ? "" : "s"}`;
      tip.style.left = `${e.clientX + 12}px`;
      tip.style.top = `${e.clientY - 34}px`;
      tip.classList.remove("hidden");
    };
    r.addEventListener("mousemove", show);
    r.addEventListener("click", show);
    r.addEventListener("mouseleave", () => tip.classList.add("hidden"));
  });
}

let sparkResize = null;
window.addEventListener("resize", () => {
  clearTimeout(sparkResize);
  sparkResize = setTimeout(drawSparks, 120);
});

function renderHistoryTable(sessions) {
  const rows = sessions.map((s) => {
    const target = s.target_label ? `${s.target_hits}/${s.target_shots} <span class="unc">${escapeHtml(s.target_label)}</span>` : "<span class=\"unc\">&ndash;</span>";
    const pct = s.accuracy_pct != null ? `${Math.round(s.accuracy_pct)}%` : "&ndash;";
    const mean = s.speed_mean != null ? s.speed_mean.toFixed(0) : "&ndash;";
    const max = s.speed_max != null ? s.speed_max.toFixed(0) : "&ndash;";
    return `<tr data-id="${s.id}">
      <td class="when">${fmtWhen(s.created_at)}</td>
      <td>${s.shots}${s.clips > 1 ? ` <span class="unc">in ${s.clips} clips</span>` : ""}</td>
      <td>${pct}</td>
      <td>${mean}</td>
      <td>${max}</td>
      <td>${target}</td>
      <td><button type="button" class="row-delete" data-id="${s.id}">Delete</button></td>
    </tr>`;
  }).join("");
  $("history-table").innerHTML =
    `<thead><tr><th>When</th><th>Shots</th><th>On net</th><th>Avg mph</th><th>Top mph</th><th>Target</th><th></th></tr></thead>
     <tbody>${rows}</tbody>`;

  $("history-table").querySelectorAll("tbody tr").forEach((tr) =>
    tr.addEventListener("click", (e) => {
      if (e.target.closest(".row-delete")) return;
      openSession(tr.dataset.id);
    }));
  // Deleting is permanent, so it takes a second click.
  $("history-table").querySelectorAll(".row-delete").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!b.classList.contains("armed")) {
        b.classList.add("armed");
        b.textContent = "Delete?";
        setTimeout(() => { b.classList.remove("armed"); b.textContent = "Delete"; }, 4000);
        return;
      }
      await fetch(`/api/jobs/${b.dataset.id}`, { method: "DELETE" });
      loadHistory();
    }));
}

async function openSession(id) {
  try {
    const job = await (await fetch(`/api/jobs/${id}`)).json();
    if (!job.result) throw new Error("That session has no results to show.");
    state.jobId = id;
    state.clip = 0;
    $("history").classList.add("hidden");
    render(job.result);
  } catch (err) {
    showError(err.message);
  }
}

loadHistory();
