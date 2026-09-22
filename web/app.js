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
  result: null,
  chartHits: [],   // click targets on the shot chart
  mark: { corners: [], img: null, info: null, frame: 0, hover: null, dragging: null },
};

/* ---------------------------------------------------------------- upload */

const fileInput = $("video-input");
const fileDrop = $("file-drop");

fileInput.addEventListener("change", () => {
  $("file-name").textContent = fileInput.files[0]?.name ?? "";
});

["dragenter", "dragover"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => { e.preventDefault(); fileDrop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  fileDrop.addEventListener(ev, (e) => { e.preventDefault(); fileDrop.classList.remove("drag"); })
);
fileDrop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    $("file-name").textContent = file.name;
  }
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;

  showError(null);
  $("submit-btn").disabled = true;
  $("progress").classList.remove("hidden");
  setProgress("Uploading", 0.02);

  const body = new FormData();
  body.append("video", file);
  const num = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  if (num("distance") !== null) body.append("shot_distance_ft", num("distance"));
  body.append("shooter_offset_ft", num("offset") ?? 0);
  if (num("hfov") !== null) body.append("hfov_deg", num("hfov"));
  if (num("fps") !== null) body.append("fps_override", num("fps"));
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
  state.mark = { corners: [], img: null, info: null, frame: 0, hover: null, dragging: null };
  $("upload-panel").classList.remove("hidden");
  $("submit-btn").disabled = false;
  $("progress").classList.add("hidden");
});

async function poll() {
  try {
    const job = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    setProgress(job.stage, job.progress);
    if (job.status === "done") {
      // No outline means every downstream number is unavailable, so offer the
      // one thing that always works: let the player point at the net.
      if (!job.result.net) return offerManualMarking(job.result);
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

function render(result) {
  state.result = result;
  $("upload-panel").classList.add("hidden");
  $("results").classList.remove("hidden");

  renderStats(result);
  renderTable(result);
  renderNotes(result);
  drawChart();

  const video = $("video");
  sizeOverlay();                     // the analysis already told us the dimensions
  video.src = `/api/jobs/${state.jobId}/video`;
  video.addEventListener("loadedmetadata", sizeOverlay, { once: true });
  video.addEventListener("error", () => {
    $("video-hint").textContent =
      "This browser cannot play the clip's codec, so the overlay is unavailable. " +
      "The shot chart and numbers below are unaffected.";
  }, { once: true });
  requestAnimationFrame(drawOverlay);
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
  // The camera distance follows from the marked net being the size it was said
  // to be. If the wrong rectangle was marked -- a backstop frame rather than
  // the goal -- this is the number that gives it away.
  if (r.camera?.position_in) {
    const ft = r.camera.position_in[2] / 12;
    cards.push({ value: `${ft.toFixed(0)} ft`, label: "camera distance",
                 sub: "does this look right?" });
  }
  $("stats").innerHTML = cards.map((c) => `
    <div>
      <div class="stat-value">${c.value}</div>
      <div class="stat-label">${c.label}</div>
      ${c.sub ? `<div class="stat-sub">${c.sub}</div>` : ""}
    </div>`).join("");
}

function renderTable(r) {
  const rows = r.shots.map((s) => {
    const speed = s.speed
      ? `${s.speed.mph.toFixed(1)} <span class="unc">${s.speed.uncertainty_mph != null ? `±${s.speed.uncertainty_mph.toFixed(1)}` : ""}</span>`
      : "–";
    const where = s.zone_label ?? s.miss_detail ?? s.outcome;
    const label = { on_net: "on net", post: "post", miss: "missed" }[s.outcome] ?? s.outcome;
    return `<tr data-frame="${s.impact_frame}">
      <td>${s.index + 1}</td>
      <td>${speed}</td>
      <td><span class="pill ${s.outcome}">${label}</span></td>
      <td>${where}</td>
      <td class="unc">${s.impact_goal_in[0] >= 0 ? "+" : ""}${s.impact_goal_in[0].toFixed(0)}", ${s.impact_goal_in[1].toFixed(0)}" up</td>
    </tr>`;
  }).join("");
  $("shot-table").innerHTML =
    `<thead><tr><th>#</th><th>Speed (mph)</th><th>Result</th><th>Where</th><th>Position</th></tr></thead>
     <tbody>${rows}</tbody>`;
  $("shot-table").querySelectorAll("tbody tr").forEach((tr) =>
    tr.addEventListener("click", () => seekToFrame(Number(tr.dataset.frame)))
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

function seekToFrame(frame) {
  const video = $("video");
  const fps = state.result?.video?.fps;
  if (!fps || !Number.isFinite(video.duration)) return;
  video.currentTime = Math.max(0, frame / fps - 0.15);
  video.pause();
  drawOverlay();
}

/* ------------------------------------------------------- marking the net */

async function offerManualMarking(result) {
  $("progress").classList.add("hidden");
  $("upload-panel").classList.add("hidden");
  $("mark-panel").classList.remove("hidden");

  const why = (result.warnings || []).find((w) => w.includes("goal")) ||
    "The goal could not be found automatically.";
  $("mark-intro").textContent = why + " Mark it once here and the clip will be re-read.";

  state.mark.info = await (await fetch(`/api/jobs/${state.jobId}/info`)).json();
  const slider = $("mark-frame");
  slider.max = Math.max(0, (state.mark.info.frame_count || 1) - 1);
  slider.value = Math.floor((state.mark.info.frame_count || 1) / 2);
  state.mark.frame = Number(slider.value);
  await loadMarkFrame();
}

function loadMarkFrame() {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      state.mark.img = img;
      const canvas = $("mark-canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      drawMark();
      resolve();
    };
    img.src = `/api/jobs/${state.jobId}/frame.png?frame=${state.mark.frame}`;
  });
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
    ctx.strokeStyle = "#38bdf8";
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (pts.length === 4) ctx.closePath();
    ctx.stroke();
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
  state.mark.corners.pop();
  $("mark-go").disabled = state.mark.corners.length !== 4;
  drawMark();
});

$("mark-go").addEventListener("click", async () => {
  const body = new FormData();
  body.append("net_quad", state.mark.corners.flat().map((v) => v.toFixed(1)).join(","));
  $("mark-go").disabled = true;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/reanalyze`, { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail ?? "could not re-analyze");
    $("mark-panel").classList.add("hidden");
    $("upload-panel").classList.remove("hidden");
    $("progress").classList.remove("hidden");
    setProgress("re-reading with your net", 0.05);
    pollAfterMark();
  } catch (err) {
    showError(err.message);
    $("mark-go").disabled = false;
  }
});

async function pollAfterMark() {
  try {
    const job = await (await fetch(`/api/jobs/${state.jobId}`)).json();
    setProgress(job.stage, job.progress);
    if (job.status === "done") return render(job.result);
    if (job.status === "error") throw new Error(job.error);
    setTimeout(pollAfterMark, POLL_MS);
  } catch (err) {
    showError(err.message);
  }
}

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

  // The pipe.
  const pipe = (goal.post_diameter_in ?? 2.375) * s;
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = pipe;
  ctx.beginPath();
  ctx.moveTo(X(-W / 2) - pipe / 2, Y(0));
  ctx.lineTo(X(-W / 2) - pipe / 2, Y(H) - pipe / 2);
  ctx.lineTo(X(W / 2) + pipe / 2, Y(H) - pipe / 2);
  ctx.lineTo(X(W / 2) + pipe / 2, Y(0));
  ctx.stroke();

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

    state.chartHits.push({ x: px, y: py, r: rad + 6, frame: shot.impact_frame });
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
  if (hit) seekToFrame(hit.frame);
});

$("show-zones").addEventListener("change", drawChart);

/* ------------------------------------------------------- video overlay */

function sizeOverlay() {
  const video = $("video");
  const canvas = $("overlay");
  const v = state.result?.video ?? {};
  canvas.width = video.videoWidth || v.width || 1280;
  canvas.height = video.videoHeight || v.height || 720;
}

window.addEventListener("resize", () => { sizeOverlay(); drawOverlay(); });

function drawOverlay() {
  const video = $("video");
  const canvas = $("overlay");
  const r = state.result;
  if (r && canvas.width) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const fps = r.video.fps || 30;
    const frame = Math.round((video.currentTime || 0) * fps);

    if (r.net?.quad) drawNet(ctx, r, $("show-zones").checked);
    drawTrails(ctx, r, frame);
    drawImpacts(ctx, r, frame);
  }
  requestAnimationFrame(drawOverlay);
}

function drawNet(ctx, r, zones) {
  const outer = r.net.quad;
  const mouth = r.goal?.mouth_quad;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#38bdf8";
  poly(ctx, outer);
  if (mouth) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(56,189,248,0.55)";
    poly(ctx, mouth);
    if (zones) {
      // Interpolate the mouth quad's edges to draw the grid in perspective.
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(56,189,248,0.3)";
      const [tl, tr, br, bl] = mouth;
      for (const f of [1 / 3, 2 / 3]) {
        line(ctx, ...lerp(tl, tr, f), ...lerp(bl, br, f));
        line(ctx, ...lerp(tl, bl, f), ...lerp(tr, br, f));
      }
    }
  }
}

const lerp = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

function poly(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
}

function drawTrails(ctx, r, frame) {
  const TRAIL = 30;
  for (const track of r.tracks ?? []) {
    const pts = track.points.filter((_, i) => {
      const f = track.frames[i];
      return f <= frame && f > frame - TRAIL;
    });
    for (let i = 1; i < pts.length; i++) {
      const age = (pts.length - i) / pts.length;
      ctx.strokeStyle = `rgba(255,255,255,${(1 - age * 0.8).toFixed(2)})`;
      ctx.lineWidth = 3;
      line(ctx, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]);
    }
    const head = track.points[track.frames.indexOf(frame)];
    if (head) {
      ctx.beginPath();
      ctx.arc(head[0], head[1], 8, 0, Math.PI * 2);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function drawImpacts(ctx, r, frame) {
  for (const shot of r.shots) {
    if (frame < shot.impact_frame) continue;
    const [x, y] = shot.impact_image;
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fillStyle = OUTCOME_COLOR[shot.outcome] ?? "#94a3b8";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#0b1220";
    ctx.stroke();

    const tag = shot.speed ? `${shot.index + 1}  ${Math.round(shot.speed.mph)} mph` : `${shot.index + 1}`;
    ctx.font = "600 18px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(0,0,0,0.75)";
    ctx.strokeText(tag, x + 16, y - 10);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(tag, x + 16, y - 10);
  }
}
