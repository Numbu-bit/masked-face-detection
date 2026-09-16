/**
 * Masked Face Detection - browser app.
 *
 * Flow
 *   1. GET /api/info          -> class names, colours, thresholds, model URL
 *   2. fetch model.onnx       -> onnxruntime-web session (WebGPU -> WASM fallback)
 *   3. webcam / upload frame  -> letterbox on an offscreen canvas -> tensor
 *   4. session.run            -> decodeOutput() (yolo.js) -> draw boxes + banner
 *
 * "Server" backend mode POSTs the frame to /api/detect instead of step 3-4.
 */

import { letterboxParams, rgbaToCHW, decodeOutput, summarise, bannerText } from "./yolo.js";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), photo: $("photo"), overlay: $("overlay"), dropzone: $("dropzone"), webcamIdle: $("webcam-idle"),
  bannerText: $("banner-text"), perf: $("perf"), statusDot: $("status-dot"), statusText: $("status-text"),
  progressWrap: $("progress-wrap"), progressBar: $("progress-bar"), legend: $("legend"),
  btnStart: $("btn-start"), btnStop: $("btn-stop"), btnFlip: $("btn-flip"), btnSnapshot: $("btn-snapshot"),
  btnDetect: $("btn-detect"), btnClear: $("btn-clear"), fileInput: $("file-input"),
  conf: $("conf"), confValue: $("conf-value"), iou: $("iou"), iouValue: $("iou-value"),
  detList: $("det-list"), detJson: $("det-json"), epLabel: $("ep-label"), toast: $("toast"),
  modelSize: $("model-size"), modelSizeWrap: $("model-size-wrap"),
};

const state = {
  info: null,            // /api/info payload
  session: null,         // ort.InferenceSession
  inputName: null, outputName: null, inputW: 640, inputH: 640,
  models: {},            // {"416": "models/model.onnx", "320": "models/model_320.onnx"}
  modelSize: null,
  mode: "webcam",        // "webcam" | "upload"
  backend: "browser",    // "browser" | "server"
  stream: null, facingMode: "user", running: false, busy: false,
  lastDets: [], lastCounts: null,
  fpsSamples: [],
};

const ctx = els.overlay.getContext("2d");
const work = document.createElement("canvas");      // letterbox scratch canvas
const workCtx = work.getContext("2d", { willReadFrequently: true });

// --------------------------------------------------------------------------- UI helpers
function setStatus(text, color) {
  els.statusText.textContent = text;
  els.statusDot.className = `inline-block h-2.5 w-2.5 rounded-full ${color}`;
}
function toast(msg, ms = 4000) {
  els.toast.textContent = msg;
  els.toast.style.opacity = "1";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (els.toast.style.opacity = "0"), ms);
}
function labelFor(d) {
  return `${state.info?.box_label || d.className} ${d.confidence.toFixed(2)}`;
}
function colorFor(name) {
  const c = state.info?.class_colors_rgb?.[name] || [255, 255, 255];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function renderLegend() {
  els.legend.innerHTML = "";
  for (const name of state.info.class_names) {
    const li = document.createElement("li");
    li.className = "flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-800/60 px-2.5 py-1";
    li.innerHTML = `<span class="inline-block h-2.5 w-2.5 rounded-sm" style="background:${colorFor(name)}"></span>${name}`;
    els.legend.appendChild(li);
  }
}
function renderDetections(dets, counts, extra) {
  els.bannerText.textContent = bannerText(counts);
  els.detList.innerHTML = "";
  if (!dets.length) {
    els.detList.innerHTML = '<li class="text-slate-500">No faces detected.</li>';
  } else {
    for (const d of dets) {
      const li = document.createElement("li");
      li.className = "flex items-center justify-between gap-2";
      li.innerHTML = `<span class="flex items-center gap-2"><span class="inline-block h-2.5 w-2.5 rounded-sm" style="background:${colorFor(d.className)}"></span>${d.className}</span><span class="font-mono text-slate-400">${d.confidence.toFixed(2)}</span>`;
      els.detList.appendChild(li);
    }
  }
  els.detJson.textContent = JSON.stringify({
    summary: counts,
    detections: dets.map((d) => ({ class_id: d.classId, class_name: d.className, confidence: +d.confidence.toFixed(4), bbox_xyxy: d.bbox.map((v) => +v.toFixed(1)) })),
    ...extra,
  }, null, 2);
}

// --------------------------------------------------------------------------- Drawing (matches src/utils.draw_detections)
function drawOverlay(dets, srcW, srcH) {
  const cw = els.overlay.clientWidth, ch = els.overlay.clientHeight;
  if (els.overlay.width !== cw || els.overlay.height !== ch) { els.overlay.width = cw; els.overlay.height = ch; }
  ctx.clearRect(0, 0, cw, ch);
  // object-contain geometry of the underlying video/img inside the overlay
  const s = Math.min(cw / srcW, ch / srcH);
  const ox = (cw - srcW * s) / 2, oy = (ch - srcH * s) / 2;
  const lw = Math.max(2, Math.round(Math.min(cw, ch) / 240));
  ctx.font = `${Math.max(12, Math.round(Math.min(cw, ch) / 32))}px ui-monospace, monospace`;
  ctx.textBaseline = "top";
  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox.map((v, i) => (i % 2 === 0 ? ox + v * s : oy + v * s));
    const color = colorFor(d.className);
    ctx.lineWidth = lw; ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = labelFor(d);
    const tw = ctx.measureText(label).width + 8, th = parseInt(ctx.font, 10) + 6;
    const ty = Math.max(y1 - th, 0);
    ctx.fillStyle = color; ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x1 + 4, ty + 3);
  }
}

// --------------------------------------------------------------------------- Model
async function loadInfo() {
  // Web-service deployment: the FastAPI backend describes itself at api/info.
  // Static deployment (no backend): fall back to config.json written by scripts/prepare_web_model.py.
  let info = null;
  try {
    const r = await fetch("api/info", { headers: { Accept: "application/json" } });
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) info = await r.json();
  } catch (_) { /* no backend */ }
  if (!info) {
    const r = await fetch("config.json");
    if (!r.ok) throw new Error("neither api/info nor config.json is available");
    info = await r.json();
    info.static = true;
  }
  state.info = info;
  // Available model variants keyed by input size. Older config/info payloads only have model_url.
  state.models = info.models && Object.keys(info.models).length
    ? info.models : { [String((info.model_input || [416])[0])]: info.model_url };
  if (info.static) {
    // No backend to enumerate files: probe the conventional names so a newly added
    // models/model_<size>.onnx shows up without editing config.json.
    await Promise.all([640, 416, 320].filter((sz) => !state.models[sz]).map(async (sz) => {
      try { const r = await fetch(`models/model_${sz}.onnx`, { method: "HEAD" }); if (r.ok) state.models[sz] = `models/model_${sz}.onnx`; } catch (_) {}
    }));
  }
  const sizes = Object.keys(state.models).map(Number).sort((a, b) => b - a);
  state.modelSize = sizes.includes(+localStorage.getItem("mfd.modelSize")) ? +localStorage.getItem("mfd.modelSize") : sizes[0];
  els.modelSize.innerHTML = "";
  for (const sz of sizes) {
    const o = document.createElement("option");
    o.value = sz; o.textContent = `${sz} px${sz === sizes[0] ? " (most accurate)" : sz === sizes[sizes.length - 1] && sizes.length > 1 ? " (fastest)" : ""}`;
    els.modelSize.appendChild(o);
  }
  els.modelSize.value = state.modelSize;
  els.modelSizeWrap.classList.toggle("hidden", sizes.length < 2);
  if (info.static) {
    // No server -> in-browser inference only.
    const serverRadio = document.querySelector("input[name=backend][value=server]");
    serverRadio.disabled = true;
    serverRadio.parentElement.classList.add("opacity-40");
    serverRadio.parentElement.title = "Not available on a static deployment";
  }
  els.conf.value = state.info.confidence_threshold; els.confValue.textContent = (+els.conf.value).toFixed(2);
  els.iou.value = state.info.iou_threshold; els.iouValue.textContent = (+els.iou.value).toFixed(2);
  renderLegend();
  if (!state.info.model_loaded) throw new Error(`Server has no model: ${state.info.model_error}`);
}

async function fetchModelWithProgress(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`model download failed (${r.status})`);
  const total = +r.headers.get("content-length") || 0;
  const reader = r.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (total) els.progressBar.style.width = `${Math.round((got / total) * 90)}%`;
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function loadModel() {
  els.progressWrap.classList.remove("hidden"); els.progressBar.style.width = "0%";
  const url = state.models[state.modelSize];
  setStatus(`Downloading model (${state.modelSize} px)…`, "bg-amber-400 animate-pulse");
  const bytes = await fetchModelWithProgress(url);
  setStatus("Initialising runtime…", "bg-amber-400 animate-pulse");
  // Self-hosted runtime (web/static/vendor/ort). Threads only work when the page is
  // cross-origin isolated (COOP/COEP headers) - otherwise ORT silently uses 1 thread.
  ort.env.wasm.wasmPaths = "vendor/ort/";
  const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  ort.env.wasm.numThreads = threads;
  const tries = navigator.gpu ? [["webgpu"], ["wasm"]] : [["wasm"]];
  let lastErr, ep = null;
  if (state.session) { try { await state.session.release(); } catch (_) {} state.session = null; }
  for (const providers of tries) {
    try {
      state.session = await ort.InferenceSession.create(bytes, { executionProviders: providers, graphOptimizationLevel: "all" });
      ep = providers[0];
      break;
    } catch (e) { lastErr = e; console.warn(`EP ${providers[0]} failed:`, e); }
  }
  if (!state.session) throw lastErr;
  els.epLabel.textContent = ep === "webgpu" ? "(WebGPU)" : `(WASM, ${threads} thread${threads > 1 ? "s" : ""}${threads === 1 && (navigator.hardwareConcurrency || 2) > 1 ? " - enable COOP/COEP headers for more" : ""})`;
  state.inputName = state.session.inputNames[0];
  state.outputName = state.session.outputNames[0];
  state.inputW = state.inputH = +state.modelSize;
  work.width = state.inputW; work.height = state.inputH;
  els.progressBar.style.width = "100%";
  setTimeout(() => els.progressWrap.classList.add("hidden"), 600);
  setStatus(`Model ready · ${state.modelSize} px · ${ep === "webgpu" ? "WebGPU" : "WASM"}`, "bg-emerald-500");
  els.btnStart.disabled = false;
  // warm-up: first run compiles kernels (WebGPU) and would otherwise show as a long first frame
  try { await state.session.run({ [state.inputName]: new ort.Tensor("float32", new Float32Array(3 * state.inputW * state.inputH), [1, 3, state.inputH, state.inputW]) }); } catch (_) {}
}

/** Run the model in-browser on any drawable source (video / img / canvas). */
async function detectInBrowser(source, srcW, srcH) {
  const { ratio, dx, dy, nw, nh } = letterboxParams(srcW, srcH, state.inputW, state.inputH);
  workCtx.fillStyle = "rgb(114,114,114)"; workCtx.fillRect(0, 0, state.inputW, state.inputH);
  workCtx.drawImage(source, 0, 0, srcW, srcH, dx, dy, nw, nh);
  const rgba = workCtx.getImageData(0, 0, state.inputW, state.inputH).data;
  const tensor = new ort.Tensor("float32", rgbaToCHW(rgba, state.inputW, state.inputH), [1, 3, state.inputH, state.inputW]);
  const t0 = performance.now();
  const out = await state.session.run({ [state.inputName]: tensor });
  const ms = performance.now() - t0;
  const o = out[state.outputName];
  const dets = decodeOutput(o.data, o.dims, {
    confThresh: +els.conf.value, iouThresh: +els.iou.value, ratio, dx, dy, origW: srcW, origH: srcH,
    classNames: state.info.class_names,
  });
  return { dets, ms, where: "browser" };
}

/** Send a JPEG of the source to the server API. */
async function detectOnServer(source, srcW, srcH) {
  const c = document.createElement("canvas"); c.width = srcW; c.height = srcH;
  c.getContext("2d").drawImage(source, 0, 0, srcW, srcH);
  const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.85));
  const fd = new FormData(); fd.append("file", blob, "frame.jpg");
  const t0 = performance.now();
  const r = await fetch(`api/detect?conf=${els.conf.value}&iou=${els.iou.value}`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`api/detect -> ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const j = await r.json();
  const dets = j.detections.map((d) => ({ bbox: d.bbox_xyxy, confidence: d.confidence, classId: d.class_id, className: d.class_name }));
  return { dets, ms: performance.now() - t0, where: "server", serverMs: j.inference_ms };
}

async function runDetection(source, srcW, srcH) {
  const res = state.backend === "server" || !state.session
    ? await detectOnServer(source, srcW, srcH)
    : await detectInBrowser(source, srcW, srcH);
  const counts = summarise(res.dets, state.info.class_names);
  state.lastDets = res.dets; state.lastCounts = counts;
  drawOverlay(res.dets, srcW, srcH);
  renderDetections(res.dets, counts, { inference_ms: +res.ms.toFixed(1), backend: res.where });
  return res;
}

// --------------------------------------------------------------------------- Webcam
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) { toast("This browser does not support camera access."); return; }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: state.facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    toast(e.name === "NotAllowedError" ? "Camera permission denied. Allow access in the address bar and try again." : `Camera error: ${e.message}`);
    return;
  }
  els.video.srcObject = state.stream;
  await els.video.play();
  els.webcamIdle.classList.add("hidden");
  els.btnStart.classList.add("hidden");
  for (const b of [els.btnStop, els.btnFlip, els.btnSnapshot]) b.classList.remove("hidden");
  state.running = true; state.fpsSamples = [];
  loop();
}
function stopCamera() {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null; els.video.srcObject = null;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.webcamIdle.classList.remove("hidden");
  els.btnStart.classList.remove("hidden");
  for (const b of [els.btnStop, els.btnFlip, els.btnSnapshot]) b.classList.add("hidden");
  els.perf.textContent = "";
}
async function loop() {
  if (!state.running) return;
  if (!state.busy && els.video.readyState >= 2) {
    state.busy = true;
    const t0 = performance.now();
    try {
      const res = await runDetection(els.video, els.video.videoWidth, els.video.videoHeight);
      const dt = performance.now() - t0;
      state.fpsSamples.push(dt); if (state.fpsSamples.length > 20) state.fpsSamples.shift();
      const avg = state.fpsSamples.reduce((a, b) => a + b, 0) / state.fpsSamples.length;
      els.perf.textContent = `${(1000 / avg).toFixed(1)} FPS · ${res.ms.toFixed(0)} ms ${res.where}${res.serverMs ? ` (model ${res.serverMs} ms)` : ""}`;
    } catch (e) {
      console.error(e); toast(`Detection failed: ${e.message}`); stopCamera(); return;
    } finally { state.busy = false; }
  }
  requestAnimationFrame(loop);
}
function downloadSnapshot() {
  const w = els.video.videoWidth, h = els.video.videoHeight;
  if (!w) return;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.drawImage(els.video, 0, 0);
  // redraw boxes at native resolution
  const lw = Math.max(2, Math.round(Math.min(w, h) / 240)); g.lineWidth = lw;
  g.font = `${Math.max(14, Math.round(Math.min(w, h) / 32))}px monospace`; g.textBaseline = "top";
  for (const d of state.lastDets) {
    const [x1, y1, x2, y2] = d.bbox; const color = colorFor(d.className);
    g.strokeStyle = color; g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = labelFor(d); const tw = g.measureText(label).width + 8, th = parseInt(g.font, 10) + 6;
    g.fillStyle = color; g.fillRect(x1, Math.max(y1 - th, 0), tw, th); g.fillStyle = "#fff"; g.fillText(label, x1 + 4, Math.max(y1 - th, 0) + 3);
  }
  g.fillStyle = "rgba(0,0,0,.7)"; g.fillRect(0, 0, w, 32); g.fillStyle = "#fff"; g.fillText(bannerText(state.lastCounts || {}), 8, 8);
  const a = document.createElement("a"); a.href = c.toDataURL("image/jpeg", 0.92); a.download = `masked-face-${Date.now()}.jpg`; a.click();
}

// --------------------------------------------------------------------------- Upload mode
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[role=tab]").forEach((b) => b.classList.toggle("tab-active", b.dataset.mode === mode));
  const webcam = mode === "webcam";
  if (!webcam) stopCamera();
  els.video.classList.toggle("hidden", !webcam);
  els.webcamIdle.classList.toggle("hidden", !webcam);
  els.btnStart.classList.toggle("hidden", !webcam);
  els.dropzone.classList.toggle("hidden", webcam); els.dropzone.classList.toggle("flex", !webcam);
  els.photo.classList.add("hidden"); els.btnDetect.classList.add("hidden"); els.btnClear.classList.add("hidden");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.perf.textContent = "";
}
function loadImageFile(file) {
  if (!file || !file.type.startsWith("image/")) { toast("Please choose an image file."); return; }
  const url = URL.createObjectURL(file);
  els.photo.onload = async () => {
    URL.revokeObjectURL(url);
    els.dropzone.classList.add("hidden"); els.dropzone.classList.remove("flex");
    els.photo.classList.remove("hidden");
    els.btnDetect.classList.remove("hidden"); els.btnClear.classList.remove("hidden");
    await detectPhoto();
  };
  els.photo.src = url;
}
async function detectPhoto() {
  if (!els.photo.naturalWidth) return;
  els.btnDetect.disabled = true;
  try {
    const res = await runDetection(els.photo, els.photo.naturalWidth, els.photo.naturalHeight);
    els.perf.textContent = `${res.ms.toFixed(0)} ms ${res.where}`;
  } catch (e) { console.error(e); toast(`Detection failed: ${e.message}`); }
  finally { els.btnDetect.disabled = false; }
}

// --------------------------------------------------------------------------- Wiring
document.querySelectorAll("[role=tab]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
els.btnStart.addEventListener("click", startCamera);
els.btnStop.addEventListener("click", stopCamera);
els.btnFlip.addEventListener("click", async () => { state.facingMode = state.facingMode === "user" ? "environment" : "user"; stopCamera(); await startCamera(); });
els.btnSnapshot.addEventListener("click", downloadSnapshot);
els.btnDetect.addEventListener("click", detectPhoto);
els.btnClear.addEventListener("click", () => setMode("upload"));
els.fileInput.addEventListener("change", (e) => loadImageFile(e.target.files[0]));
for (const ev of ["dragenter", "dragover"]) els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add("border-emerald-500"); });
for (const ev of ["dragleave", "drop"]) els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove("border-emerald-500"); });
els.dropzone.addEventListener("drop", (e) => loadImageFile(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f && state.mode === "upload") loadImageFile(f); });
els.conf.addEventListener("input", () => { els.confValue.textContent = (+els.conf.value).toFixed(2); if (state.mode === "upload") detectPhoto(); });
els.iou.addEventListener("input", () => { els.iouValue.textContent = (+els.iou.value).toFixed(2); if (state.mode === "upload") detectPhoto(); });
els.modelSize.addEventListener("change", async () => {
  state.modelSize = +els.modelSize.value;
  try { localStorage.setItem("mfd.modelSize", state.modelSize); } catch (_) {}
  const wasRunning = state.running;
  if (wasRunning) stopCamera();
  els.btnStart.disabled = true;
  try { await loadModel(); } catch (e) { console.error(e); toast(`Could not load model: ${e.message}`); }
  if (wasRunning) startCamera(); else if (state.mode === "upload") detectPhoto();
});
document.querySelectorAll("input[name=backend]").forEach((r) => r.addEventListener("change", (e) => { state.backend = e.target.value; state.fpsSamples = []; if (state.mode === "upload") detectPhoto(); }));
window.addEventListener("resize", () => { if (state.lastDets.length && !state.running) { const s = state.mode === "webcam" ? els.video : els.photo; drawOverlay(state.lastDets, s.videoWidth || s.naturalWidth, s.videoHeight || s.naturalHeight); } });

// --------------------------------------------------------------------------- Boot
(async () => {
  try {
    await loadInfo();
    try {
      await loadModel();
    } catch (e) {
      console.error(e);
      if (state.info.static) throw new Error(`in-browser runtime failed (${e.message}) and this deployment has no server API`);
      // Browser inference unavailable (old browser, blocked WASM) - fall back to the API transparently.
      state.session = null; state.backend = "server";
      document.querySelector("input[name=backend][value=server]").checked = true;
      document.querySelector("input[name=backend][value=browser]").disabled = true;
      els.progressWrap.classList.add("hidden");
      setStatus("Using server API (browser runtime unavailable)", "bg-amber-400");
      els.btnStart.disabled = false;
      toast("In-browser runtime failed - falling back to the server API.");
    }
  } catch (e) {
    console.error(e);
    setStatus("Model unavailable", "bg-rose-500");
    els.progressWrap.classList.add("hidden");
    toast(`Cannot start: ${e.message}`, 10000);
  }
})();
