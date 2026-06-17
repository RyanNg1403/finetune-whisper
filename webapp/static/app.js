// webapp/static/app.js — mic capture -> 16k WAV -> /api/transcribe -> A/B render
const $ = (id) => document.getElementById(id);
const TARGET_SR = 16000;

const state = { recording: false, ctx: null, stream: null, proc: null, analyser: null,
  chunks: [], raf: 0, t0: 0, timer: 0, busy: false, keywords: [] };

// ---------- user-added Granite keywords ----------
function addKeywords(raw) {
  raw.split(",").map((s) => s.trim()).filter(Boolean).forEach((k) => {
    if (!state.keywords.some((x) => x.toLowerCase() === k.toLowerCase())) state.keywords.push(k);
  });
  renderChips();
}
function renderChips() {
  const c = $("kwChips");
  c.innerHTML = state.keywords.map((k, i) =>
    `<span class="kwchip">${esc(k)}<button class="x" data-i="${i}" aria-label="remove ${esc(k)}">×</button></span>`).join("");
  c.querySelectorAll("button.x").forEach((b) =>
    b.addEventListener("click", () => { state.keywords.splice(+b.dataset.i, 1); renderChips(); }));
}

// ---------- boot ----------
async function boot() {
  try {
    const r = await fetch("/api/checkpoints");
    const { checkpoints, device, keywords } = await r.json();
    $("deviceLabel").textContent = `device · ${device}`;
    $("deviceDot").classList.add("ok");
    const sel = $("ckptSel");
    sel.innerHTML = "";
    if (!checkpoints.length) {
      sel.innerHTML = "<option>no checkpoints found</option>";
    } else {
      for (const c of checkpoints) {
        const o = document.createElement("option");
        o.value = c.name; o.textContent = c.label;
        sel.appendChild(o);
      }
    }
    // read-only Granite keyword-biasing list: header is the only selectable option,
    // every term is a disabled (viewable, non-selectable) option.
    const kw = keywords || [];
    $("kwSelect").innerHTML =
      `<option selected>▾ ${kw.length} dataset terms (read-only)</option>` +
      kw.map((t) => `<option disabled>${esc(t)}</option>`).join("");
  } catch (e) {
    $("deviceLabel").textContent = "backend offline";
  }
  // suggestion chips — vocabulary that exercises the finetune
  const tries = ["DeepSeek", "Qwen Coder", "Groq", "GGUF", "vLLM", "SGLang", "Cerebras",
    "llama.cpp", "RoPE", "QLoRA", "Stagehand", "Claude Fable 5"];
  $("tryChips").innerHTML = tries.map((t) => `<span class="t">${t}</span>`).join("");
  $("recBtn").addEventListener("click", () => (state.recording ? stop() : start()));
  // add-your-own Granite keywords: Enter or comma commits; blur commits a pending word too
  const kwIn = $("kwInput");
  kwIn.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addKeywords(kwIn.value); kwIn.value = ""; }
  });
  kwIn.addEventListener("blur", () => { if (kwIn.value.trim()) { addKeywords(kwIn.value); kwIn.value = ""; } });
}

// ---------- recording ----------
async function start() {
  if (state.busy) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    $("hint").textContent = "mic denied"; return;
  }
  state.ctx = new (window.AudioContext || window.webkitAudioContext)();
  const src = state.ctx.createMediaStreamSource(state.stream);
  state.analyser = state.ctx.createAnalyser();
  state.analyser.fftSize = 2048;
  src.connect(state.analyser);

  state.proc = state.ctx.createScriptProcessor(4096, 1, 1);
  state.chunks = [];
  state.proc.onaudioprocess = (e) => {
    state.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  const mute = state.ctx.createGain(); mute.gain.value = 0;   // route capture, no feedback
  src.connect(state.proc); state.proc.connect(mute); mute.connect(state.ctx.destination);

  state.recording = true;
  $("recBtn").setAttribute("aria-pressed", "true");
  $("recBtn").querySelector(".rec-label").textContent = "tap to stop";
  $("hint").textContent = "listening";
  state.t0 = performance.now();
  state.timer = setInterval(tick, 200);
  drawWave();
}

async function stop() {
  if (!state.recording) return;
  state.recording = false;
  clearInterval(state.timer);
  cancelAnimationFrame(state.raf);
  $("recBtn").setAttribute("aria-pressed", "false");
  $("recBtn").querySelector(".rec-label").textContent = "hold / tap to speak";
  $("hint").textContent = "transcribing";

  const inRate = state.ctx.sampleRate;
  state.proc.disconnect(); state.analyser.disconnect();
  state.stream.getTracks().forEach((t) => t.stop());
  await state.ctx.close();

  const flat = flatten(state.chunks);
  clearWave();
  if (flat.length < inRate * 0.25) { $("hint").textContent = "too short"; return; }
  const wav = encodeWAV(downsample(flat, inRate, TARGET_SR), TARGET_SR);
  await send(wav);
}

function tick() {
  const s = Math.floor((performance.now() - state.t0) / 1000);
  $("timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// ---------- audio dsp ----------
function flatten(chunks) {
  const len = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(len); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function downsample(buf, inRate, outRate) {
  if (outRate >= inRate) return buf;
  const ratio = inRate / outRate;
  const outLen = Math.round(buf.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio), end = Math.min(Math.floor((i + 1) * ratio), buf.length);
    let sum = 0, n = 0;
    for (let j = start; j < end; j++) { sum += buf[j]; n++; }
    out[i] = n ? sum / n : 0;
  }
  return out;
}
function encodeWAV(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  ws(36, "data"); v.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ---------- waveform ----------
function drawWave() {
  const cv = $("wave"), g = cv.getContext("2d");
  const data = new Uint8Array(state.analyser.fftSize);
  const loop = () => {
    state.raf = requestAnimationFrame(loop);
    state.analyser.getByteTimeDomainData(data);
    g.clearRect(0, 0, cv.width, cv.height);
    g.lineWidth = 2; g.strokeStyle = "#f2a83b"; g.beginPath();
    const slice = cv.width / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * (cv.height / 2);
      i ? g.lineTo(i * slice, y) : g.moveTo(0, y);
    }
    g.stroke();
  };
  loop();
}
function clearWave() {
  const cv = $("wave"); cv.getContext("2d").clearRect(0, 0, cv.width, cv.height);
}

// ---------- transcribe + render ----------
async function send(wavBlob) {
  state.busy = true;
  ["panel-base", "panel-ft", "panel-granite"].forEach((p) => $(p).classList.add("busy"));
  $("hint").textContent = "transcribing (Granite loads on first use)…";
  const ckpt = encodeURIComponent($("ckptSel").value);
  const kw = encodeURIComponent(state.keywords.join(","));
  try {
    const r = await fetch(`/api/transcribe?ckpt=${ckpt}&kw=${kw}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: wavBlob });
    const j = await r.json();
    if (j.error) { $("hint").textContent = j.error; return; }
    render("base", j.baseline);
    render("ft", j.finetuned);
    $("ft-tag").textContent = j.finetuned.checkpoint;
    if (j.granite && j.granite.error) {
      $("text-granite").innerHTML = `<span class="placeholder">Granite unavailable: ${esc(j.granite.error)}</span>`;
      $("hits-granite").textContent = ""; $("ms-granite").textContent = "";
    } else if (j.granite) {
      render("granite", j.granite, state.keywords);   // also highlight the user's own keywords
    }
    $("hint").textContent = "done";
  } catch (e) {
    $("hint").textContent = "request failed";
  } finally {
    state.busy = false;
    ["panel-base", "panel-ft", "panel-granite"].forEach((p) => $(p).classList.remove("busy"));
  }
}

function esc(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function reEsc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function highlight(text, marks) {
  let html = esc(text);
  // longer terms first so multi-word names win over their substrings
  for (const { term, cs, cls } of [...marks].sort((a, b) => b.term.length - a.term.length)) {
    // homophones (cs) highlight case-sensitively; ordinary terms case-insensitively.
    // cls "kw" tints the user's own keywords cyan; default (green) is a dataset-term hit.
    const re = new RegExp(`(?<![A-Za-z0-9])(${reEsc(esc(term))})(?![A-Za-z0-9])`, cs ? "g" : "gi");
    html = html.replace(re, cls ? `<mark class="${cls}">$1</mark>` : "<mark>$1</mark>");
  }
  return html;
}

function render(which, d, userKeywords = []) {
  // dataset terms the model spelled correctly (green) + the user's own keywords (cyan, Granite only)
  const marks = d.terms.filter((x) => x.correct).map((x) => ({ term: x.term, cs: x.cs, cls: "" }));
  const datasetLower = new Set(d.terms.map((x) => x.term.toLowerCase()));
  for (const k of userKeywords) {
    if (k && !datasetLower.has(k.toLowerCase())) marks.push({ term: k, cs: false, cls: "kw" });
  }
  const missed = d.terms.filter((x) => !x.correct).map((x) => x.term);
  $(`text-${which}`).innerHTML = d.text ? highlight(d.text, marks)
    : '<span class="placeholder">(silence)</span>';
  const hits = $(`hits-${which}`);
  hits.textContent = `${d.hits}/${d.total} terms`;
  hits.classList.toggle("full", d.total > 0 && d.hits === d.total);
  hits.classList.toggle("zero", d.total > 0 && d.hits === 0);
  $(`ms-${which}`).textContent = `${d.latency_ms} ms`;
  $(`missed-${which}`).innerHTML = missed.map((t) => `<span class="chip">${esc(t)}</span>`).join("");
}

boot();
