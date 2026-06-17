// webapp/static/app.js — mic capture -> 16k WAV -> /api/transcribe -> A/B render
const $ = (id) => document.getElementById(id);
const TARGET_SR = 16000;

const state = { recording: false, ctx: null, stream: null, proc: null, analyser: null,
  chunks: [], raf: 0, t0: 0, timer: 0, busy: false };

// ---------- boot ----------
async function boot() {
  try {
    const r = await fetch("/api/checkpoints");
    const { checkpoints, device } = await r.json();
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
  } catch (e) {
    $("deviceLabel").textContent = "backend offline";
  }
  // suggestion chips — vocabulary that exercises the finetune
  const tries = ["DeepSeek", "Qwen Coder", "Groq", "GGUF", "vLLM", "SGLang", "Cerebras",
    "llama.cpp", "RoPE", "QLoRA", "Stagehand", "Claude Fable 5"];
  $("tryChips").innerHTML = tries.map((t) => `<span class="t">${t}</span>`).join("");
  $("recBtn").addEventListener("click", () => (state.recording ? stop() : start()));
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
  $("panel-base").classList.add("busy"); $("panel-ft").classList.add("busy");
  const ckpt = encodeURIComponent($("ckptSel").value);
  try {
    const r = await fetch(`/api/transcribe?ckpt=${ckpt}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: wavBlob });
    const j = await r.json();
    if (j.error) { $("hint").textContent = j.error; return; }
    render("base", j.baseline);
    render("ft", j.finetuned);
    $("ft-tag").textContent = j.finetuned.checkpoint;
    $("hint").textContent = "done";
  } catch (e) {
    $("hint").textContent = "request failed";
  } finally {
    state.busy = false;
    $("panel-base").classList.remove("busy"); $("panel-ft").classList.remove("busy");
  }
}

function esc(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function reEsc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

function highlight(text, correctTerms) {
  let html = esc(text);
  // longer terms first so multi-word names win over their substrings
  for (const { term, cs } of [...correctTerms].sort((a, b) => b.term.length - a.term.length)) {
    // homophones (cs) highlight case-sensitively; ordinary terms case-insensitively,
    // matching how the backend credited the hit.
    const re = new RegExp(`(?<![A-Za-z0-9])(${reEsc(esc(term))})(?![A-Za-z0-9])`, cs ? "g" : "gi");
    html = html.replace(re, "<mark>$1</mark>");
  }
  return html;
}

function render(which, d) {
  const correct = d.terms.filter((x) => x.correct).map((x) => ({ term: x.term, cs: x.cs }));
  const missed = d.terms.filter((x) => !x.correct).map((x) => x.term);
  $(`text-${which}`).innerHTML = d.text ? highlight(d.text, correct)
    : '<span class="placeholder">(silence)</span>';
  const hits = $(`hits-${which}`);
  hits.textContent = `${d.hits}/${d.total} terms`;
  hits.classList.toggle("full", d.total > 0 && d.hits === d.total);
  hits.classList.toggle("zero", d.total > 0 && d.hits === 0);
  $(`ms-${which}`).textContent = `${d.latency_ms} ms`;
  $(`missed-${which}`).innerHTML = missed.map((t) => `<span class="chip">${esc(t)}</span>`).join("");
}

boot();
