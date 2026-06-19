# webapp/server.py
# Local transcription harness: record mic audio in the browser, transcribe it three ways and
# compare term spelling — stock whisper-base.en, a chosen finetune checkpoint, and (zero-shot)
# Granite Speech 4.1-2b served via mlx-audio with keyword biasing on our term list.
#
# Stdlib-only HTTP server (no FastAPI) so it runs against the project venv as-is.
# Reuses the project's own term-matching logic (src.normalize) so the "hit" scoring
# matches our eval metric exactly. Whisper runs on torch/MPS; Granite on mlx-audio (Metal).
#
# Run:  PYTHONPATH=. .venv/bin/python -m webapp.server   (then open http://127.0.0.1:8765)
import io
import os
import re
import json
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src import config
from src.normalize import (load_terms, canonicalize, english_normalize, cased_count)

HERE = Path(__file__).parent
STATIC = HERE / "static"
HOST, PORT = "127.0.0.1", int(os.environ.get("WHISPER_PORT", "8765"))
DEVICE = os.environ.get("WHISPER_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
GRANITE_ID = "ibm-granite/granite-speech-4.1-2b"   # 3rd option: speech-LLM, keyword-biased, no finetune

# ---- model registry -------------------------------------------------------
# One processor (tokenizer + feature extractor) is shared: finetuning changed only
# the weights, never the vocab, so the base processor decodes every checkpoint.
_lock = threading.Lock()              # serialize inference (single MPS context)
_proc = None
_models = {}                          # key -> WhisperForConditionalGeneration (cached)
_terms = None
_granite = None                       # lazily-loaded (proc, model, biased_prompt) — heavy (~4GB bf16)


def _load(key, src):
    if key not in _models:
        print(f"[load] {key} <- {src} on {DEVICE}")
        _models[key] = WhisperForConditionalGeneration.from_pretrained(src).to(DEVICE).eval()
    return _models[key]


def boot():
    global _proc, _terms
    _proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    _terms = load_terms()
    _load("baseline", config.MODEL_ID)   # warm the baseline at startup
    print(f"[ready] device={DEVICE}  terms={len(_terms.canonicals)}")


# Friendly alias for the DEFAULT checkpoint (sorts first → preselected on load / hard-reload).
# `final/` is hidden (byte-identical copy of the epoch-2 model). All four epochs land ~98.2-98.8%
# strict recall — within noise — so the pick is marginal; epoch 4 has the highest strict (term)
# recall (98.79%) and is the default here. (Epoch 2 has the lowest WER and is the "shipped" model
# on main.) The rest are listed raw for comparison.
NAMED = {
    2144: ("finetuned", "epoch 4 · best strict recall (default)", 0),
}


def list_checkpoints():
    """Weight dirs under checkpoints/. The recommended checkpoint gets a named alias and
    sorts first; `final/` is excluded (it duplicates the epoch-2 model)."""
    out = []
    root = config.CHECKPOINTS
    if root.exists():
        for d in sorted(root.glob("checkpoint-*")):     # no glob("final") — hidden by design
            if not (d / "model.safetensors").exists():
                continue
            m = re.search(r"checkpoint-(\d+)", d.name)
            step = int(m.group(1)) if m else 0
            if step in NAMED:
                alias, desc, rank = NAMED[step]
                label = f"{alias}  ·  {desc}"
            else:
                label = d.name + (f"  (epoch ~{step/536:.2f})" if step else "")
                rank = 2
            out.append({"name": str(d), "label": label, "step": step, "rank": rank})
    out.sort(key=lambda x: (x["rank"], -x["step"]))
    return out


# ---- transcription + term scoring ----------------------------------------
def transcribe(model, wav):
    feats = _proc.feature_extractor(
        wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        # base.en is English-only: passing language/task raises in transformers.
        ids = model.generate(feats, max_new_tokens=128)
    return _proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _load_granite():
    """Lazy-load Granite Speech 4.1 (2B) on first use via mlx-audio — native Apple-Silicon
    MLX, ~3-9x faster than transformers-on-MPS at identical accuracy (verified 8/10 on our
    hard terms, ~1.7s/clip)."""
    global _granite
    if _granite is None:
        from mlx_audio.stt.utils import load as mlx_load
        print(f"[load] granite (mlx-audio) <- {GRANITE_ID} (first use)…")
        _granite = mlx_load(GRANITE_ID)
        print("[load] granite ready")
    return _granite


def _granite_prompt(extra_keywords=()):
    """Keyword-biasing prompt: our dataset terms + any user-added keywords from the UI
    (the model's intended, training-free way to handle jargon). Built per request so the
    user's live keyword edits take effect immediately."""
    kws = list(_terms.canonicals) + [k for k in extra_keywords if k]
    return ("transcribe the speech with proper punctuation and capitalization. "
            f"Keywords: {', '.join(kws)}.")


def transcribe_granite(wav, extra_keywords=()):
    model = _load_granite()
    out = model.generate(audio=np.asarray(wav, dtype=np.float32),
                         prompt=_granite_prompt(extra_keywords),
                         temperature=0.0, max_tokens=200)
    return (out.text if hasattr(out, "text") else str(out)).strip()


def _heard(text, t):
    """Lenient: did the model *hear* the term (alt-map forgives phonetic misspellings)?
    Homophones bypass the alt-map and require the exact cased form."""
    if t in _terms.homophones:
        return cased_count(text, t) > 0
    tn = english_normalize(t)
    return re.search(rf"(?<![a-z0-9]){re.escape(tn)}(?![a-z0-9])", canonicalize(text, _terms)) is not None


def _correct(text, t):
    """Strict spelling, mirroring src.metrics.term_recall(strict=True): homophones match
    case-sensitively (rope != RoPE); ordinary terms match normalized (case-insensitive, so
    a sentence-initial 'Inference' still counts as 'inference') with NO alt-map forgiveness."""
    if t in _terms.homophones:
        return cased_count(text, t) > 0
    tn = english_normalize(t)
    return re.search(rf"(?<![a-z0-9]){re.escape(tn)}(?![a-z0-9])", english_normalize(text)) is not None


def score(*texts):
    """Candidate term set = anything ANY model heard or spelled. Per model, a hit is an
    exactly-correct spelling. Returns (candidates, [detail per text, in the given order])."""
    cand = [t for t in _terms.canonicals
            if any(_heard(x, t) or _correct(x, t) for x in texts)]
    def detail(text):
        # cs = case-sensitive highlighting (homophones only), so the UI underlines the
        # exact surface form the strict metric credited.
        terms = [{"term": t, "correct": _correct(text, t), "cs": t in _terms.homophones} for t in cand]
        return {"terms": terms, "hits": sum(x["correct"] for x in terms), "total": len(cand)}
    return cand, [detail(x) for x in texts]


# ---- HTTP -----------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404); return
        ctype = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}.get(
            path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            return self._file(STATIC / "index.html")
        if route == "/api/checkpoints":
            return self._json({"checkpoints": list_checkpoints(), "device": DEVICE,
                               "keywords": _terms.canonicals})   # read-only: Granite biasing terms
        if route.startswith("/static/"):
            return self._file(STATIC / route[len("/static/"):])
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/transcribe":
            return self.send_error(404)
        q = parse_qs(parsed.query)
        ckpt = q.get("ckpt", [""])[0]
        extra_kw = [k.strip() for k in q.get("kw", [""])[0].split(",") if k.strip()]   # user keywords
        if not ckpt:
            return self._json({"error": "no checkpoint selected"}, 400)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != config.SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        except Exception as e:
            return self._json({"error": f"bad audio: {e}"}, 400)
        with _lock:
            try:
                base_m = _load("baseline", config.MODEL_ID)
                ft_m = _load(ckpt, ckpt)
                t0 = time.time(); base_text = transcribe(base_m, wav); t1 = time.time()
                ft_text = transcribe(ft_m, wav); t2 = time.time()
            except Exception as e:
                return self._json({"error": f"inference failed: {e}"}, 500)
            # Granite is heavy + slower; a failure here is non-fatal — still return the A/B.
            g_text, g_ms, g_err = "", 0, None
            try:
                g0 = time.time(); g_text = transcribe_granite(wav, extra_kw); g_ms = round((time.time() - g0) * 1000)
            except Exception as e:
                g_err = str(e)
        cand, (base_d, ft_d, g_d) = score(base_text, ft_text, g_text)
        self._json({
            "candidates": cand,
            "baseline": {"text": base_text, "latency_ms": round((t1 - t0) * 1000), **base_d},
            "finetuned": {"text": ft_text, "latency_ms": round((t2 - t1) * 1000), **ft_d,
                          "checkpoint": Path(ckpt).name},
            "granite": {"text": g_text, "latency_ms": g_ms, "error": g_err, **g_d},
        })


if __name__ == "__main__":
    boot()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[serve] http://{HOST}:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop]")
