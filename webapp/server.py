# webapp/server.py
# Local A/B transcription harness: record mic audio in the browser, transcribe it with
# stock whisper-base.en AND a chosen finetune checkpoint, and compare term spelling.
#
# Stdlib-only HTTP server (no FastAPI) so it runs against the project venv as-is.
# Reuses the project's own term-matching logic (src.normalize) so the "hit" scoring
# matches our eval metric exactly.
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

# ---- model registry -------------------------------------------------------
# One processor (tokenizer + feature extractor) is shared: finetuning changed only
# the weights, never the vocab, so the base processor decodes every checkpoint.
_lock = threading.Lock()              # serialize inference (single MPS context)
_proc = None
_models = {}                          # key -> WhisperForConditionalGeneration (cached)
_terms = None


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


# Friendly aliases for the two checkpoints worth picking. `final/` is hidden from the UI
# because it is a byte-identical copy of best_wer (epoch 2) and only caused confusion.
NAMED = {
    1072: ("best_wer", "epoch 2 · WER 1.61% (lowest val WER)", 0),
    1608: ("best_strict_recall", "epoch 3 · strict recall 93.20% (highest)", 1),
}


def list_checkpoints():
    """Weight dirs under checkpoints/ (+ the archived prior-run dir). The two best
    checkpoints get named aliases and sort first; `final/` is excluded."""
    out = []
    roots = [config.CHECKPOINTS, config.CHECKPOINTS.parent / "checkpoints_prior_50term"]
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.glob("checkpoint-*")):     # note: no glob("final") — hidden by design
            if not (d / "model.safetensors").exists():
                continue
            m = re.search(r"checkpoint-(\d+)", d.name)
            step = int(m.group(1)) if m else 0
            prior = root.name.endswith("prior_50term")
            if step in NAMED and not prior:
                alias, desc, rank = NAMED[step]
                label = f"{alias}  ·  {desc}"
            else:
                label = d.name + (f"  (epoch ~{step/536:.2f})" if step else "")
                if prior:
                    label += "  [prior 50-term run]"
                rank = 2
            out.append({"name": str(d), "label": label, "step": step, "prior": prior, "rank": rank})
    out.sort(key=lambda x: (x["prior"], x["rank"], -x["step"]))
    return out


# ---- transcription + term scoring ----------------------------------------
def transcribe(model, wav):
    feats = _proc.feature_extractor(
        wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        # base.en is English-only: passing language/task raises in transformers.
        ids = model.generate(feats, max_new_tokens=128)
    return _proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


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


def score(base_text, ft_text):
    """Candidate term set = anything either model heard or spelled. Per model, a hit is
    an exactly-correct spelling. This surfaces 'baseline heard it but misspelled' cases."""
    cand = []
    for t in _terms.canonicals:
        if _heard(base_text, t) or _heard(ft_text, t) or _correct(base_text, t) or _correct(ft_text, t):
            cand.append(t)
    def detail(text):
        # cs = case-sensitive highlighting (homophones only), so the UI underlines the
        # exact surface form the strict metric credited.
        terms = [{"term": t, "correct": _correct(text, t), "cs": t in _terms.homophones} for t in cand]
        return {"terms": terms, "hits": sum(x["correct"] for x in terms), "total": len(cand)}
    return cand, detail(base_text), detail(ft_text)


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
            return self._json({"checkpoints": list_checkpoints(), "device": DEVICE})
        if route.startswith("/static/"):
            return self._file(STATIC / route[len("/static/"):])
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/transcribe":
            return self.send_error(404)
        ckpt = parse_qs(parsed.query).get("ckpt", [""])[0]
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
        cand, base_d, ft_d = score(base_text, ft_text)
        self._json({
            "candidates": cand,
            "baseline": {"text": base_text, "latency_ms": round((t1 - t0) * 1000), **base_d},
            "finetuned": {"text": ft_text, "latency_ms": round((t2 - t1) * 1000), **ft_d,
                          "checkpoint": Path(ckpt).name},
        })


if __name__ == "__main__":
    boot()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[serve] http://{HOST}:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop]")
