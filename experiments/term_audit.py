"""One-time per-term pronunciation audit for Kokoro (deterministic engine).
For each term, synthesize a real corpus sentence containing it across all train voices and
check the term's intelligibility via the QC judge. Reports terms by voice-pass-rate so we can
separate REAL mispronunciations from QC-judge ignorance (which needs ear-check, not dropping).
Run: .venv/bin/python -m experiments.term_audit
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import numpy as np
import librosa

for p in ["/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/lib/libespeak-ng.1.dylib"]:
    if os.path.exists(p):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", p)

from kokoro_onnx import Kokoro  # noqa: E402
from src import config  # noqa: E402
from src.qc import get_reference, clip_passes  # noqa: E402
from src.normalize import load_terms, to_spoken  # noqa: E402

ts = load_terms()
k = Kokoro(str(config.KOKORO_ONNX), str(config.KOKORO_VOICES_BIN))
ref = get_reference(config.QC_MODEL)
voices = config.KOKORO_TRAIN_VOICES
rows = [json.loads(l) for l in open(config.CORPUS_PATH)]


def synth(text, voice):
    s, sr = k.create(to_spoken(text, ts), voice=voice, speed=1.0, lang="en-us")
    return librosa.resample(np.asarray(s, np.float32), orig_sr=sr, target_sr=16000)


results = []
for term in ts.canonicals:
    sent = next((r["text"] for r in rows if term in r["terms"]), None)
    if not sent:
        continue
    passes = sum(clip_passes(ref.transcribe(synth(sent, v)), [term], ts) for v in voices)
    results.append((passes / len(voices), term))

results.sort()
print(f"\n=== Kokoro per-term audit (voice-pass-rate over {len(voices)} train voices) ===")
print("WORST terms (candidates for ear-check / fix):")
for rate, term in results:
    if rate < 1.0:
        print(f"  {rate*100:3.0f}%  {term}")
clean = sum(1 for rate, _ in results if rate == 1.0)
print(f"\n{clean}/{len(results)} terms pass on ALL voices.")
