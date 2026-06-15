"""Kokoro QC viability, head-to-head with Supertonic (same 60 train sentences, same QC).
Kokoro (ONNX) is deterministic per (text,voice,speed), so retries cycle to a different voice.
Run: .venv/bin/python -m experiments.kokoro_viability [N]
"""
import os
import sys
import collections
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
from src.build_corpus import load_corpus  # noqa: E402
from src.qc import get_reference, clip_passes  # noqa: E402
from src.normalize import load_terms, to_spoken  # noqa: E402

TS = load_terms()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAX_TRIES = 3
VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam", "am_onyx"]
SPEEDS = [0.95, 1.0, 1.1]

k = Kokoro("kokoro_models/kokoro-v1.0.onnx", "kokoro_models/voices-v1.0.bin")
ref = get_reference("openai/whisper-small.en")


def synth(text, voice, speed):
    samples, sr = k.create(to_spoken(text, TS), voice=voice, speed=speed, lang="en-us")
    return librosa.resample(np.asarray(samples, np.float32), orig_sr=sr, target_sr=16000)


train = [r for r in load_corpus() if r["split"] == "train" and r["terms"]][:N]
pass1 = pass3 = 0
fail_terms = collections.Counter()
tries_hist = collections.Counter()
for i, row in enumerate(train):
    speed = SPEEDS[i % len(SPEEDS)]
    ok_at = None
    for attempt in range(MAX_TRIES):
        voice = VOICES[(i + attempt) % len(VOICES)]   # cycle voices on retry
        if clip_passes(ref.transcribe(synth(row["text"], voice, speed)), row["terms"]):
            ok_at = attempt + 1
            break
    if ok_at == 1:
        pass1 += 1
    if ok_at is not None:
        pass3 += 1
        tries_hist[ok_at] += 1
    else:
        tries_hist["fail"] += 1
        for t in row["terms"]:
            v = VOICES[i % len(VOICES)]
            if not clip_passes(ref.transcribe(synth(row["text"], v, speed)), [t]):
                fail_terms[t] += 1

n = len(train)
print(f"\n=== Kokoro QC viability over {n} train sentences (max_tries={MAX_TRIES}) ===")
print(f"pass@1 (first take clean):      {pass1}/{n} = {100*pass1/n:.0f}%")
print(f"pass@3 (clean within 3 takes):  {pass3}/{n} = {100*pass3/n:.0f}%")
print(f"dropped (fail after 3):         {n-pass3}/{n} = {100*(n-pass3)/n:.0f}%")
print("tries to pass:", dict(tries_hist))
if fail_terms:
    print("persistently-failing terms:", dict(fail_terms.most_common()))
