"""Measure Supertonic QC pass rate to decide viability vs researching alternatives.
For a sample of train sentences, synthesize with rotating train voices, then check the
back-transcription QC (whisper-small.en). Reports pass@1, pass@3 (regeneration), and which
terms fail persistently. Run: .venv/bin/python -m experiments.qc_viability
"""
import sys
import json
import collections
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from src import config
from src.build_corpus import load_corpus
from src.qc import get_reference, clip_passes
sys.path.insert(0, str(config.SUPERTONIC_DIR / "py"))
from helper import load_text_to_speech  # noqa: E402
from src.synth_supertonic import synth_one  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAX_TRIES = 3

tts = load_text_to_speech(str(config.SUPERTONIC_ONNX), use_gpu=False)
ref = get_reference("openai/whisper-small.en")
voices = config.SUPERTONIC_TRAIN_VOICES
speeds = config.SUPERTONIC_SPEEDS

train = [r for r in load_corpus() if r["split"] == "train" and r["terms"]][:N]
pass1 = pass3 = 0
fail_terms = collections.Counter()
tries_hist = collections.Counter()
for i, row in enumerate(train):
    voice = voices[i % len(voices)]
    speed = speeds[i % len(speeds)]
    ok_at = None
    for attempt in range(1, MAX_TRIES + 1):
        wav = synth_one(tts, row["text"], voice, speed)
        if clip_passes(ref.transcribe(wav), row["terms"]):
            ok_at = attempt
            break
    if ok_at == 1:
        pass1 += 1
    if ok_at is not None:
        pass3 += 1
        tries_hist[ok_at] += 1
    else:
        tries_hist["fail"] += 1
        for t in row["terms"]:
            if not clip_passes(ref.transcribe(synth_one(tts, row["text"], voice, speed)), [t]):
                fail_terms[t] += 1

n = len(train)
print(f"\n=== Supertonic QC viability over {n} train sentences (max_tries={MAX_TRIES}) ===")
print(f"pass@1 (first take clean):      {pass1}/{n} = {100*pass1/n:.0f}%")
print(f"pass@3 (clean within 3 takes):  {pass3}/{n} = {100*pass3/n:.0f}%")
print(f"dropped (fail after 3):         {n-pass3}/{n} = {100*(n-pass3)/n:.0f}%")
print("tries to pass:", dict(tries_hist))
if fail_terms:
    print("persistently-failing terms:", dict(fail_terms.most_common()))
