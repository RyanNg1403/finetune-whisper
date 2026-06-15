# src/synth_supertonic.py
import sys
import json
import argparse
import numpy as np
import librosa

from src import config
from src.audio_io import save_wav_16k
from src.build_corpus import load_corpus

# Supertonic's helper lives in the cloned repo; add it to the path.
sys.path.insert(0, str(config.SUPERTONIC_DIR / "py"))
from helper import load_text_to_speech, load_voice_style  # noqa: E402


def _voice_path(name):
    return str(config.SUPERTONIC_VOICE_DIR / f"{name}.json")


def synth_one(tts, text, voice, speed, total_step=8):
    style = load_voice_style([_voice_path(voice)])
    wav, dur = tts(text, "en", style, total_step, speed)  # wav: [1, T] @ tts.sample_rate
    w = np.asarray(wav[0], dtype=np.float32)
    if tts.sample_rate != config.SAMPLE_RATE:
        w = librosa.resample(w, orig_sr=tts.sample_rate, target_sr=config.SAMPLE_RATE)
    return np.asarray(w, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap sentences (for sampling)")
    args = ap.parse_args()

    tts = load_text_to_speech(str(config.SUPERTONIC_ONNX), use_gpu=False)
    corpus = [r for r in load_corpus() if r["split"] == args.split]
    if args.limit:
        corpus = corpus[:args.limit]

    if args.split == "train":
        out_dir = config.TRAIN_DIR
        voices = config.SUPERTONIC_TRAIN_VOICES
        speeds = config.SUPERTONIC_SPEEDS
        per_sentence = max(1, round(config.ENGINE_RATIO[0] * config.TARGET_TRAIN_CLIPS / len(corpus)))
    else:
        out_dir = config.VAL_CLEAN_DIR
        voices = config.SUPERTONIC_VAL_VOICES
        speeds = [1.05]
        per_sentence = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_supertonic.jsonl"
    n = 0
    with open(manifest, "w") as mf:
        for row in corpus:
            for k in range(per_sentence):
                voice = voices[n % len(voices)]
                speed = speeds[n % len(speeds)]
                wav = synth_one(tts, row["text"], voice, speed)
                fname = f"st_{row['id']}_{voice}_{int(speed*100)}_{k}.wav"
                save_wav_16k(out_dir / fname, wav)
                mf.write(json.dumps({"audio": fname, "text": row["text"], "terms": row["terms"],
                                     "engine": "supertonic", "voice": voice, "speed": speed}) + "\n")
                n += 1
    print(f"supertonic {args.split}: wrote {n} clips to {out_dir}")


if __name__ == "__main__":
    main()
