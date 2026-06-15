# src/synth_kokoro.py
import os
import json
import argparse
import numpy as np
import librosa

# help phonemizer locate espeak-ng before kokoro_onnx imports it
for _p in ["/opt/homebrew/lib/libespeak-ng.dylib", "/opt/homebrew/lib/libespeak-ng.1.dylib"]:
    if os.path.exists(_p):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _p)

from src import config
from src.audio_io import save_wav_16k
from src.build_corpus import load_corpus
from src.normalize import load_terms, to_spoken
from src.qc import get_reference, accept_or_regenerate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-qc", action="store_true", help="disable back-transcription QC")
    ap.add_argument("--max-tries", type=int, default=3)
    args = ap.parse_args()

    from kokoro_onnx import Kokoro
    kok = Kokoro(str(config.KOKORO_ONNX), str(config.KOKORO_VOICES_BIN))
    ts = load_terms()
    ref = None if args.no_qc else get_reference(config.QC_MODEL)

    full = [r for r in load_corpus() if r["split"] == args.split]
    n_full = len(full)
    corpus = full[:args.limit] if args.limit else full

    if args.split == "train":
        out_dir, voices, speeds = config.TRAIN_DIR, config.KOKORO_TRAIN_VOICES, config.KOKORO_SPEEDS
        per_sentence = max(1, round(config.ENGINE_RATIO[0] * config.TARGET_TRAIN_CLIPS / n_full))
    else:
        out_dir, voices, speeds = config.VAL_CLEAN_DIR, config.KOKORO_VAL_VOICES, [1.0]
        per_sentence = 1

    def synth(text, voice, speed):
        s, sr = kok.create(text, voice=voice, speed=speed, lang="en-us")
        return librosa.resample(np.asarray(s, np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_kokoro.jsonl"
    n = 0
    dropped = []
    with open(manifest, "w") as mf:
        for row in corpus:
            spoken_text = to_spoken(row["text"], ts)   # TTS hears 'rag'; transcript stays 'RAG'
            for j in range(per_sentence):
                base, speed = n, speeds[n % len(speeds)]
                # Kokoro is deterministic per (text,voice,speed); cycle voices across QC retries
                pick = {"i": 0, "voice": voices[base % len(voices)]}

                def make_wav():
                    pick["voice"] = voices[(base + pick["i"]) % len(voices)]
                    pick["i"] += 1
                    return synth(spoken_text, pick["voice"], speed)

                wav, passed, tries = accept_or_regenerate(make_wav, row["terms"], ref, args.max_tries, ts)
                if not passed:
                    dropped.append(row["id"])
                    n += 1
                    continue
                fname = f"ko_{row['id']}_{pick['voice']}_{int(speed*100)}_{j}.wav"
                save_wav_16k(out_dir / fname, wav)
                mf.write(json.dumps({"audio": fname, "text": row["text"], "terms": row["terms"],
                                     "engine": "kokoro", "voice": pick["voice"], "speed": speed}) + "\n")
                n += 1
    print(f"kokoro {args.split}: wrote {n - len(dropped)} clips ({len(dropped)} dropped by QC) to {out_dir}")


if __name__ == "__main__":
    main()
