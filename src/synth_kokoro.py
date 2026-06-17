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
    ap.add_argument("--append", action="store_true",
                    help="skip sentences already in the manifest and append (don't truncate existing audio)")
    ap.add_argument("--per-sentence", type=int, default=None,
                    help="override renditions per sentence (e.g. to match the existing density)")
    # Kokoro is deterministic + pronunciation is controlled via spoken-form and validated by
    # the one-time per-term audit, so per-clip QC is OFF by default (the QC judge can't read
    # our domain jargon and would false-drop good audio). Opt in with --qc to re-enable.
    ap.add_argument("--qc", action="store_true", help="enable per-clip back-transcription QC")
    ap.add_argument("--max-tries", type=int, default=3)
    args = ap.parse_args()

    from kokoro_onnx import Kokoro
    kok = Kokoro(str(config.KOKORO_ONNX), str(config.KOKORO_VOICES_BIN))
    ts = load_terms()
    ref = get_reference(config.QC_MODEL) if args.qc else None

    full = [r for r in load_corpus() if r["split"] == args.split]

    if args.split == "train":
        out_dir, voices, speeds = config.TRAIN_DIR, config.KOKORO_TRAIN_VOICES, config.KOKORO_SPEEDS
        default_ps = max(1, round(config.ENGINE_RATIO[0] * config.TARGET_TRAIN_CLIPS / len(full)))
    else:
        out_dir, voices, speeds = config.VAL_CLEAN_DIR, config.KOKORO_VAL_VOICES, [1.0]
        default_ps = 1
    per_sentence = args.per_sentence or default_ps

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_kokoro.jsonl"
    done_texts = set()
    if args.append and manifest.exists():
        done_texts = {json.loads(l)["text"] for l in open(manifest) if l.strip()}
    corpus = [r for r in full if r["text"] not in done_texts]
    if args.limit:
        corpus = corpus[:args.limit]
    print(f"kokoro {args.split}: {len(corpus)} new sentences x{per_sentence} "
          f"(skipped {len(full) - len(corpus)} already synthesized)")

    def synth(text, voice, speed):
        s, sr = kok.create(text, voice=voice, speed=speed, lang="en-us")
        return librosa.resample(np.asarray(s, np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)

    n = 0
    dropped = []
    with open(manifest, "a" if args.append else "w") as mf:
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
