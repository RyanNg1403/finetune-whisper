# src/synth_openai.py
import io
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from dotenv import load_dotenv

from src import config
from src.audio_io import save_wav_16k
from src.build_corpus import load_corpus
from src.normalize import load_terms, to_spoken
from src.qc import get_reference, accept_or_regenerate


def _client():
    load_dotenv(config.ROOT / ".env")
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from env


def synth_one(client, text, voice):
    resp = client.audio.speech.create(
        model=config.OPENAI_TTS_MODEL, voice=voice, input=text, response_format="wav")
    data = resp.read() if hasattr(resp, "read") else resp.content
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return wav.astype(np.float32)


def plan(split):
    """Build the (out_dir, jobs) for a split. per_sentence derives from the FULL split size."""
    corpus = [r for r in load_corpus() if r["split"] == split]
    n_full = len(corpus)
    if split == "train":
        voices = config.OPENAI_TRAIN_VOICES
        per_sentence = max(1, round(config.ENGINE_RATIO[1] * config.TARGET_TRAIN_CLIPS / n_full))
        out_dir = config.TRAIN_DIR
    else:
        voices = config.OPENAI_VAL_VOICES
        per_sentence = 1
        out_dir = config.VAL_CLEAN_DIR
    jobs, n = [], 0
    for row in corpus:
        for k in range(per_sentence):
            jobs.append((row, voices[n % len(voices)], k))
            n += 1
    return out_dir, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-qc", action="store_true", help="disable back-transcription QC")
    ap.add_argument("--max-tries", type=int, default=2, help="QC retries (low: each retry is a paid call)")
    args = ap.parse_args()

    out_dir, jobs = plan(args.split)
    if args.limit:
        jobs = jobs[:args.limit]
    ts = load_terms()
    chars = sum(len(to_spoken(j[0]["text"], ts)) for j in jobs)
    # Rough upper bound using tts-1 list price ($15 / 1M chars); gpt-4o-mini-tts bills per
    # audio token and is typically cheaper for short clips.
    print(f"PLAN {args.split}: {len(jobs)} clips, ~{chars} chars, rough cost <= ${chars/1_000_000*15:.2f}")
    if args.estimate_only:
        return

    client = _client()
    ref = None if args.no_qc else get_reference(config.QC_MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_openai.jsonl"
    n = 0
    dropped = []
    with open(manifest, "w") as mf:
        for row, voice, k in jobs:
            spoken_text = to_spoken(row["text"], ts)   # TTS hears 'rag'; transcript stays 'RAG'
            wav, passed, tries = accept_or_regenerate(
                lambda: synth_one(client, spoken_text, voice), row["terms"], ref, args.max_tries, ts)
            if not passed:
                dropped.append(row["id"])
                continue
            fname = f"oa_{row['id']}_{voice}_{k}.wav"
            save_wav_16k(out_dir / fname, wav)
            mf.write(json.dumps({"audio": fname, "text": row["text"], "terms": row["terms"],
                                 "engine": "openai", "voice": voice, "speed": 1.0}) + "\n")
            n += 1
    print(f"openai {args.split}: wrote {n} clips ({len(dropped)} dropped by QC) to {out_dir}")


if __name__ == "__main__":
    main()
