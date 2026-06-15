# src/eval.py
import json
import glob
import argparse
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src import config
from src.audio_io import load_wav_16k_mono
from src.metrics import compute_wer, term_recall
from src.normalize import load_terms


def _load_manifest(d):
    rows = []
    for m in glob.glob(str(Path(d) / "manifest*.jsonl")):
        for line in open(m):
            rows.append(json.loads(line))
    return rows


def evaluate(model_path, manifest_dir, device=None, batch_size=16):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    rows = _load_manifest(manifest_dir)
    refs, hyps = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        wavs = [load_wav_16k_mono(f"{manifest_dir}/{r['audio']}") for r in chunk]
        feats = proc.feature_extractor(
            wavs, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            # base.en is English-only: do NOT pass language/task (transformers raises for .en)
            ids = model.generate(feats, max_new_tokens=128)
        hyps.extend(proc.batch_decode(ids, skip_special_tokens=True))
        refs.extend(r["text"] for r in chunk)
    ts = load_terms()
    tr = term_recall(refs, hyps, ts)
    return {"n": len(rows), "wer": compute_wer(refs, hyps),
            "term_recall": tr, "refs": refs, "hyps": hyps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--manifest-dir", required=True)
    args = ap.parse_args()
    rep = evaluate(args.model, args.manifest_dir)
    print(json.dumps({"model": args.model, "n": rep["n"], "wer": round(rep["wer"], 4),
                      "term_recall_overall": round(rep["term_recall"]["overall"], 4)}, indent=2))


if __name__ == "__main__":
    main()
