# finetune-whisper

Finetuning OpenAI's `whisper-base.en` (74M) to transcribe 2025–2026 AI-engineering
vocabulary — terms like *Claude Opus*, *Qwen*, *vLLM*, *RAG*, *MCP*, *GGUF*, *Ollama* — that the
stock model mis-hears. A learning project, run entirely on-device on an Apple M4 (MPS).

## Results

Best checkpoint of a 4-epoch run (epoch 2, lowest validation WER), on held-out eval —
**unseen sentences *and* unseen speaker voices**:

| Metric — `val_clean` | Baseline `base.en` | Finetuned |
|---|---|---|
| Word Error Rate | 7.74% | **1.61%** |
| Strict term-recall | 67.2% | **98.58%** |

Robustness on `val_aug` (noise / reverb / EQ / MP3): WER 8.26% → **1.76%**, strict
term-recall 65.6% → **98.38%**. Most of the gain lands by epoch 1.

*Strict term-recall = the model spelled the canonical term correctly, with no phonetic
credit — the honest domain-vocabulary number. Matched word-boundary delimited (cased for
homophones).*

## How it works

- **Vocabulary** — 92 domain terms in `data/terms.yaml`: 50 core terms plus 42 genuinely
  ASR-hard 2026 names (OOV models/tools like *Qwen*, *SGLang*, *Cerebras*, and
  context-anchored homophones like *Modal* vs "model", *Grok* vs *Groq*).
- **Data** — a Claude-authored sentence corpus (`data/corpus.jsonl`, ~1.6k sentences),
  synthesized to speech by two TTS engines for voice/prosody diversity: **Kokoro** ONNX
  (local, ~2/3) and **OpenAI** `gpt-4o-mini-tts` (~1/3), with audio augmentation layered on.
- **Pronunciation control** — a per-term `spoken` field rewrites canonical → spoken for the
  TTS input (e.g. `RAG`→"rag", `Qwen`→"kwen", `Ollama`→"Oh-lama") while transcripts stay canonical.
- **Homophones** — context-dependent terms carry `anchors` + `everyday` fields and are scored
  case-sensitively, so "rope" never counts as `RoPE` and "model" never becomes `Modal`.
- **Split** — two disjoint axes (held-out sentences *and* held-out voices), so the validation
  sets measure generalization, not memorization.
- **Training** — HuggingFace `Seq2SeqTrainer`, full finetune, fp32 on Apple-Silicon MPS.
- **Eval** — WER (jiwer) plus a strict per-term recall metric (`src/metrics.py`).

## Dataset + checkpoints (Google Drive)

The ~1.2 GB of pre-synthesized speech (`data/audio/`) and the trained `checkpoints/` are hosted
on Google Drive — too large for git, fully reproducible from the synth + train scripts.

**Drive folder:** https://drive.google.com/drive/folders/1SjRzRAguKup-FInrpSE5XVxAp_kt8ScS

It holds two archives — the audio dataset and the trained checkpoints. Fetch + extract both:

```bash
bash scripts/download_audio.sh        # → data/audio/ and checkpoints/
```

Or download the two files from the folder manually and extract:

```bash
tar xzf whisper-aidev-audio.tar.gz       -C data    # → data/audio/
tar xzf whisper-aidev-checkpoints.tar.gz            # → checkpoints/
```

With the checkpoints in place the web-app demo and finetuned eval run immediately — no retraining.
Or synthesize the audio yourself (see **Run** below) — needs `espeak-ng` for Kokoro and an OpenAI
key for the paid 1/3.

## Try it — voice A/B demo

`webapp/` is a small local page: record a sentence with your mic and compare stock `base.en`
vs your finetuned checkpoint side-by-side, with domain terms highlighted and a per-model hit tally.

```bash
PYTHONPATH=. python -m webapp.server      # → http://127.0.0.1:8765
```

It lists every checkpoint under `checkpoints/` — download them from the Drive folder above (or
train your own with `python -m src.train`). The dropdown's `finetuned · epoch 2` entry is the
shipped model. (No new dependencies — stdlib HTTP server + the project's stack.)

## Layout

```
data/         terms.yaml (92 terms), corpus.jsonl     (audio/ via Drive, gitignored)
src/          build_corpus · synth_kokoro · synth_openai · augment · dataset · train · eval · metrics · normalize
webapp/       local A/B voice demo (server + static page)
tests/        pytest suite
experiments/  one-off diagnostics
scripts/      download_audio.sh
```

See [`FILEMAP.md`](FILEMAP.md) for a one-line description of every file.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # test suite

python -m src.build_corpus      # validate per-term coverage + train/val split

# --- audio: either download (above) or synthesize ---
# Kokoro needs system espeak-ng (brew install espeak-ng);
# the OpenAI 1/3 (paid) reads OPENAI_API_KEY from a local .env.
python -m src.synth_kokoro --split train
python -m src.synth_openai --split train
python -m src.bake_val_aug

# --- train + eval (same code path for baseline vs finetuned) ---
python -m src.train
python -m src.eval --model openai/whisper-base.en --manifest-dir data/audio/val_clean
python -m src.eval --model checkpoints/final --manifest-dir data/audio/val_clean
```

Generated audio (`data/audio/`), TTS model weights (`kokoro_models/`), and training
checkpoints (`checkpoints/`) are gitignored — large and reproducible from these scripts.

## License

[MIT](LICENSE).
