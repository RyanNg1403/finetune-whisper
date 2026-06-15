# finetune-whisper

Finetuning OpenAI's `whisper-base.en` (74M) to transcribe 2025–2026 AI-engineering
vocabulary — terms like *Claude Opus*, *RAG*, *MCP*, *vibe coding*, *Ollama* — that the
stock model mis-hears. A learning project, run entirely on-device on an Apple M4 (MPS).

## Results (2 epochs, held-out eval)

| Metric — `val_clean` | Baseline `base.en` | Finetuned |
|---|---|---|
| Word Error Rate | 6.95% | **1.32%** |
| Strict term-recall | 80.1% | **98.99%** |

Robustness on `val_aug` (noise / reverb / EQ / MP3): WER 7.28% → **1.17%**, strict
term-recall 78.7% → **99.32%**. Most of the gain lands by epoch 1. The full visual
before/after is in [`results/finetune-report.html`](results/finetune-report.html).

*Strict term-recall = the model spelled the canonical term correctly, with no phonetic
credit — the honest domain-vocabulary number.*

## How it works

- **Data** — a Claude-authored sentence corpus (`data/corpus.jsonl`, ~1k sentences over
  50 domain terms in `data/terms.yaml`), synthesized to speech by two TTS engines for
  voice/prosody diversity: **Kokoro** ONNX (local, ~70%) and **OpenAI** `gpt-4o-mini-tts`
  (~30%), with on-the-fly audio augmentation layered on for acoustic realism.
- **Pronunciation control** — a per-term `spoken` field rewrites canonical → spoken for the
  TTS input (e.g. `RAG`→"rag", `Ollama`→"Oh-lama") while transcripts stay canonical.
- **Split** — two disjoint axes (held-out sentences *and* held-out speaker voices), so the
  validation sets measure generalization rather than memorization.
- **Training** — HuggingFace `Seq2SeqTrainer`, full finetune, fp32 on Apple-Silicon MPS.
- **Eval** — WER (jiwer) plus a strict per-term recall metric.

`data/terms_hard.yaml` holds a separate, research-backed set of genuinely ASR-hard
2026 terms (OOV names + context-anchored homophones) staged for a future training run.

## Layout

```
data/         terms.yaml, terms_hard.yaml, corpus.jsonl   (audio/ is gitignored)
src/          build_corpus, synth_kokoro, synth_openai, augment, dataset, train, eval
tests/        pytest suite
experiments/  one-off diagnostics
results/      finetune-report.html
```

See [`FILEMAP.md`](FILEMAP.md) for a one-line description of every file.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # test suite

python -m src.build_corpus      # validate per-term coverage + train/val split

# Synthesis needs system espeak-ng for Kokoro (brew install espeak-ng);
# the OpenAI 30% (paid) reads OPENAI_API_KEY from a local .env.
python -m src.synth_kokoro --split train
python -m src.synth_openai --split train
python -m src.bake_val_aug

# Baseline vs finetuned, same code path:
python -m src.eval --model openai/whisper-base.en --manifest-dir data/audio/val_clean
python -m src.train
```

Generated audio (`data/audio/`), TTS model weights (`kokoro_models/`), and training
checkpoints (`checkpoints/`) are gitignored — they're large and reproducible from these
scripts.

## License

[MIT](LICENSE).
