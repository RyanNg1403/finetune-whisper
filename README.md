# finetune-whisper

Finetune OpenAI's `whisper-base.en` (74M) to transcribe 2025–2026 AI-engineering vocabulary —
*Claude Opus*, *Qwen*, *vLLM*, *GGUF*, *Ollama* — that the stock model mis-hears. Runs entirely
on-device on Apple Silicon (M4 / MPS). A learning project.

## Results

Finetuned vs stock `base.en` on held-out eval (**unseen sentences *and* speaker voices**):

| `val_clean` | Baseline | Finetuned |
|---|---|---|
| Word Error Rate | 7.74% | **1.61%** |
| Strict term-recall | 67.2% | **98.58%** |

Holds up under noise/reverb (`val_aug`): WER 8.26% → **1.76%**, strict recall 65.6% → **98.38%**.

*Strict term-recall = the model spelled the term exactly right, no phonetic credit — the honest
domain-vocabulary number.*

## Quickstart

```bash
git clone https://github.com/RyanNg1403/finetune-whisper && cd finetune-whisper
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/download_audio.sh         # audio + trained checkpoints from Google Drive (~2.5 GB)
PYTHONPATH=. python -m webapp.server   # → http://127.0.0.1:8765
```

`download_audio.sh` pulls two archives from the [Drive folder](https://drive.google.com/drive/folders/1SjRzRAguKup-FInrpSE5XVxAp_kt8ScS)
and extracts `data/audio/` + `checkpoints/`. (Manual alternative: download the two files, then
`tar xzf whisper-aidev-audio.tar.gz -C data` and `tar xzf whisper-aidev-checkpoints.tar.gz`.)
With the checkpoints in place the demo runs immediately — no retraining.

## The demo — three-way voice comparison

`webapp/` is a local page: record a sentence with your mic and compare three transcriptions side
by side, with the domain terms highlighted and a per-model hit tally.

- **Baseline** — stock `whisper-base.en`.
- **Finetuned** — your trained checkpoint (pick from the dropdown; epoch 4 = highest strict recall, the default).
- **Granite** — IBM Granite Speech 4.1 (2B), zero-shot. Optional — see below.

### Granite panel (optional · Apple Silicon)

The third panel runs Granite Speech 4.1 with **no finetuning** — it just receives our 92 terms as
keyword-biasing hints. It uses Apple's MLX, so it's Apple-Silicon only:

```bash
pip install -r requirements-granite.txt    # mlx-audio
```

The model (**~5 GB, Apache-2.0, no Hugging Face login**) auto-downloads on the first Granite
transcription, then is cached. To pre-fetch it ahead of time instead:

```bash
python -c "from mlx_audio.stt.utils import load; load('ibm-granite/granite-speech-4.1-2b')"
```

Without `mlx-audio` (or off Apple Silicon) the Granite panel shows "unavailable" — Baseline and
Finetuned still work.

## How it works

- **Vocabulary** — 92 terms in `data/terms.yaml`: 50 core + 42 ASR-hard 2026 names (OOV like *Qwen*,
  *SGLang*, *Cerebras*, plus context homophones like *Modal* vs "model", *Grok* vs *Groq*).
- **Data** — a Claude-authored corpus (`data/corpus.jsonl`, ~1.6k sentences) synthesized by two TTS
  engines for voice diversity: **Kokoro** ONNX (~2/3) + **OpenAI** `gpt-4o-mini-tts` (~1/3), with
  audio augmentation layered on.
- **Pronunciation** — a per-term `spoken` field steers the TTS (e.g. `Qwen`→"kwen") while transcripts
  stay canonical. Homophones carry `anchors`/`everyday` and are scored case-sensitively, so "rope"
  never counts as `RoPE`.
- **Split** — disjoint sentences *and* speaker voices, so eval measures generalization, not memorization.
- **Train / eval** — HuggingFace `Seq2SeqTrainer`, fp32 on MPS; WER (jiwer) + strict per-term recall.

## Train from scratch

```bash
pytest -q                       # tests
python -m src.build_corpus      # validate term coverage + train/val split

# synthesize the audio instead of downloading it:
#   Kokoro needs `brew install espeak-ng`; the OpenAI 1/3 (paid) reads OPENAI_API_KEY from .env
python -m src.synth_kokoro --split train
python -m src.synth_openai --split train
python -m src.bake_val_aug

python -m src.train                                                       # 4 epochs, MPS
python -m src.eval --model checkpoints/final --manifest-dir data/audio/val_clean
```

## Repo layout

```
data/        terms.yaml (92 terms) · corpus.jsonl       (audio/ via Drive, gitignored)
src/         build_corpus · synth_kokoro · synth_openai · augment · dataset · train · eval · metrics · normalize
webapp/      3-way voice demo (baseline / finetuned / Granite)
scripts/     download_audio.sh
tests/       pytest suite
```

See [`FILEMAP.md`](FILEMAP.md) for every file. Audio, checkpoints, and TTS weights are gitignored
(large — on Drive or reproducible from the scripts).

## License

[MIT](LICENSE).
