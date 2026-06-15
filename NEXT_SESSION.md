# Handoff — Whisper base.en domain finetune

**Project:** `/Users/PhatNguyen/Desktop/finetune-whisper` (git, local-only, working tree clean).
Finetune `openai/whisper-base.en` to transcribe 2025–2026 AI-dev vocabulary. Learning-focused.

## Where we are (end of session 1)
- **Pipeline complete & committed.** 27 tests green. Dataset built: `data/audio/{train=5778,
  val_clean=168, val_aug=168}` (gitignored). Corpus `data/corpus.jsonl` (1047 sentences,
  `terms` recomputed from text, 50 terms).
- **Finetune done (2 epochs).** Checkpoints kept: `checkpoints/{checkpoint-362, checkpoint-724,
  final}`. Results:
  - Baseline base.en — val_clean WER **6.95%** / strict term-recall **80.1%**; val_aug 7.28% / 78.7%.
  - Finetuned (2ep) — val_clean WER **1.32%** / strict **98.99%**; val_aug 1.17% / 99.32%.
- **Deliverables:** `results/finetune-report.html` (visual report), `research/2026-06-15-hard-terms-v2-tech.md`.

## PENDING USER INPUT (do first)
User reviews `research/2026-06-15-hard-terms-v2-tech.md` and ticks the keeper terms
(Group A = OOV-hard, Group B = context-dependent homophones, Group C = optional). Wait for
that selection before touching `terms.yaml`.

## NEXT SESSION: augment dataset with new terms → resume finetune from epoch 3
Cascading tasks, in order:

1. **Finalize term set** from the ticked v2 file (tech-only; no human names per user).
2. **`data/terms.yaml`**: add new terms — canonical spelling, category, `alts` (use the
   base.en mis-renderings as alts, e.g. Qwen→"qn3"/"gwen", Pydantic→"pedantic", RoPE→"row pe"),
   and `spoken` overrides where Kokoro/espeak mispronounces (likely Qwen→"chwen", Yi→"ee",
   Phi→"fee", RoPE pronounce "rope", Codestral, DSPy, etc.).
3. **Pronunciation audit (new terms only)**: `experiments/term_audit.py`-style Kokoro+QC pass,
   then **ear-check** the flagged ones with the user → finalize `spoken` overrides (Gate-B-lite).
4. **Corpus authoring (new sentences)**: for each new term, context-anchored **positive**
   sentences (RoPE near "positional encoding"); for Group B homophones also add **negative**
   sentences (literal "rope"/"continue"/"together"/"bolt" in everyday context) so the model
   learns to disambiguate, not always emit the tech spelling. Target ≥8–12 coverage each.
   Add to BOTH train and val splits (new terms must appear in val to be measurable).
5. **Revalidate**: `python -m src.build_corpus` (coverage + re-split). → **GATE A** transcript review.
6. **Synthesize new audio** (only the new sentences; append to existing manifests):
   `python -m src.synth_kokoro --split train/val` (70%, free) and
   `python -m src.synth_openai --split train/val` (30%, PAID ~needs user $ approval; spoken-form).
   Then bake new augmented val: `python -m src.bake_val_aug`. → **GATE B** audio review.
7. **RERUN BASELINE eval** on the now-expanded val sets (dataset changed → old baseline is
   invalid): `python -m src.eval --model openai/whisper-base.en --manifest-dir data/audio/val_clean`
   (and `val_aug`). Expect baseline WER to RISE (base.en fails the hard terms) — that's the point;
   bigger headroom = more dramatic before/after. Record new before-numbers.
8. **Resume finetune from epoch 3** on the augmented train set. ⚠️ Code change needed: `src/train.py`
   currently inits from `config.MODEL_ID` (base.en) every run. To "start at epoch 3", init the
   model from `checkpoints/final` (the epoch-2 weights) and train ~2 more epochs — add a
   `--init-from` arg (default base.en; pass `checkpoints/final`). Keep `dataloader_num_workers=0`,
   fp32, save-all-checkpoints. (Alternative to discuss: retrain fresh from base.en on the full
   augmented set for a cleaner comparison — user chose continue-from-epoch-2.)
9. **Final eval** finetuned model on expanded val_clean + val_aug → new **GATE C** before/after table.
10. **Regenerate** `results/finetune-report.html` with the new numbers; update `FILEMAP.md`; commit.

## Key technical state & gotchas (don't re-learn these)
- Env: `.venv` (Python 3.13), torch 2.12 (**MPS ok**), transformers **5.12**.
- **`.en` models REJECT `language=`/`task=` in `generate()`** — never pass them (eval/train already handle).
- **`dataloader_num_workers=0` is REQUIRED** — macOS forked workers deadlock with the audio libs.
- Augmentation is cheap (~37 ms/clip) so serial loading is fine. RoomSimulator needs
  `pyroomacoustics`; MP3 needs `fast_mp3_augment`.
- **Kokoro** is the bulk engine (`kokoro_models/` gitignored, needs system `espeak-ng`;
  scripts set `PHONEMIZER_ESPEAK_LIBRARY`). It's deterministic → **per-clip QC OFF**; correctness
  via one-time per-term audit + ear-check. Voices: train `af_heart/af_bella/af_nicole/am_michael/
  am_adam/am_onyx`, val held-out `af_sarah/am_puck`.
- **OpenAI** TTS (`gpt-4o-mini-tts`, `.env` `OPENAI_API_KEY`, also `HF_TOKEN` present) = 30%;
  QC off; spoken-form input. Engine ratio rounds to **2:1 (66.7/33.3)**, user accepted.
- **`normalize.to_spoken()`** maps canonical→spoken for TTS input (RAG→rag, Ollama→Oh-lama);
  transcript stays canonical. **`metrics.term_recall(strict=True)`** = correct-spelling metric.
- Whisper base.en already handles easy terms (Claude Opus, RAG, MCP…) — that's why we're adding
  genuinely hard ones (Qwen, RoPE, Pydantic, safetensors, Yi, Zed, DSPy, Codestral, OpenHands…).
- Supertonic (`supertonic/`) was dropped (garbles acronyms) — kept for reference, gitignored.

## Quick resume commands
```bash
cd /Users/PhatNguyen/Desktop/finetune-whisper
.venv/bin/python -m pytest tests/ -q            # confirm 27 green
.venv/bin/python -m src.build_corpus            # corpus stats
git log --oneline -5
```
