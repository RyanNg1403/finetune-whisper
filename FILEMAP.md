# FILEMAP — Whisper base.en domain finetune

Concise map of every file and its purpose. Kept under 100 lines. `[planned]` = not yet built.

## Root
| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned Python deps (torch/MPS, transformers, audiomentations, TTS clients). |
| `.gitignore` | Ignores `.env`, `.venv/`, `data/audio/`, `checkpoints/`, `kokoro_models/`. |
| `.env` | `OPENAI_API_KEY` (gitignored, never printed). |
| `FILEMAP.md` | This file. |

## docs/superpowers/
| File | Purpose |
|------|---------|
| `specs/2026-06-15-whisper-domain-finetune-design.md` | Approved design spec (pipeline, dataset, training, eval). |
| `plans/2026-06-15-whisper-domain-finetune.md` | Step-by-step TDD implementation plan. |

## data/
| File | Purpose |
|------|---------|
| `terms.yaml` | 50 curated AI-dev terms: canonical spelling + category + spoken alts. |
| `terms_hard.yaml` | 42 research-backed hard terms (Jun-2026 trending pass): Group A OOV + Group B context-anchored homophones (`anchors`/`everyday` fields). Not yet merged into the dataset. |
| `corpus.jsonl` | 1047 authored sentences `{id,text,terms,split}` (963 train / 84 val). |
| `audio/{train,val_clean,val_aug}/` | `[planned]` generated WAVs + per-engine manifests (gitignored). |

## src/
| File | Purpose |
|------|---------|
| `config.py` | Single source of truth: paths + constants (sample rate, ratios, voices, model ids). |
| `normalize.py` | Load terms; `english_normalize` (WER), `canonicalize` (alt→canonical), `to_spoken` (TTS pronunciation override). |
| `metrics.py` | `compute_wer` (jiwer) and `term_recall` (per-term domain-vocab accuracy). |
| `qc.py` | Back-transcription QC: reference ASR + fuzzy term-intelligibility check + regenerate-on-fail. |
| `build_corpus.py` | Validate per-term coverage (≥8) + train/val split. CLI. |
| `audio_io.py` | Save / load 16 kHz mono WAV (resample + downmix). |
| `augment.py` | Train aug chain (noise/reverb/EQ/MP3) + deterministic seeded val chain. |
| `synth_kokoro.py` | **Bulk engine.** Render 70% of train + half of val_clean via Kokoro ONNX TTS (+spoken-form +QC). CLI. |
| `synth_openai.py` | Render 30% of train + half of val_clean via OpenAI TTS (+spoken-form +QC, paid, gated). CLI. |
| `bake_val_aug.py` | Bake fixed-seed augmented val set from val_clean. CLI. |
| `dataset.py` | Whisper ASR dataset + padding collator (on-the-fly train aug). |
| `eval.py` | WER + term-recall over any model/manifest (baseline + finetuned). CLI. |
| `train.py` | Seq2SeqTrainer finetune loop (MPS, fp32, save-all-checkpoints). CLI. |

## tests/
| File | Purpose |
|------|---------|
| `test_config.py` | Constants + voice-split sanity. |
| `test_normalize.py` | Term loading, normalization, alt→canonical. |
| `test_metrics.py` | WER values + term-recall hit/miss. |
| `test_build_corpus.py` | Coverage validation + disjoint split. |
| `test_audio_io.py` | 16 kHz mono roundtrip + resample/downmix. |
| `test_augment.py` | Train aug changes signal; seeded val aug is deterministic. |
| `test_qc.py` | QC fuzzy term-intelligibility + regenerate helper. |
| `test_dataset.py` / `test_eval.py` | dataset/collator + eval wiring (run the model on CPU). |

## experiments/ (saved diagnostics, not pipeline)
| File | Purpose |
|------|---------|
| `term_audit.py` | Per-term Kokoro pronunciation audit across voices (reused for new hard terms). |

## research/ (experimental notes)
| File | Purpose |
|------|---------|
| `2026-06-15-hard-terms-v2-tech.md` | Research notes + evidence behind the hard-terms pass; distilled into `data/terms_hard.yaml`. |

## results/ (deliverables)
| File | Purpose |
|------|---------|
| `finetune-report.html` | Visual before/after results report (WER + term-recall, training curve, fixes). |

## kokoro_models/ (gitignored)
Kokoro ONNX bulk-TTS engine: `kokoro-v1.0.onnx` + `voices-v1.0.bin`. Needs system `espeak-ng`.
Sole TTS engine for synthesis; Supertonic was removed from the project.
