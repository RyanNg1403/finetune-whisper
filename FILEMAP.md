# FILEMAP — Whisper base.en domain finetune

Concise map of every file and its purpose. Kept under 100 lines. `[planned]` = not yet built.

## Root
| File | Purpose |
|------|---------|
| `README.md` | Project overview, results, and how to run. |
| `LICENSE` | MIT license. |
| `requirements.txt` | Pinned Python deps (torch/MPS, transformers, audiomentations, TTS clients). |
| `requirements-granite.txt` | Optional dep (`mlx-audio`) for the Granite web-app panel — Apple Silicon only. |
| `.gitignore` | Ignores `.env`, `.venv/`, `data/audio/`, `checkpoints/`, `kokoro_models/`, `*.tar.gz`, local-only `results/` `analysis/` `logs/` `checkpoints_prior_50term/` `more_terms.text`, and dev-only `docs/` / `research/`. |
| `.env` | `OPENAI_API_KEY` (gitignored, never printed). |
| `FILEMAP.md` | This file. |

## data/
| File | Purpose |
|------|---------|
| `terms.yaml` | 92 AI-dev terms (50 core + 42 hard, merged): canonical + category + spoken alts; Group B homophones add `anchors`/`everyday`. |
| `corpus.jsonl` | 1626 authored sentences `{id,text,terms,split}` (1428 train / 198 val). |
| `audio/{train,val_clean,val_aug}/` | `[planned]` generated WAVs + per-engine manifests (gitignored). |

## src/
| File | Purpose |
|------|---------|
| `config.py` | Single source of truth: paths + constants (sample rate, ratios, voices, model ids). |
| `normalize.py` | Load combined terms; `english_normalize` (WER), `canonicalize` (alt→canonical), `cased_count` (homophones), `to_spoken`. |
| `metrics.py` | `compute_wer` (jiwer), `term_recall` (boundary-matched, homophone-aware strict/lenient), `false_triggers`. |
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

## webapp/ (local 3-way voice demo)
| File | Purpose |
|------|---------|
| `server.py` | Stdlib HTTP server: transcribes mic audio 3 ways — `base.en`, a chosen checkpoint, and Granite Speech 4.1 (mlx-audio, keyword-biased, lazy-loaded) — and scores each, term-highlighted. |
| `static/` | `index.html` + `styles.css` + `app.js` — mic capture → 16 kHz WAV → 3-panel compare + read-only Granite keyword list. |

## scripts/
| File | Purpose |
|------|---------|
| `download_audio.sh` | Fetch + extract the pre-synthesized audio dataset (~1.2 GB) from Google Drive. |

## Local-only (gitignored, not in repo)
| Path | Purpose |
|------|---------|
| `results/*.html` | Generated visual reports (finetune results, residual-miss diagnosis). |
| `analysis/` | One-off diagnostic scripts (e.g. `diagnose_misses.py`). |

## kokoro_models/ (gitignored)
Kokoro ONNX bulk-TTS engine: `kokoro-v1.0.onnx` + `voices-v1.0.bin`. Needs system `espeak-ng`.
Sole TTS engine for synthesis; Supertonic was removed from the project.
