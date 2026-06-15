# Whisper `base.en` Domain-Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ⚠️ **SUPERSEDED IN PART — read before following any step.** This is the *original* plan as
> executed. During the build, the bulk 70% TTS engine was migrated **Supertonic → Kokoro**, and
> Supertonic was later **removed from the project entirely** (clone + ONNX weights deleted, no
> `src/synth_supertonic.py`, no `SUPERTONIC_*` config). Wherever a step below says "Supertonic"
> / `synth_supertonic.py`, the live equivalent is **Kokoro** / `src/synth_kokoro.py` (voices
> `af_*`/`am_*`, assets in `kokoro_models/`). Full rationale: **Addendum A** of the companion
> spec. The 7:3 engine ratio, two-axis split, augmentation, training, and eval steps are unchanged.

**Goal:** Finetune `openai/whisper-base.en` so it reliably transcribes 2025–2026 AI-dev jargon (e.g. "harness engineering", "Claude Opus", "vibe coding"), using TTS-synthesized training audio (Supertonic 70% + OpenAI 30%) plus audio augmentation, on Apple-Silicon MPS.

**Architecture:** A linear, gated pipeline: curated term list → Claude-authored sentence corpus (**Gate A: transcript review**) → TTS synthesis by two engines at 7:3 ratio (**Gate B: audio review**) → on-the-fly augmented finetuning on MPS → WER + term-recall eval with a baseline-first before/after table (**Gate C: results review**).

**Tech Stack:** Python 3.13 (`.venv`), PyTorch + MPS, HuggingFace `transformers`/`datasets`/`evaluate`, `jiwer`, `audiomentations`, Supertonic ONNX TTS, OpenAI TTS API, `soundfile`/`librosa`.

**Companion spec:** `docs/superpowers/specs/2026-06-15-whisper-domain-finetune-design.md` — read it first.

---

## File Structure (single source of truth for names)

```
src/
  config.py          # all paths + constants (SAMPLE_RATE=16000, ENGINE_RATIO, MIN_TERM_COVERAGE, voices...)
  normalize.py       # load_terms(), TermSet, canonicalize(text), english_normalize(text)
  metrics.py         # compute_wer(refs, hyps), term_recall(refs, hyps, termset)
  audio_io.py        # load_wav_16k_mono(path), save_wav_16k(path, wav)
  augment.py         # build_train_augmenter(), build_fixed_val_augmenter(seed), apply(aug, wav)
  build_corpus.py    # validate_coverage(corpus, termset), split_corpus(corpus) — CLI
  synth_supertonic.py# CLI: render train(70%) + half of val_clean (local)
  synth_openai.py    # CLI: render train(30%) + half of val_clean (paid, gated)
  dataset.py         # WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding
  eval.py            # evaluate(model, manifest, termset) -> report dict — CLI
  train.py           # Seq2SeqTrainer finetune — CLI
data/
  terms.yaml
  corpus.jsonl
  audio/{train,val_clean,val_aug}/   # WAVs + manifest.jsonl per dir (gitignored)
tests/
  test_normalize.py test_metrics.py test_audio_io.py test_augment.py
  test_build_corpus.py test_dataset.py test_eval.py
  fixtures/          # tiny wavs + mini manifest for eval/dataset tests
```

**Shared constants (defined once in `src/config.py`, referenced everywhere):**
`SAMPLE_RATE=16000`, `MIN_TERM_COVERAGE=8`, `ENGINE_RATIO=(0.7, 0.3)`,
`SUPERTONIC_TRAIN_VOICES=["M1","M2","M3","F1","F2","F3"]`,
`SUPERTONIC_VAL_VOICES=["M4","F4"]`,
`OPENAI_TRAIN_VOICES=["alloy","echo","fable","onyx","nova","shimmer"]`,
`OPENAI_VAL_VOICES=["ash","sage"]`,
`SUPERTONIC_SPEEDS=[0.95,1.05,1.15]`, `MODEL_ID="openai/whisper-base.en"`.

---

## Phase 0 — Environment & skeleton

### Task 0.1: Create venv and install dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write `requirements.txt`**

```
torch
torchaudio
transformers>=4.44
datasets
accelerate
evaluate
jiwer
audiomentations
soundfile
librosa
numpy
onnxruntime
openai>=1.40
python-dotenv
pyyaml
pytest
```

- [ ] **Step 2: Create the venv and install**

Run:
```bash
cd /Users/PhatNguyen/Desktop/finetune-whisper
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Expected: all install without error. If `torch` lacks a 3.13 wheel, recreate the venv with `python3.11 -m venv .venv` and reinstall (note in commit message).

- [ ] **Step 3: Verify MPS is available**

Run:
```bash
.venv/bin/python -c "import torch; print('mps', torch.backends.mps.is_available())"
```
Expected: `mps True`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin dependencies and create venv"
```

### Task 0.2: Project config module

**Files:**
- Create: `src/__init__.py` (empty)
- Create: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from src import config

def test_core_constants():
    assert config.SAMPLE_RATE == 16000
    assert abs(sum(config.ENGINE_RATIO) - 1.0) < 1e-9
    assert config.MIN_TERM_COVERAGE >= 1
    assert config.MODEL_ID == "openai/whisper-base.en"
    # train and val voices must be disjoint (no speaker leakage)
    assert not (set(config.SUPERTONIC_TRAIN_VOICES) & set(config.SUPERTONIC_VAL_VOICES))
    assert not (set(config.OPENAI_TRAIN_VOICES) & set(config.OPENAI_VAL_VOICES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: src.config`)

- [ ] **Step 3: Write `src/config.py`**

```python
# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUDIO = DATA / "audio"
TRAIN_DIR = AUDIO / "train"
VAL_CLEAN_DIR = AUDIO / "val_clean"
VAL_AUG_DIR = AUDIO / "val_aug"
TERMS_PATH = DATA / "terms.yaml"
CORPUS_PATH = DATA / "corpus.jsonl"
CHECKPOINTS = ROOT / "checkpoints"
SUPERTONIC_DIR = ROOT / "supertonic"
SUPERTONIC_ONNX = SUPERTONIC_DIR / "assets" / "onnx"
SUPERTONIC_VOICE_DIR = SUPERTONIC_DIR / "assets" / "voice_styles"

SAMPLE_RATE = 16000
MIN_TERM_COVERAGE = 8
ENGINE_RATIO = (0.7, 0.3)  # (supertonic, openai) of TRAIN clips
TARGET_TRAIN_CLIPS = 6000
MODEL_ID = "openai/whisper-base.en"

SUPERTONIC_TRAIN_VOICES = ["M1", "M2", "M3", "F1", "F2", "F3"]
SUPERTONIC_VAL_VOICES = ["M4", "F4"]
SUPERTONIC_SPEEDS = [0.95, 1.05, 1.15]
OPENAI_TRAIN_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_VAL_VOICES = ["ash", "sage"]
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
SEED = 42
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/__init__.py src/config.py tests/test_config.py
git commit -m "feat: project config with shared constants"
```

### Task 0.3: Fetch Supertonic assets

- [ ] **Step 1: Download ONNX models + voice styles via Git LFS**

Run:
```bash
cd /Users/PhatNguyen/Desktop/finetune-whisper
git lfs install
git lfs clone https://huggingface.co/Supertone/supertonic-3 supertonic/assets
```
Expected: `supertonic/assets/onnx/*.onnx` and `supertonic/assets/voice_styles/*.json` exist.

- [ ] **Step 2: Verify assets present**

Run:
```bash
ls supertonic/assets/onnx/*.onnx && ls supertonic/assets/voice_styles/*.json | head
```
Expected: 4 onnx files (duration_predictor, text_encoder, vector_estimator, vocoder) + voice JSONs. (No commit — `supertonic/` is gitignored.)

---

## Phase 1 — Terms & normalization (TDD)

### Task 1.1: Author `data/terms.yaml`

**Files:**
- Create: `data/terms.yaml`

- [ ] **Step 1: Write the term list** (canonical spelling + spoken alts the stock model may emit). Use this schema; author 40–60 entries across categories:

```yaml
# data/terms.yaml
terms:
  - canonical: "Claude Opus"
    category: model
    alts: ["cloud opus", "claud opus", "clod opus"]
  - canonical: "Claude Sonnet"
    category: model
    alts: ["cloud sonnet", "claude sonet"]
  - canonical: "Claude Code"
    category: model
    alts: ["cloud code"]
  - canonical: "GPT-5"
    category: model
    alts: ["gpt 5", "gpt five", "g p t five"]
  - canonical: "Codex"
    category: model
    alts: ["codecs", "co dex"]
  - canonical: "Cursor"
    category: product
    alts: ["curser"]
  - canonical: "Windsurf"
    category: product
    alts: ["wind surf"]
  - canonical: "Devin"
    category: product
    alts: ["devon", "deaven"]
  - canonical: "Anthropic"
    category: lab
    alts: ["anthropics", "anthropik"]
  - canonical: "vibe coding"
    category: concept
    alts: ["vibe coating", "five coding"]
  - canonical: "harness engineering"
    category: concept
    alts: ["harnes engineering", "harness engineer"]
  - canonical: "context engineering"
    category: concept
    alts: ["contacts engineering"]
  - canonical: "agentic coding"
    category: concept
    alts: ["a gentic coding", "agentic coating"]
  - canonical: "subagents"
    category: concept
    alts: ["sub agents", "sub-agents"]
  - canonical: "tool calling"
    category: concept
    alts: ["toole calling"]
  - canonical: "RAG"
    category: concept
    alts: ["rag", "rack", "rg"]
  - canonical: "MCP"
    category: concept
    alts: ["m c p", "mcp server", "em see pee"]
  - canonical: "evals"
    category: concept
    alts: ["e vals", "emails"]
  - canonical: "fine-tuning"
    category: concept
    alts: ["fine tuning", "finetuning"]
  - canonical: "LoRA"
    category: concept
    alts: ["lora", "laura", "low rank"]
  - canonical: "RLHF"
    category: concept
    alts: ["r l h f", "are l h f"]
  - canonical: "test-time compute"
    category: concept
    alts: ["test time compute"]
  - canonical: "system prompt"
    category: concept
    alts: ["sistem prompt"]
  - canonical: "hallucination"
    category: concept
    alts: ["halucination"]
  - canonical: "Hugging Face"
    category: tool
    alts: ["hugging phase", "huggingface"]
  - canonical: "vLLM"
    category: tool
    alts: ["v l l m", "velm"]
  - canonical: "Ollama"
    category: tool
    alts: ["olama", "o llama"]
  - canonical: "Whisper"
    category: tool
    alts: ["whisperer"]
  # ... continue to 40–60 entries: Claude Haiku, Gemini, Copilot, OpenAI,
  # DeepMind, Mistral, xAI, LangChain, LlamaIndex, embeddings, chain of thought,
  # guardrails, prompt engineering, token, inference, quantization, etc.
```

- [ ] **Step 2: Validate YAML parses**

Run: `.venv/bin/python -c "import yaml; d=yaml.safe_load(open('data/terms.yaml')); print(len(d['terms']),'terms')"`
Expected: prints a count between 40 and 60.

- [ ] **Step 3: Commit**

```bash
git add data/terms.yaml
git commit -m "feat: curated AI-dev term list with canonical spellings"
```

### Task 1.2: Normalization & term loading module

**Files:**
- Create: `src/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_normalize.py
from src.normalize import load_terms, canonicalize, english_normalize

def test_load_terms():
    ts = load_terms()
    assert len(ts.terms) >= 40
    assert "Claude Opus" in ts.canonicals

def test_english_normalize_lowercases_and_strips_punct():
    assert english_normalize("Claude Opus, really!") == "claude opus really"

def test_canonicalize_maps_alts_to_canonical():
    # "cloud opus" is an alt of "Claude Opus" -> canonical surface form
    out = canonicalize("i love cloud opus")
    assert "claude opus" in out  # canonical, normalized to lowercase compare form

def test_canonicalize_handles_acronyms():
    assert "mcp" in canonicalize("we use m c p servers")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_normalize.py -v`
Expected: FAIL (`ModuleNotFoundError: src.normalize`)

- [ ] **Step 3: Write `src/normalize.py`**

```python
# src/normalize.py
import re
import yaml
from dataclasses import dataclass, field
from src import config

@dataclass
class TermSet:
    terms: list                      # list of dicts {canonical, category, alts}
    canonicals: list = field(default_factory=list)
    _alt_map: dict = field(default_factory=dict)  # normalized alt -> normalized canonical

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # drop punctuation/hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s

def english_normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for WER scoring."""
    return _norm(text)

def load_terms(path=None) -> TermSet:
    path = path or config.TERMS_PATH
    data = yaml.safe_load(open(path))
    terms = data["terms"]
    ts = TermSet(terms=terms, canonicals=[t["canonical"] for t in terms])
    for t in terms:
        canon_norm = _norm(t["canonical"])
        ts._alt_map[canon_norm] = canon_norm
        for alt in t.get("alts", []):
            ts._alt_map[_norm(alt)] = canon_norm
    return ts

def canonicalize(text: str, termset: TermSet = None) -> str:
    """Normalize text, then rewrite any known alt phrase to its canonical (normalized) form.
    Longer alts are matched first so multi-word terms win over single words."""
    termset = termset or load_terms()
    out = _norm(text)
    for alt in sorted(termset._alt_map, key=lambda a: -len(a.split())):
        canon = termset._alt_map[alt]
        if alt == canon:
            continue
        out = re.sub(rf"\b{re.escape(alt)}\b", canon, out)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/normalize.py tests/test_normalize.py
git commit -m "feat: term loading, english normalization, alt->canonical rewrite"
```

### Task 1.3: Metrics — WER + term recall (TDD)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from src.normalize import load_terms
from src.metrics import compute_wer, term_recall

def test_wer_perfect_is_zero():
    assert compute_wer(["claude opus is great"], ["claude opus is great"]) == 0.0

def test_wer_counts_one_sub():
    # one wrong word out of four -> 0.25
    assert abs(compute_wer(["claude opus is great"], ["claude opus is good"]) - 0.25) < 1e-6

def test_term_recall_hits_canonical():
    ts = load_terms()
    refs = ["i tried claude opus today"]
    hyps_good = ["i tried claude opus today"]
    hyps_bad = ["i tried cloud opus today"]   # 'cloud opus' is an alt -> canonicalizes to claude opus
    r_good = term_recall(refs, hyps_good, ts)
    assert r_good["overall"] == 1.0
    # after canonicalization the bad hyp also maps to claude opus -> still a hit (measures phonetic capture)
    r_bad = term_recall(refs, hyps_bad, ts)
    assert r_bad["overall"] == 1.0

def test_term_recall_misses_unrelated():
    ts = load_terms()
    refs = ["i tried claude opus today"]
    hyps = ["i tried the weather today"]
    assert term_recall(refs, hyps, ts)["overall"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL (`ModuleNotFoundError: src.metrics`)

- [ ] **Step 3: Write `src/metrics.py`**

```python
# src/metrics.py
import jiwer
from src.normalize import english_normalize, canonicalize, load_terms

def compute_wer(refs, hyps):
    refs_n = [english_normalize(r) for r in refs]
    hyps_n = [english_normalize(h) for h in hyps]
    return float(jiwer.wer(refs_n, hyps_n))

def term_recall(refs, hyps, termset=None):
    """For each canonical term occurrence in the (canonicalized) reference, count it a hit
    if the canonical surface form appears in the (canonicalized) hypothesis.
    Returns {'overall': float, 'per_term': {term: (hits, total)}}."""
    termset = termset or load_terms()
    per_term = {}
    total_hits = total = 0
    for ref, hyp in zip(refs, hyps):
        ref_c = canonicalize(ref, termset)
        hyp_c = canonicalize(hyp, termset)
        for t in termset.canonicals:
            t_n = english_normalize(t)
            n_ref = ref_c.count(t_n)
            if n_ref == 0:
                continue
            n_hit = min(n_ref, hyp_c.count(t_n))
            h, tot = per_term.get(t, (0, 0))
            per_term[t] = (h + n_hit, tot + n_ref)
            total_hits += n_hit
            total += n_ref
    return {
        "overall": (total_hits / total) if total else 0.0,
        "per_term": per_term,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: WER and term-recall metrics"
```

---

## Phase 2 — Corpus & build_corpus (Gate A)

### Task 2.1: Coverage validator + splitter (TDD)

**Files:**
- Create: `src/build_corpus.py`
- Test: `tests/test_build_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_corpus.py
import pytest
from src.normalize import load_terms
from src.build_corpus import validate_coverage, split_corpus, CoverageError

def _corpus(term, n):
    return [{"id": f"s{i}", "text": f"i used {term} again", "terms": [term]} for i in range(n)]

def test_validate_passes_when_covered():
    ts = load_terms()
    term = ts.canonicals[0]
    corpus = _corpus(term, 8)
    # only checks the one term we seeded; pass the subset
    validate_coverage(corpus, ts, only_terms=[term], min_coverage=8)

def test_validate_fails_when_undercovered():
    ts = load_terms()
    term = ts.canonicals[0]
    corpus = _corpus(term, 3)
    with pytest.raises(CoverageError):
        validate_coverage(corpus, ts, only_terms=[term], min_coverage=8)

def test_split_is_disjoint_and_sized():
    corpus = [{"id": f"s{i}", "text": "x", "terms": []} for i in range(100)]
    train, val = split_corpus(corpus, val_frac=0.1, seed=42)
    assert len(val) == 10 and len(train) == 90
    assert not (set(s["id"] for s in train) & set(s["id"] for s in val))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_build_corpus.py -v`
Expected: FAIL (`ModuleNotFoundError: src.build_corpus`)

- [ ] **Step 3: Write `src/build_corpus.py`**

```python
# src/build_corpus.py
import json
import random
import sys
from collections import Counter
from src import config
from src.normalize import load_terms, canonicalize, english_normalize

class CoverageError(Exception):
    pass

def _count_terms(corpus, termset):
    counts = Counter()
    for row in corpus:
        text_c = canonicalize(row["text"], termset)
        for t in termset.canonicals:
            if english_normalize(t) in text_c:
                counts[t] += 1
    return counts

def validate_coverage(corpus, termset=None, only_terms=None, min_coverage=None):
    termset = termset or load_terms()
    min_coverage = min_coverage or config.MIN_TERM_COVERAGE
    counts = _count_terms(corpus, termset)
    check = only_terms or termset.canonicals
    under = {t: counts.get(t, 0) for t in check if counts.get(t, 0) < min_coverage}
    if under:
        raise CoverageError(f"Under-covered terms (< {min_coverage}): {under}")
    return counts

def split_corpus(corpus, val_frac=0.08, seed=42):
    rng = random.Random(seed)
    shuffled = corpus[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    for s in train: s["split"] = "train"
    for s in val: s["split"] = "val"
    return train, val

def load_corpus(path=None):
    path = path or config.CORPUS_PATH
    return [json.loads(l) for l in open(path) if l.strip()]

def main():
    ts = load_terms()
    corpus = load_corpus()
    counts = validate_coverage(corpus, ts)
    train, val = split_corpus(corpus)
    out = config.CORPUS_PATH
    with open(out, "w") as f:
        for row in train + val:
            f.write(json.dumps(row) + "\n")
    print(f"OK: {len(corpus)} sentences, {len(train)} train / {len(val)} val")
    print("min term coverage:", min(counts.values()))

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_build_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/build_corpus.py tests/test_build_corpus.py
git commit -m "feat: corpus coverage validator and train/val splitter"
```

### Task 2.2: Author the sentence corpus (Claude-authored data)

**Files:**
- Create: `data/corpus.jsonl`

- [ ] **Step 1: Author ~980 sentences** into `data/corpus.jsonl`, one JSON object per line:
`{"id": "c0001", "text": "...", "terms": ["Claude Opus", ...]}`. Rules:
  - 6–25 words, natural conversational software-dev speech.
  - Every term from `terms.yaml` appears in **≥ 8** sentences across varied contexts.
  - Use the **canonical spelling** verbatim in `text` (so transcripts are pre-normalized).
  - Mix single-term and multi-term sentences.
  - Author in batches (e.g. 100 at a time); may dispatch parallel subagents each assigned a
    term subset, then concatenate — but the validator (Step 2) is the gate on correctness.

- [ ] **Step 2: Run the validator + splitter**

Run: `.venv/bin/python -m src.build_corpus`
Expected: `OK: ~980 sentences, ~900 train / ~80 val` and `min term coverage: >= 8`.
If it raises `CoverageError`, add sentences for the listed terms and re-run.

- [ ] **Step 3: Commit**

```bash
git add data/corpus.jsonl
git commit -m "feat: authored AI-dev sentence corpus (validated coverage)"
```

> ### ⛔ GATE A — TRANSCRIPT REVIEW (STOP)
> Present the corpus to the user: term coverage table (from validator) + a sample of
> sentences per category. **Do not proceed to synthesis until the user approves the
> transcripts** (terms present, correctly spelled, used naturally).

---

## Phase 3 — Synthesis (Gate B)

### Task 3.1: Audio I/O utilities (TDD)

**Files:**
- Create: `src/audio_io.py`
- Test: `tests/test_audio_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_io.py
import numpy as np
from src.audio_io import save_wav_16k, load_wav_16k_mono
from src import config

def test_roundtrip_16k_mono(tmp_path):
    wav = (0.1 * np.random.randn(8000)).astype(np.float32)  # 0.5s @16k
    p = tmp_path / "a.wav"
    save_wav_16k(p, wav)
    out = load_wav_16k_mono(p)
    assert out.ndim == 1
    assert len(out) == 8000

def test_resamples_and_downmixes(tmp_path):
    import soundfile as sf
    stereo = (0.1 * np.random.randn(48000, 2)).astype(np.float32)  # 1s @48k stereo
    p = tmp_path / "b.wav"
    sf.write(p, stereo, 48000)
    out = load_wav_16k_mono(p)
    assert out.ndim == 1
    assert abs(len(out) - 16000) <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_audio_io.py -v`
Expected: FAIL (`ModuleNotFoundError: src.audio_io`)

- [ ] **Step 3: Write `src/audio_io.py`**

```python
# src/audio_io.py
import numpy as np
import soundfile as sf
import librosa
from src import config

def save_wav_16k(path, wav):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    sf.write(str(path), wav, config.SAMPLE_RATE)

def load_wav_16k_mono(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return wav.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_audio_io.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio_io.py tests/test_audio_io.py
git commit -m "feat: 16kHz mono wav io with resample/downmix"
```

### Task 3.2: Supertonic synthesis script

**Files:**
- Create: `src/synth_supertonic.py`

- [ ] **Step 1: Write `src/synth_supertonic.py`** (renders 70% of train + half of val_clean). It imports Supertonic's helper from the cloned repo.

```python
# src/synth_supertonic.py
import sys, json, argparse
sys.path.insert(0, str(__import__("src.config", fromlist=["SUPERTONIC_DIR"]).SUPERTONIC_DIR / "py"))
from helper import load_text_to_speech, load_voice_style  # from supertonic/py
from src import config
from src.audio_io import save_wav_16k
from src.build_corpus import load_corpus
import numpy as np, librosa

def _voice_path(name):
    return str(config.SUPERTONIC_VOICE_DIR / f"{name}.json")

def synth_one(tts, text, voice, speed, total_step=8):
    style = load_voice_style([_voice_path(voice)])
    wav, dur = tts(text, "en", style, total_step, speed)  # wav: [1, T] @ tts.sample_rate
    w = wav[0]
    if tts.sample_rate != config.SAMPLE_RATE:
        w = librosa.resample(np.asarray(w, np.float32), orig_sr=tts.sample_rate, target_sr=config.SAMPLE_RATE)
    return np.asarray(w, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap sentences (for sampling)")
    args = ap.parse_args()

    tts = load_text_to_speech(str(config.SUPERTONIC_ONNX), use_gpu=False)
    corpus = [r for r in load_corpus() if r["split"] == args.split]
    if args.limit:
        corpus = corpus[:args.limit]

    if args.split == "train":
        out_dir, voices, speeds = config.TRAIN_DIR, config.SUPERTONIC_TRAIN_VOICES, config.SUPERTONIC_SPEEDS
        # 70% of TARGET_TRAIN_CLIPS spread over train sentences:
        per_sentence = max(1, round(config.ENGINE_RATIO[0] * config.TARGET_TRAIN_CLIPS / len(corpus)))
    else:
        out_dir, voices, speeds = config.VAL_CLEAN_DIR, config.SUPERTONIC_VAL_VOICES, [1.05]
        per_sentence = 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_supertonic.jsonl"
    rng = np.random.default_rng(config.SEED)
    n = 0
    with open(manifest, "w") as mf:
        for row in corpus:
            for k in range(per_sentence):
                voice = voices[(n) % len(voices)]
                speed = speeds[(n) % len(speeds)]
                wav = synth_one(tts, row["text"], voice, speed)
                fname = f"st_{row['id']}_{voice}_{int(speed*100)}_{k}.wav"
                save_wav_16k(out_dir / fname, wav)
                mf.write(json.dumps({"audio": fname, "text": row["text"], "terms": row["terms"],
                                     "engine": "supertonic", "voice": voice, "speed": speed}) + "\n")
                n += 1
    print(f"supertonic {args.split}: wrote {n} clips to {out_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test — render 3 train clips**

Run: `.venv/bin/python -m src.synth_supertonic --split train --limit 1`
Expected: writes a few `st_*.wav` into `data/audio/train/` + manifest lines; no error.

- [ ] **Step 3: Assert wavs are valid 16k mono non-silent**

Run:
```bash
.venv/bin/python -c "
from src.audio_io import load_wav_16k_mono; import glob, numpy as np
fs=glob.glob('data/audio/train/st_*.wav'); assert fs
for f in fs:
    w=load_wav_16k_mono(f); assert w.ndim==1 and (w**2).mean()**0.5>1e-3, f
print('ok', len(fs))"
```
Expected: `ok N`

- [ ] **Step 4: Commit**

```bash
git add src/synth_supertonic.py
git commit -m "feat: Supertonic synthesis to 16kHz mono with manifest"
```

### Task 3.3: OpenAI synthesis script (paid — gated)

**Files:**
- Create: `src/synth_openai.py`

- [ ] **Step 1: Write `src/synth_openai.py`** (renders 30% of train + half of val_clean; supports `--estimate-only`).

```python
# src/synth_openai.py
import io, os, json, argparse
import numpy as np, librosa, soundfile as sf
from dotenv import load_dotenv
from src import config
from src.audio_io import save_wav_16k
from src.build_corpus import load_corpus

def _client():
    load_dotenv(config.ROOT / ".env")
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from env

def synth_one(client, text, voice):
    resp = client.audio.speech.create(model=config.OPENAI_TTS_MODEL, voice=voice,
                                       input=text, response_format="wav")
    data = resp.read() if hasattr(resp, "read") else resp.content
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return wav.astype(np.float32)

def plan(split):
    corpus = [r for r in load_corpus() if r["split"] == split]
    if split == "train":
        voices = config.OPENAI_TRAIN_VOICES
        per_sentence = max(1, round(config.ENGINE_RATIO[1] * config.TARGET_TRAIN_CLIPS / len(corpus)))
        out_dir = config.TRAIN_DIR
    else:
        voices = config.OPENAI_VAL_VOICES
        per_sentence = 1
        out_dir = config.VAL_CLEAN_DIR
    jobs = []
    n = 0
    for row in corpus:
        for k in range(per_sentence):
            jobs.append((row, voices[n % len(voices)], k)); n += 1
    return out_dir, jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir, jobs = plan(args.split)
    if args.limit: jobs = jobs[:args.limit]
    chars = sum(len(j[0]["text"]) for j in jobs)
    # gpt-4o-mini-tts approx audio-token pricing; tts-1 is ~$15/1M chars as a rough upper bound
    print(f"PLAN {args.split}: {len(jobs)} clips, ~{chars} chars, rough cost <= ${chars/1_000_000*15:.2f}")
    if args.estimate_only:
        return

    client = _client()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest_openai.jsonl"
    n = 0
    with open(manifest, "w") as mf:
        for row, voice, k in jobs:
            wav = synth_one(client, row["text"], voice)
            fname = f"oa_{row['id']}_{voice}_{k}.wav"
            save_wav_16k(out_dir / fname, wav)
            mf.write(json.dumps({"audio": fname, "text": row["text"], "terms": row["terms"],
                                 "engine": "openai", "voice": voice, "speed": 1.0}) + "\n")
            n += 1
    print(f"openai {args.split}: wrote {n} clips to {out_dir}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Print cost estimate (no API call)**

Run: `.venv/bin/python -m src.synth_openai --split train --estimate-only`
Expected: prints clip count + char count + rough cost (low single digits).

> ### ⛔ PAID-RUN CONFIRMATION (STOP)
> Show the user the estimate from Step 2 and the small-sample plan. **Get explicit
> approval before any real OpenAI call** (per the user's API-cost rule).

- [ ] **Step 3: Small paid sample (after approval)**

Run: `.venv/bin/python -m src.synth_openai --split train --limit 5`
Expected: 5 `oa_*.wav` written; verify non-silent with the Task 3.2 Step 3 snippet (glob `oa_*`).

- [ ] **Step 4: Commit**

```bash
git add src/synth_openai.py
git commit -m "feat: OpenAI TTS synthesis with cost-estimate gate"
```

### Task 3.4: Full generation run

- [ ] **Step 1: Generate the full sets (after Gate A + paid confirmation)**

Run:
```bash
.venv/bin/python -m src.synth_supertonic --split train
.venv/bin/python -m src.synth_supertonic --split val
.venv/bin/python -m src.synth_openai --split train
.venv/bin/python -m src.synth_openai --split val
```
Expected: ~4,200 supertonic + ~1,800 openai train clips (≈7:3); ~160 val_clean clips.

- [ ] **Step 2: Verify ratio + counts**

Run:
```bash
.venv/bin/python -c "
import json,glob,collections
c=collections.Counter()
for m in glob.glob('data/audio/train/manifest_*.jsonl'):
    for l in open(m): c[json.loads(l)['engine']]+=1
tot=sum(c.values()); print(c, 'supertonic frac=%.2f'%(c['supertonic']/tot))"
```
Expected: supertonic fraction ≈ 0.70.

> ### ⛔ GATE B — AUDIO REVIEW (STOP)
> Provide the user a sample of generated clips from **both** engines (a few train + val).
> **Do not start training until the user approves audio quality.**

---

## Phase 4 — Augmentation (TDD)

### Task 4.1: Augmentation module

**Files:**
- Create: `src/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import numpy as np
from src.augment import build_train_augmenter, build_fixed_val_augmenter, apply
from src import config

def test_train_aug_changes_signal():
    rng = np.random.default_rng(0)
    wav = (0.1*rng.standard_normal(16000)).astype(np.float32)
    aug = build_train_augmenter()
    out = apply(aug, wav)
    assert out.shape[0] >= 1 and not np.array_equal(out, wav)

def test_fixed_val_aug_is_deterministic():
    wav = (0.1*np.random.default_rng(1).standard_normal(16000)).astype(np.float32)
    a = apply(build_fixed_val_augmenter(seed=123), wav)
    b = apply(build_fixed_val_augmenter(seed=123), wav)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_augment.py -v`
Expected: FAIL (`ModuleNotFoundError: src.augment`)

- [ ] **Step 3: Write `src/augment.py`**

```python
# src/augment.py
import random
import numpy as np
from audiomentations import (Compose, AddColoredNoise, RoomSimulator, Gain,
                             Mp3Compression, SevenBandParametricEQ)
from src import config

def _chain():
    return Compose([
        AddColoredNoise(min_snr_db=8.0, max_snr_db=30.0, p=0.7),
        RoomSimulator(p=0.4),
        Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.4),
        SevenBandParametricEQ(p=0.3),
        Mp3Compression(min_bitrate=32, max_bitrate=96, p=0.3),
    ])

def build_train_augmenter():
    return _chain()

def build_fixed_val_augmenter(seed=config.SEED):
    random.seed(seed); np.random.seed(seed)
    return _chain()

def apply(aug, wav):
    return aug(samples=np.asarray(wav, np.float32), sample_rate=config.SAMPLE_RATE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_augment.py -v`
Expected: PASS. (If `RoomSimulator` requires extra deps, drop it from `_chain()` and rely on `AddColoredNoise`+EQ+Mp3; note in commit.)

- [ ] **Step 5: Commit**

```bash
git add src/augment.py tests/test_augment.py
git commit -m "feat: train + fixed-seed val audio augmentation chain"
```

### Task 4.2: Bake the fixed-seed val_aug set

**Files:**
- Create: `src/bake_val_aug.py`

- [ ] **Step 1: Write `src/bake_val_aug.py`**

```python
# src/bake_val_aug.py
import json, glob
from src import config
from src.audio_io import load_wav_16k_mono, save_wav_16k
from src.augment import build_fixed_val_augmenter, apply

def main():
    config.VAL_AUG_DIR.mkdir(parents=True, exist_ok=True)
    aug = build_fixed_val_augmenter(seed=config.SEED)
    out_mf = open(config.VAL_AUG_DIR / "manifest.jsonl", "w")
    n = 0
    for m in glob.glob(str(config.VAL_CLEAN_DIR / "manifest_*.jsonl")):
        for line in open(m):
            row = json.loads(line)
            wav = load_wav_16k_mono(config.VAL_CLEAN_DIR / row["audio"])
            wav = apply(aug, wav)
            fn = "aug_" + row["audio"]
            save_wav_16k(config.VAL_AUG_DIR / fn, wav)
            row = dict(row, audio=fn)
            out_mf.write(json.dumps(row) + "\n"); n += 1
    out_mf.close()
    print(f"baked {n} augmented val clips")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after Gate B)**

Run: `.venv/bin/python -m src.bake_val_aug`
Expected: `baked ~160 augmented val clips` into `data/audio/val_aug/`.

- [ ] **Step 3: Commit**

```bash
git add src/bake_val_aug.py
git commit -m "feat: bake fixed-seed augmented val set"
```

---

## Phase 5 — Evaluation (baseline-first)

### Task 5.1: Dataset + collator (TDD)

**Files:**
- Create: `src/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import numpy as np, json, soundfile as sf
from transformers import WhisperProcessor
from src.dataset import WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding
from src import config

def _mini(tmp_path):
    d = tmp_path / "train"; d.mkdir()
    for i in range(2):
        sf.write(d / f"a{i}.wav", (0.1*np.random.randn(16000)).astype(np.float32), config.SAMPLE_RATE)
    mf = d / "manifest.jsonl"
    with open(mf, "w") as f:
        for i in range(2):
            f.write(json.dumps({"audio": f"a{i}.wav", "text": "claude opus", "terms": ["Claude Opus"]})+"\n")
    return d

def test_dataset_yields_features_and_labels(tmp_path):
    d = _mini(tmp_path)
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    ds = WhisperASRDataset([d], proc, augment=False)
    item = ds[0]
    assert "input_features" in item and "labels" in item
    assert len(item["labels"]) > 0

def test_collator_pads_batch(tmp_path):
    d = _mini(tmp_path)
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    ds = WhisperASRDataset([d], proc, augment=False)
    coll = DataCollatorSpeechSeq2SeqWithPadding(proc)
    batch = coll([ds[0], ds[1]])
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL (`ModuleNotFoundError: src.dataset`)

- [ ] **Step 3: Write `src/dataset.py`**

```python
# src/dataset.py
import json, glob
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from src import config
from src.audio_io import load_wav_16k_mono
from src.augment import build_train_augmenter, apply

class WhisperASRDataset(Dataset):
    def __init__(self, dirs, processor, augment=False):
        self.processor = processor
        self.augment = augment
        self.aug = build_train_augmenter() if augment else None
        self.rows = []
        for d in dirs:
            for m in glob.glob(str(d / "manifest*.jsonl")):
                for line in open(m):
                    r = json.loads(line); r["_dir"] = str(d); self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_wav_16k_mono(f"{r['_dir']}/{r['audio']}")
        if self.augment:
            wav = apply(self.aug, wav)
        feats = self.processor.feature_extractor(
            wav, sampling_rate=config.SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(r["text"]).input_ids
        return {"input_features": feats, "labels": labels}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    def __call__(self, features):
        input_feats = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
        label_feats = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: PASS (downloads the processor on first run).

- [ ] **Step 5: Commit**

```bash
git add src/dataset.py tests/test_dataset.py
git commit -m "feat: Whisper ASR dataset and padding collator"
```

### Task 5.2: Eval script (TDD on metrics wiring)

**Files:**
- Create: `src/eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test** (uses tiny real audio so it actually runs the model on CPU)

```python
# tests/test_eval.py
import numpy as np, json, soundfile as sf
from src.eval import evaluate
from src import config

def test_evaluate_returns_report(tmp_path):
    d = tmp_path / "val"; d.mkdir()
    sf.write(d / "a.wav", (0.01*np.random.randn(16000)).astype(np.float32), config.SAMPLE_RATE)
    with open(d / "manifest.jsonl", "w") as f:
        f.write(json.dumps({"audio": "a.wav", "text": "claude opus", "terms": ["Claude Opus"]})+"\n")
    rep = evaluate(config.MODEL_ID, d, device="cpu")
    assert "wer" in rep and "term_recall" in rep and 0.0 <= rep["wer"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: FAIL (`ModuleNotFoundError: src.eval`)

- [ ] **Step 3: Write `src/eval.py`**

```python
# src/eval.py
import json, glob, argparse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from src import config
from src.audio_io import load_wav_16k_mono
from src.metrics import compute_wer, term_recall
from src.normalize import load_terms

def _load_manifest(d):
    rows = []
    for m in glob.glob(str(d / "manifest*.jsonl")):
        for line in open(m):
            rows.append(json.loads(line))
    return rows

def evaluate(model_path, manifest_dir, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device).eval()
    rows = _load_manifest(manifest_dir)
    refs, hyps = [], []
    for r in rows:
        wav = load_wav_16k_mono(f"{manifest_dir}/{r['audio']}")
        feats = proc.feature_extractor(wav, sampling_rate=config.SAMPLE_RATE,
                                       return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats, language="en", task="transcribe", max_new_tokens=128)
        hyps.append(proc.batch_decode(ids, skip_special_tokens=True)[0])
        refs.append(r["text"])
    ts = load_terms()
    return {"n": len(rows), "wer": compute_wer(refs, hyps),
            "term_recall": term_recall(refs, hyps, ts), "refs": refs, "hyps": hyps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--manifest-dir", required=True)
    args = ap.parse_args()
    from pathlib import Path
    rep = evaluate(args.model, Path(args.manifest_dir))
    print(json.dumps({k: rep[k] for k in ("n", "wer")}, indent=2))
    print("term_recall.overall =", rep["term_recall"]["overall"])

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_eval.py -v`
Expected: PASS

- [ ] **Step 5: Run the BASELINE eval (un-finetuned model)**

Run:
```bash
.venv/bin/python -m src.eval --model openai/whisper-base.en --manifest-dir data/audio/val_clean
.venv/bin/python -m src.eval --model openai/whisper-base.en --manifest-dir data/audio/val_aug
```
Expected: prints baseline WER + term-recall on each set. **Record these numbers** — they are the "before" column for Gate C.

- [ ] **Step 6: Commit**

```bash
git add src/eval.py tests/test_eval.py
git commit -m "feat: WER + term-recall eval over any model/manifest"
```

---

## Phase 6 — Training (Gate C)

### Task 6.1: Training script

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write `src/train.py`**

```python
# src/train.py
import argparse, os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from src import config
from src.dataset import WhisperASRDataset, DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_wer
from src.normalize import load_terms
from src.metrics import term_recall

def build_compute_metrics(processor):
    def _fn(pred):
        ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(ids, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        ts = load_terms()
        return {"wer": compute_wer(refs, hyps),
                "term_recall": term_recall(refs, hyps, ts)["overall"]}
    return _fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    proc = WhisperProcessor.from_pretrained(config.MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_ID)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None

    train_ds = WhisperASRDataset([config.TRAIN_DIR], proc, augment=True)
    eval_ds = WhisperASRDataset([config.VAL_CLEAN_DIR], proc, augment=False)
    collator = DataCollatorSpeechSeq2SeqWithPadding(proc)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(config.CHECKPOINTS),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=200,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,            # keep EVERY checkpoint (user decision)
        predict_with_generate=True,
        generation_max_length=128,
        fp16=False, bf16=False,           # fp32 on MPS
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to=[],
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, compute_metrics=build_compute_metrics(proc),
        tokenizer=proc.feature_extractor,
    )
    trainer.train()
    trainer.save_model(str(config.CHECKPOINTS / "final"))
    print("done; checkpoints in", config.CHECKPOINTS)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Tiny smoke run (1 epoch, capped data)** — confirm the loop runs on MPS before the full run.

Temporarily point `TRAIN_DIR`/`VAL_CLEAN_DIR` at a small subset, or run:
```bash
.venv/bin/python -m src.train --epochs 1 --batch-size 2 --grad-accum 1
```
Expected: training starts, one epoch completes, a checkpoint dir appears under `checkpoints/`. If OOM: lower `--batch-size` to 4/2 and raise `--grad-accum`.

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: Whisper base.en finetune loop (MPS, save-all-checkpoints)"
```

### Task 6.2: Full training run

- [ ] **Step 1: Run the full finetune** (after Gate B + baseline recorded)

Run: `.venv/bin/python -m src.train --epochs 4 --batch-size 8 --grad-accum 2`
Expected: ~1–3 h; a checkpoint per epoch retained in `checkpoints/` (none pruned) + `checkpoints/final`.

- [ ] **Step 2: Verify all epoch checkpoints retained**

Run: `ls checkpoints/ | grep checkpoint`
Expected: one `checkpoint-*` per epoch (not just the best).

### Task 6.3: Finetuned eval + before/after table

- [ ] **Step 1: Eval the finetuned model on both val sets**

Run:
```bash
.venv/bin/python -m src.eval --model checkpoints/final --manifest-dir data/audio/val_clean
.venv/bin/python -m src.eval --model checkpoints/final --manifest-dir data/audio/val_aug
```
Expected: finetuned WER + term-recall for each set.

- [ ] **Step 2: Assemble the before/after table** (baseline from Task 5.2 Step 5 vs finetuned):

| Set | WER before | WER after | Term-recall before | Term-recall after |
|---|---|---|---|---|
| val_clean | … | … | … | … |
| val_aug | … | … | … | … |

> ### ⛔ GATE C — RESULTS REVIEW (STOP)
> Present the before/after table + per-term recall + sample qualitative diffs
> (e.g. `Claude Opus → cloud opus` fixed/not-fixed). Let the user decide next steps
> (more epochs, different checkpoint, hyperparameter changes, optional real-recording tier).

---

## Self-Review notes (coverage check)

- **Spec §2.1 terms** → Task 1.1. **§2.2 normalization** → Task 1.2 + `metrics`. **§2.3 corpus**
  → Tasks 2.1–2.2. **§2.4 splits** → `split_corpus` + voice constants in `config`.
  **§2.5 synthesis 7:3** → Tasks 3.2–3.4. **§2.6 augmentation** → Tasks 4.1–4.2.
  **§3 training (base.en, fp32 MPS, save-all)** → Task 6.1. **§3.6 save_total_limit=None** →
  Task 6.1 args. **§4 eval (WER + term-recall, baseline-first)** → Tasks 5.2, 6.3.
  **§5 env/deps/tests** → Phase 0 + per-task tests. **Gates A/B/C** → explicit STOP blocks.
- Out of scope (per spec): LoRA, multilingual, real-recording tier (optional follow-up).
```
