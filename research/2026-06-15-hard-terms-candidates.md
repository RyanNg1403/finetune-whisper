# Candidate "hard" terms for Whisper — experimental shortlist

**Date:** 2026-06-15
**Why:** The current finetune vocabulary (Claude Opus, RAG, MCP, …) turned out too easy —
stock `whisper-base.en` already handled most of it (lenient term-recall ~91% at baseline).
This is a search for terms that genuinely break Whisper, to make a sharper before/after.

**Status:** EXPERIMENTAL — for review only. Not added to the finetune dataset yet.
Tick the `[x]` boxes for the ones you consider relevant to AI-driven software development.

**How the evidence was produced:** each term was put in a natural carrier sentence,
synthesized with **Kokoro** (`am_michael`), and transcribed by stock **`whisper-base.en`**.
"base.en wrote" = what the stock model actually produced.

> Caveat: some errors are partly the TTS engine's pronunciation (e.g. Phi→"five"), not purely
> Whisper. But that's realistic of how these terms get spoken aloud, and the OOV/homophone
> cases (Qwen, Groq, Aider, GGUF) are genuine Whisper weaknesses regardless of engine.

---

## Tier 1 — Confirmed hard (base.en transcribed these WRONG)

| ✓ | Term | Category | base.en wrote | Failure mode |
|---|------|----------|---------------|--------------|
| [ ] | **Grok** | xAI model | "Grock" | neologism; clashes with Groq |
| [ ] | **Groq** | AI chip company | "GROC" | homophone with Grok *and* "rock" |
| [ ] | **Qwen** | Alibaba model | "QN3" | OOV — couldn't spell it at all |
| [ ] | **Cline** | coding agent | "climb" | near-homophone |
| [ ] | **Aider** | coding agent | "AITER" | OOV; "raider / aid-her" |
| [ ] | **GGUF** | model file format | "Giga file" | invented a plausible phrase |
| [ ] | **SGLang** | inference engine | "SG-LANG" | initialism + word blend |
| [ ] | **Phi** | Microsoft model | "5.4" | heard "Phi four" as a number |
| [ ] | **Sutskever** | Ilya, SSI / ex-OpenAI | "Sutsgiver" | Slavic surname |
| [ ] | **Amodei** | Dario, Anthropic CEO | "Amaldai" | Italian surname |
| [ ] | **ElevenLabs** | voice AI | "11 labs" | digits vs word |
| [ ] | **DeepSeek** | model lab | "Deep-Seek are one" | garbled in running speech |

## Tier 2 — Wobbled (partial errors)

| ✓ | Term | Category | base.en wrote |
|---|------|----------|---------------|
| [ ] | **LeCun** | Yann, Meta chief AI | "Yan LeCun" (dropped an *n*) |
| [ ] | **Kimi K2** | Moonshot model | "Kimike 2" |
| [ ] | **Unsloth** | finetuning lib | got "Unsloth" but mangled inner *LoRA* → "low RA" |

## Tier 3 — Untested, high-confidence hard (by reasoning)

| ✓ | Term | Category | Likely error |
|---|------|----------|--------------|
| [ ] | **Goose** | Block coding agent | pure homophone "goose" |
| [ ] | **v0** | Vercel app builder | "vee-oh" / "VO" / "vzero" |
| [ ] | **Pydantic** | Python lib | → "pedantic" |
| [ ] | **RoPE** | positional encoding | → "rope" |
| [ ] | **Arthur Mensch** | Mistral CEO | surname "Mensch" / "mench" |
| [ ] | **Aravind Srinivas** | Perplexity CEO | full name mangled |
| [ ] | **llama.cpp** | inference runtime | "llama dot c-p-p" / "llama cpp" |
| [ ] | **Codestral** | Mistral code model | "code-stral" / "Costral" |

---

## Carrier sentences used (Tier 1 & 2)

- Grok — "xAI released Grok four with much better reasoning."
- Groq — "We run inference on Groq for very low latency."
- Qwen — "Qwen three is now my default open model."
- Cline — "I switched from Copilot to Cline for agentic edits."
- Aider — "Aider commits every change straight to git."
- GGUF — "Download the GGUF file and load it in Ollama."
- SGLang — "SGLang handles structured generation cleanly."
- Phi — "Microsoft's Phi four punches above its weight."
- Sutskever — "Ilya Sutskever started a new lab."
- Amodei — "Dario Amodei runs Anthropic."
- ElevenLabs — "We use ElevenLabs for voice synthesis."
- DeepSeek — "DeepSeek R1 reasons step by step."
- LeCun — "Yann LeCun is skeptical of large language models."
- Kimi — "Moonshot's Kimi K2 is a strong open model."
- Unsloth — "Unsloth makes LoRA fine-tuning twice as fast."

## Terms that base.en actually got RIGHT (drop candidates)

RWKV, Axolotl, Gemma, Mixtral, vLLM — already transcribed correctly, so low value for a
"hard" set (RWKV/vLLM are spelled-out initialisms the model handles; Axolotl/Gemma/Mixtral
are evidently in-distribution enough).

---

## Next steps (when you're ready — not now)

1. You mark the relevant boxes above.
2. Add the keepers to `data/terms.yaml` (with `spoken` overrides where TTS mispronounces,
   e.g. Qwen→"chwen", Phi→"fee", Groq/Grok disambiguation).
3. Author corpus sentences for them, regenerate that slice, and re-run the before/after —
   this time the baseline should genuinely struggle, making the finetuning gain dramatic.
