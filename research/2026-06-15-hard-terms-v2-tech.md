# Hard terms v2 — tech/software only, context-aware (40 candidates)

**Date:** 2026-06-15 · supersedes v1 (`2026-06-15-hard-terms-candidates.md`)
**Preferences applied:** (1) no human names — tech/software only; (2) homophone/ambiguous
terms must be **context-anchored** — only transcribed the technical way when neighboring
words support it (your RoPE example).

**Evidence:** `[base.en: "…"]` = what stock `whisper-base.en` wrote when the term was spoken
(Kokoro, `am_michael`) in the listed carrier sentence. Tick `[x]` for keepers.

---

## Group A — OOV / spelling-hard (base.en mis-spells the token itself)

These fail regardless of context — the model has no idea how to spell them. Highest value.

| ✓ | Term | Category | base.en wrote |
|---|------|----------|---------------|
| [ ] | **Qwen** | Alibaba model | "QN3" |
| [ ] | **Groq** | inference chip (LPU) | "GROC" |
| [ ] | **Cline** | coding agent | "climb" |
| [ ] | **Aider** | coding agent | "AITER" |
| [ ] | **GGUF** | model file format | "Giga file" / "G-Guff" |
| [ ] | **SGLang** | inference engine | "SG-LANG" |
| [ ] | **DeepSeek** | model lab | "Deep-Seek are one" / "Deep-Seq" |
| [ ] | **ElevenLabs** | voice AI | "11 labs" |
| [ ] | **Yi** | 01.AI model | (dropped entirely) |
| [ ] | **Zed** | code editor (Rust) | "Z" |
| [ ] | **Pydantic** | Python validation lib | "pedantic" |
| [ ] | **llama.cpp** | local inference runtime | "LAMA, CPP" |
| [ ] | **safetensors** | weight format | "saffit sensors" |
| [ ] | **DSPy** | prompt-compilation framework | "Despy" |
| [ ] | **Codestral** | Mistral code model | "CodeStrull" |
| [ ] | **OpenHands** | autonomous agent platform | "Open Hand" |
| [ ] | **Kimi** | Moonshot model (K2) | "Kimike 2" |
| [ ] | **Unsloth** | finetuning lib | ok, but inner LoRA → "low RA" |
| [ ] | **QLoRA** | quantized finetuning | "Q-Lo RA" |
| [ ] | **vLLM** | inference engine | (in v1 set; spelled-out, borderline) |

## Group B — Homophone / context-dependent (MUST be anchored by neighbors)

Your rule: transcribe the **tech** spelling **only** when the context fits; otherwise the
everyday word. Corpus sentences for these must surround the term with the **anchor words**.
`★` = base.en fails *even with* context (high priority); others base.en gets right *with*
context (so the value is robustness + not false-triggering in non-tech speech).

| ✓ | Term | Tech sense | Anchor context (must co-occur) | Everyday word to NOT trigger |
|---|------|-----------|-------------------------------|------------------------------|
| [ ] | **RoPE** ★ | rotary positional encoding | positional encoding, context window, attention, extrapolation | "rope" [base.en: "row PE"] |
| [ ] | **Grok** | xAI's model | xAI, model, chatbot, Grok 4 | "grok" (to understand) / "rock" |
| [ ] | **Phi** ★ | Microsoft small model | Microsoft, SLM, Phi-4, small model | "fee" / math "phi" [base.en: "5.4"] |
| [ ] | **Goose** | Block's coding agent | coding agent, Block, terminal, open source | the bird |
| [ ] | **Continue** | VS Code AI extension | extension, VS Code, autocomplete, open source | the verb "continue" |
| [ ] | **Modal** | serverless GPU platform | serverless, GPU, deploy, container | UI/logic "modal" |
| [ ] | **Together** | Together AI (inference host) | Together AI, inference, host, endpoint | "together" |
| [ ] | **Bolt** | bolt.new app builder | bolt.new, StackBlitz, app builder, prompt | "bolt" |
| [ ] | **Granite** | IBM model family | IBM, model, enterprise | the rock |
| [ ] | **Falcon** | TII model | model, weights, TII, open weights | the bird |
| [ ] | **Mamba** | state-space architecture | architecture, state space, sequence, attention | the snake |
| [ ] | **Triton** | GPU kernel language | kernel, GPU, compiler, OpenAI | the moon |
| [ ] | **Sora** | OpenAI video model | video, generation, OpenAI, text-to-video | the name |
| [ ] | **Flux** | image diffusion model | image, diffusion, generation, Black Forest | "flux" |

## Group C — base.en already handles these (low priority / optional negatives)

base.en transcribed these correctly even in isolation/context, so low finetuning value —
but useful as "negative"/control terms, or easy wins: **Mixtral, Axolotl, RWKV, GPTQ, AWQ,
bitsandbytes, TensorRT, FlashAttention, speculative decoding, mixture of experts, DPO, GRPO,
v0, SWE-bench, ComfyUI, Cody**.

---

## Carrier sentences used (Group A & B, for reference)

- RoPE — "We extended the context window by scaling RoPE for positional encoding." → "row PE"
- Yi — "The Yi model from zero one AI is bilingual." → (dropped)
- Zed — "Zed is a fast collaborative code editor written in Rust." → "Z"
- Pydantic — "We validate the agent output with Pydantic schemas." → "pedantic"
- safetensors — "Save the weights as safetensors instead of pickle." → "saffit sensors"
- llama.cpp — "I run the GGUF in llama.cpp on my laptop." → "LAMA, CPP"
- DSPy — "DSPy compiles prompts into optimized pipelines." → "Despy"
- Codestral — "Codestral is Mistral's model for code completion." → "CodeStrull"
- OpenHands — "OpenHands runs an autonomous agent in a sandbox." → "Open Hand"

## Count

Group A: 20 · Group B: 14 · → **34 priority candidates** (+ Group C pool of ~16 if you want
to push toward 50). Tick the keepers and I'll wire them into `terms.yaml` (with `spoken`
overrides + per-term anchor-context rules for the corpus) when you're ready.
