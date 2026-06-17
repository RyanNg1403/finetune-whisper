# Whisper A/B — voice transcription comparison

A small local web app: record your voice and compare how stock `whisper-base.en`
and a chosen finetune checkpoint transcribe AI-software vocabulary, side by side.
Known dataset terms are highlighted (correct spelling in green), with a per-model
hit tally, live waveform, and per-model latency.

## Run

```bash
# from the repo root, with the project venv
PYTHONPATH=. .venv/bin/python -m webapp.server
# then open http://127.0.0.1:8765
```

By default it uses MPS. To force CPU (e.g. while a finetune is using the GPU):

```bash
PYTHONPATH=. WHISPER_DEVICE=cpu .venv/bin/python -m webapp.server
```

Pick a checkpoint from the dropdown (every saved epoch checkpoint is listed, plus the
archived prior run), tap the dial to record a sentence, tap again to stop. Mic capture
requires `localhost`/`127.0.0.1` (a secure context) — which this server already is.

## How scoring works

The candidate term set is everything either model *heard* (lenient, alt-map forgiving)
or *spelled correctly*. A model scores a "hit" only on an exact cased spelling — the same
strict criterion as `src.metrics.term_recall`. So "baseline heard it but misspelled it"
shows up as a red miss chip while the finetune marks it green.

No new dependencies: stdlib `http.server` + the project's existing `torch`/`transformers`/
`soundfile` stack. Audio is captured in-browser as 16 kHz mono WAV.
