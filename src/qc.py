# src/qc.py
"""Back-transcription QC: transcribe a synthesized clip with a strong reference ASR model
and check each target term was rendered intelligibly. Uses fuzzy matching so benign ASR
quirks (e.g. 'Anthropix' for Anthropic) pass, but TTS garbles (e.g. 'raggy' for RAG) fail."""
import torch
from rapidfuzz import fuzz
from src import config
from src.normalize import canonicalize, english_normalize, load_terms

_REF = {}


class ReferenceASR:
    def __init__(self, model_id="openai/whisper-small.en", device=None):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(self.device).eval()

    def transcribe(self, wav):
        feats = self.proc.feature_extractor(
            wav, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            out = self.model.generate(feats, max_new_tokens=128)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()


def get_reference(model_id="openai/whisper-small.en"):
    if model_id not in _REF:
        _REF[model_id] = ReferenceASR(model_id)
    return _REF[model_id]


def _term_ok(term, hyp_toks, threshold=85):
    tw = english_normalize(term).split()
    n = len(tw)
    if n == 0:
        return True
    for i in range(len(hyp_toks) - n + 1):
        window = hyp_toks[i:i + n]
        if window == tw:                                    # exact whole-token match
            return True
        if fuzz.ratio(" ".join(tw), " ".join(window)) >= threshold:  # benign ASR quirk
            return True
    return False


def clip_passes(hyp, clip_terms, termset=None, threshold=85):
    """True if every term in clip_terms is intelligibly present in the reference hypothesis."""
    termset = termset or load_terms()
    hyp_toks = canonicalize(hyp, termset).split()
    return all(_term_ok(t, hyp_toks, threshold) for t in clip_terms)


def accept_or_regenerate(make_wav, terms, ref, max_tries=3, termset=None):
    """Synthesize via make_wav(); if QC fails, regenerate a fresh take up to max_tries.
    Returns (wav, passed, tries). With no reference or no terms, accepts the first take."""
    wav = make_wav()
    if ref is None or not terms:
        return wav, True, 1
    for attempt in range(1, max_tries + 1):
        if clip_passes(ref.transcribe(wav), terms, termset):
            return wav, True, attempt
        if attempt < max_tries:
            wav = make_wav()
    return wav, False, max_tries
