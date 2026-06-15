# tests/test_normalize.py
from src.normalize import load_terms, canonicalize, english_normalize, to_spoken


def test_load_terms():
    ts = load_terms()
    assert len(ts.terms) >= 40
    assert "Claude Opus" in ts.canonicals


def test_english_normalize_lowercases_and_strips_punct():
    assert english_normalize("Claude Opus, really!") == "claude opus really"


def test_canonicalize_maps_alts_to_canonical():
    # "cloud opus" is an alt of "Claude Opus" -> canonical surface form
    out = canonicalize("i love cloud opus")
    assert "claude opus" in out


def test_canonicalize_handles_acronyms():
    assert "mcp" in canonicalize("we use m c p servers")


def test_to_spoken_rewrites_word_acronyms_only():
    ts = load_terms()
    # RAG -> 'rag' (word), MCP stays (true initialism), transcript-only terms untouched
    out = to_spoken("Our RAG layer behind MCP uses LoRA.", ts)
    assert "rag" in out and "RAG" not in out      # word-acronym lowered for TTS
    assert "MCP" in out                            # initialism preserved (spelled out)
    assert "lora" in out and "LoRA" not in out
