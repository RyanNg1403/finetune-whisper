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
    assert "claude opus" in out


def test_canonicalize_handles_acronyms():
    assert "mcp" in canonicalize("we use m c p servers")
