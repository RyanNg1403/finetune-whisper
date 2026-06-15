# tests/test_qc.py
from src.qc import clip_passes, accept_or_regenerate


def test_clean_term_passes():
    assert clip_passes("a tight rag layer", ["RAG"]) is True


def test_garbled_acronym_fails():
    assert clip_passes("a tight raggy layer", ["RAG"]) is False
    assert clip_passes("a tight ragged layer", ["RAG"]) is False


def test_benign_asr_quirk_passes():
    # 'Anthropix' is a near-miss the reference ASR makes even on clean audio -> accept
    assert clip_passes("Anthropix MCP plus a tight rag layer", ["Anthropic", "MCP", "RAG"]) is True


def test_known_alt_passes_via_canonicalize():
    # 'laura' is a registered alt of LoRA; 'gpt five' an alt of GPT-5
    assert clip_passes("we ran a quick laura pass", ["LoRA"]) is True
    assert clip_passes("the gpt five release", ["GPT-5"]) is True


def test_all_terms_required():
    # RAG clean but MCP missing entirely -> clip fails
    assert clip_passes("a tight rag layer", ["RAG", "MCP"]) is False


def test_accept_or_regenerate_no_ref_accepts_first():
    calls = {"n": 0}
    def make():
        calls["n"] += 1
        return "wav"
    wav, passed, tries = accept_or_regenerate(make, ["RAG"], ref=None, max_tries=3)
    assert wav == "wav" and passed is True and tries == 1 and calls["n"] == 1


def test_accept_or_regenerate_no_terms_accepts_first():
    wav, passed, tries = accept_or_regenerate(lambda: "w", [], ref=object(), max_tries=3)
    assert passed is True and tries == 1
