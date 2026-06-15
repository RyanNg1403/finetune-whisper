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
    # after canonicalization the bad hyp also maps to claude opus -> still a hit
    r_bad = term_recall(refs, hyps_bad, ts)
    assert r_bad["overall"] == 1.0


def test_term_recall_strict_requires_correct_spelling():
    ts = load_terms()
    refs = ["i tried claude opus today"]
    hyps = ["i tried cloud opus today"]   # phonetic near-miss
    # lenient: alt 'cloud opus' canonicalizes to claude opus -> hit
    assert term_recall(refs, hyps, ts)["overall"] == 1.0
    # strict: must spell it right -> miss
    assert term_recall(refs, hyps, ts, strict=True)["overall"] == 0.0


def test_term_recall_misses_unrelated():
    ts = load_terms()
    refs = ["i tried claude opus today"]
    hyps = ["i tried the weather today"]
    assert term_recall(refs, hyps, ts)["overall"] == 0.0
