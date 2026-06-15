# tests/test_metrics.py
from src.normalize import load_terms
from src.metrics import compute_wer, term_recall, false_triggers


def test_term_recall_homophone_is_case_sensitive():
    ts = load_terms()
    refs = ["the RoPE trick was applied here"]   # 'RoPE' is the only term present
    # correct tech spelling -> hit
    assert term_recall(refs, ["the RoPE trick was applied here"], ts)["per_term"]["RoPE"] == (1, 1)
    # everyday spelling 'rope' must NOT count as the term, even leniently
    assert term_recall(refs, ["the rope trick was applied here"], ts)["per_term"]["RoPE"] == (0, 1)
    assert term_recall(refs, ["the rope trick was applied here"], ts, strict=True)["per_term"]["RoPE"] == (0, 1)


def test_false_triggers_flags_overtrigger_on_negatives():
    ts = load_terms()
    hyps = ["he tied the RoPE to the post", "he tied the rope to the post"]
    neg = [["RoPE"], ["RoPE"]]   # both are everyday-sense negatives
    r = false_triggers(hyps, neg, ts)
    assert r["per_term"]["RoPE"] == (1, 2)   # first wrongly emits 'RoPE'; second is correct
    assert r["overall"] == 0.5


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
