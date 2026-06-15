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


def test_homophone_coverage_excludes_negatives():
    ts = load_terms()
    corpus = [
        {"id": "p1", "text": "We scaled RoPE for the context window", "terms": ["RoPE"]},
        {"id": "n1", "text": "he tied the rope to the post", "terms": []},
        {"id": "n2", "text": "he bought more rope at the store", "terms": []},
    ]
    # only the one positive counts -> coverage 1: require 1 passes, require 2 fails
    validate_coverage(corpus, ts, only_terms=["RoPE"], min_coverage=1)
    with pytest.raises(CoverageError):
        validate_coverage(corpus, ts, only_terms=["RoPE"], min_coverage=2)


def test_split_is_disjoint_and_sized():
    corpus = [{"id": f"s{i}", "text": "x", "terms": []} for i in range(100)]
    train, val = split_corpus(corpus, val_frac=0.1, seed=42)
    assert len(val) == 10 and len(train) == 90
    assert not (set(s["id"] for s in train) & set(s["id"] for s in val))
