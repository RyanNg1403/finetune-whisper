# src/build_corpus.py
import json
import random
import sys
from collections import Counter
from src import config
from src.normalize import load_terms, canonicalize, english_normalize, cased_count


class CoverageError(Exception):
    pass


def _count_terms(corpus, termset):
    """Per-term sentence coverage. Homophones (Group B) are counted case-sensitively on the
    raw text, so a negative sentence ('he tied the rope') does NOT count toward 'RoPE'."""
    homs = set(termset.homophones)
    counts = Counter()
    for row in corpus:
        text_c = canonicalize(row["text"], termset)
        for t in termset.canonicals:
            present = cased_count(row["text"], t) > 0 if t in homs else english_normalize(t) in text_c
            if present:
                counts[t] += 1
    return counts


def validate_coverage(corpus, termset=None, only_terms=None, min_coverage=None):
    termset = termset or load_terms()
    min_coverage = min_coverage or config.MIN_TERM_COVERAGE
    counts = _count_terms(corpus, termset)
    check = only_terms or termset.canonicals
    under = {t: counts.get(t, 0) for t in check if counts.get(t, 0) < min_coverage}
    if under:
        raise CoverageError(f"Under-covered terms (< {min_coverage}): {under}")
    return counts


def split_corpus(corpus, val_frac=0.08, seed=42):
    rng = random.Random(seed)
    shuffled = corpus[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_frac)
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    for s in train:
        s["split"] = "train"
    for s in val:
        s["split"] = "val"
    return train, val


def load_corpus(path=None):
    path = path or config.CORPUS_PATH
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ts = load_terms()
    corpus = load_corpus()
    counts = validate_coverage(corpus, ts)
    train, val = split_corpus(corpus)
    out = config.CORPUS_PATH
    with open(out, "w") as f:
        for row in train + val:
            f.write(json.dumps(row) + "\n")
    print(f"OK: {len(corpus)} sentences, {len(train)} train / {len(val)} val")
    print("min term coverage:", min(counts.values()))


if __name__ == "__main__":
    sys.exit(main())
