# src/metrics.py
import jiwer
from src.normalize import english_normalize, canonicalize, load_terms


def compute_wer(refs, hyps):
    refs_n = [english_normalize(r) for r in refs]
    hyps_n = [english_normalize(h) for h in hyps]
    return float(jiwer.wer(refs_n, hyps_n))


def term_recall(refs, hyps, termset=None, strict=False):
    """For each canonical term occurrence in the (canonicalized) reference, count it a hit
    if the canonical surface form appears in the hypothesis.

    strict=False (lenient): the hypothesis is alt-canonicalized first, so a phonetic near-miss
      ('cloud opus') still counts as capturing 'Claude Opus' — measures *did it hear the term*.
    strict=True: the hypothesis is only normalized (no alt mapping), so the model must spell
      the term correctly — measures *did it transcribe the term right* (the finetuning target).
    Returns {'overall': float, 'per_term': {term: (hits, total)}}."""
    termset = termset or load_terms()
    per_term = {}
    total_hits = total = 0
    for ref, hyp in zip(refs, hyps):
        ref_c = canonicalize(ref, termset)
        hyp_c = english_normalize(hyp) if strict else canonicalize(hyp, termset)
        for t in termset.canonicals:
            t_n = english_normalize(t)
            n_ref = ref_c.count(t_n)
            if n_ref == 0:
                continue
            n_hit = min(n_ref, hyp_c.count(t_n))
            h, tot = per_term.get(t, (0, 0))
            per_term[t] = (h + n_hit, tot + n_ref)
            total_hits += n_hit
            total += n_ref
    return {
        "overall": (total_hits / total) if total else 0.0,
        "per_term": per_term,
    }
