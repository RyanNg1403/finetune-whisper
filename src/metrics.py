# src/metrics.py
import re
import jiwer
from src.normalize import english_normalize, canonicalize, cased_count, load_terms


def compute_wer(refs, hyps):
    refs_n = [english_normalize(r) for r in refs]
    hyps_n = [english_normalize(h) for h in hyps]
    return float(jiwer.wer(refs_n, hyps_n))


def _bcount(text, term):
    """Word-boundary-delimited occurrence count in already-normalized (lowercase) text.
    Boundaries are non-alphanumeric, so 'rag' does not match inside 'storage' — unlike a
    naive substring .count(), which over-counts short terms and inflates the denominator."""
    return len(re.findall(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))


def term_recall(refs, hyps, termset=None, strict=False):
    """For each canonical term occurrence in the reference, count it a hit if the canonical
    surface form appears in the hypothesis.

    Non-homophone terms (Group A + official): matched on normalized text, word-boundary delimited.
      The REFERENCE is matched literally (no alt-map): it is ground-truth canonical text, so counting
      its actual term occurrences is the goal. Alt-mapping the reference would rewrite everyday words
      that happen to be phonetic aliases ('when'->qwen, 'hero'->kiro) into phantom term occurrences and
      depress recall. Only the hypothesis is alt-mapped, and only in lenient mode:
      strict=False (lenient): the hypothesis is alt-canonicalized first, so a phonetic near-miss
        ('cloud opus') still counts as capturing 'Claude Opus' — *did it hear the term*.
      strict=True: the hypothesis is only normalized (no alt mapping), so the model must spell
        the term correctly — *did it transcribe the term right* (the finetuning target).
    Homophone terms (Group B, with an `everyday` sense): always matched CASE-SENSITIVELY on the
      raw text, so 'rope' never counts as 'RoPE'. `strict` has no effect on these — exact spelling
      is the whole point. Over-triggering on everyday usage is measured separately by
      `false_triggers`.
    Returns {'overall': float, 'per_term': {term: (hits, total)}}."""
    termset = termset or load_terms()
    homs = set(termset.homophones)
    per_term = {}
    total_hits = total = 0
    for ref, hyp in zip(refs, hyps):
        ref_n = english_normalize(ref)                                      # literal reference, no alt-map
        hyp_c = english_normalize(hyp) if strict else canonicalize(hyp, termset)
        for t in termset.canonicals:
            if t in homs:
                n_ref = cased_count(ref, t)
                if n_ref == 0:
                    continue
                n_hit = min(n_ref, cased_count(hyp, t))
            else:
                t_n = english_normalize(t)
                n_ref = _bcount(ref_n, t_n)
                if n_ref == 0:
                    continue
                n_hit = min(n_ref, _bcount(hyp_c, t_n))
            h, tot = per_term.get(t, (0, 0))
            per_term[t] = (h + n_hit, tot + n_ref)
            total_hits += n_hit
            total += n_ref
    return {
        "overall": (total_hits / total) if total else 0.0,
        "per_term": per_term,
    }


def false_triggers(hyps, neg_terms, termset=None):
    """Over-trigger rate for Group B homophones on NEGATIVE (everyday-sense) sentences.

    `neg_terms[i]` is the list of homophone canonicals that sentence i is a negative for
    (everyday usage — the model must NOT emit the tech spelling). A false trigger = the
    cased canonical appears in the hypothesis anyway. Lower is better.
    Returns {'overall': float, 'per_term': {term: (false_triggers, total)}}."""
    termset = termset or load_terms()
    per_term = {}
    trig = total = 0
    for hyp, terms in zip(hyps, neg_terms):
        for t in terms:
            ft = 1 if cased_count(hyp, t) > 0 else 0
            tr, to = per_term.get(t, (0, 0))
            per_term[t] = (tr + ft, to + 1)
            trig += ft
            total += 1
    return {
        "overall": (trig / total) if total else 0.0,
        "per_term": per_term,
    }
