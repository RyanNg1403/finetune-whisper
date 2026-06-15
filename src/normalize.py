# src/normalize.py
import re
import yaml
from dataclasses import dataclass, field
from src import config


@dataclass
class TermSet:
    terms: list                      # list of dicts {canonical, category, alts}
    canonicals: list = field(default_factory=list)
    _alt_map: dict = field(default_factory=dict)  # normalized alt -> normalized canonical


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # drop punctuation/hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s


def english_normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for WER scoring."""
    return _norm(text)


def load_terms(path=None) -> TermSet:
    path = path or config.TERMS_PATH
    data = yaml.safe_load(open(path))
    terms = data["terms"]
    ts = TermSet(terms=terms, canonicals=[t["canonical"] for t in terms])
    for t in terms:
        canon_norm = _norm(t["canonical"])
        ts._alt_map[canon_norm] = canon_norm
        for alt in t.get("alts", []):
            ts._alt_map[_norm(alt)] = canon_norm
    return ts


def canonicalize(text: str, termset: TermSet = None) -> str:
    """Normalize text, then rewrite any known alt phrase to its canonical (normalized) form.
    Longer alts are matched first so multi-word terms win over single words."""
    termset = termset or load_terms()
    out = _norm(text)
    for alt in sorted(termset._alt_map, key=lambda a: -len(a.split())):
        canon = termset._alt_map[alt]
        if alt == canon:
            continue
        out = re.sub(rf"\b{re.escape(alt)}\b", canon, out)
    return out


def to_spoken(text: str, termset: TermSet = None) -> str:
    """Build the TTS input text from a (canonical) transcript: replace each term whose
    `spoken` form differs from its canonical with that spoken form, so the TTS pronounces
    word-acronyms (RAG -> 'rag') as words. The transcript itself is left untouched."""
    termset = termset or load_terms()
    out = text
    for t in termset.terms:
        spoken = t.get("spoken")
        if spoken and spoken != t["canonical"]:
            out = re.sub(rf"\b{re.escape(t['canonical'])}\b", spoken, out)
    return out
