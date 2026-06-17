# src/normalize.py
import re
import yaml
from dataclasses import dataclass, field
from src import config


@dataclass
class TermSet:
    terms: list                      # list of dicts {canonical, category, alts, [everyday/anchors]}
    canonicals: list = field(default_factory=list)
    homophones: list = field(default_factory=list)  # canonicals with an `everyday` sense (Group B)
    _alt_map: dict = field(default_factory=dict)  # normalized alt -> normalized canonical


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # drop punctuation/hyphens
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cased_count(text: str, term: str) -> int:
    """Count case-SENSITIVE, boundary-delimited occurrences of `term` in raw `text`.
    Used for Group B homophones, where the everyday lowercase word (e.g. 'rope') must NOT
    count as the tech term ('RoPE'). Boundaries are non-alphanumeric so trailing punctuation
    is fine ('Bolt.' matches) but substrings are not ('RoPEx' does not match 'RoPE')."""
    return len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text))


def english_normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used for WER scoring."""
    return _norm(text)


def load_terms(path=None) -> TermSet:
    """Load the term set. Default = the single combined `terms.yaml` (official + hard, 92 terms).
    Group B homophones (terms with an `everyday` field) are deliberately kept OUT of the
    alt-map: they're scored case-sensitively (see `cased_count`) so the everyday word never
    collapses into the canonical (e.g. 'model' must not become 'Modal')."""
    paths = path if isinstance(path, (list, tuple)) else ([path] if path else [config.TERMS_PATH])
    terms = []
    for p in paths:
        terms.extend(yaml.safe_load(open(p))["terms"])
    ts = TermSet(
        terms=terms,
        canonicals=[t["canonical"] for t in terms],
        homophones=[t["canonical"] for t in terms if t.get("everyday")],
    )
    homset = set(ts.homophones)
    for t in terms:
        if t["canonical"] in homset:
            continue   # homophone — measured cased, excluded from alt rewriting
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
