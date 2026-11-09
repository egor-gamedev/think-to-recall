"""Extraction without a generative model (ARCHITECTURE_V2 §3.2).

- Sentences: code blocks become separate `code` fragments (identifiers only, no text embedding);
  prose is split into sentences (list items are sentences).
- Concepts: our extractor (`ctxgraph.ingest.extract_entities` + `is_concept`: YAKE keyphrases,
  noun lemmas via pymorphy3, identifiers). English path via spaCy is plugged in by `EnglishExtractor`
  when spaCy is installed (§3.2, to be checked on 30 English passages before external benchmarks).
- Names: optional GLiNER (`NameExtractor`) for person / organization / location / software /
  product; merged with concepts by canonical form.
- Privacy: `ctxgraph.privacy.redact` runs before anything is stored; only type and offset of each
  redaction are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ctxgraph import text as T
from ctxgraph.ingest import extract_entities
from ctxgraph.privacy import redact
from ctxgraph.sources.experience import is_concept

_CODE = re.compile(r"```[^\n]*\n?(.*?)```", re.S)
_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MARK = re.compile(r"\[REDACTED:(\w+)\]")
_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
#: English sentences go through spaCy (ARCHITECTURE_V2 §3.2) when it is installed.
ENGLISH_PATH = True


@dataclass
class Piece:
    idx: int
    kind: str  # sentence | code
    text: str
    lang: str
    concepts: list[tuple[str, str, str]] = field(default_factory=list)  # (canonical, surface, kind)


def scrub(text: str) -> tuple[str, list[dict]]:
    """Redact secrets/PII; return clean text and [{type, offset}] (no values)."""
    clean = redact(text).text
    return clean, [{"type": m.group(1), "offset": m.start()} for m in _MARK.finditer(clean)]


def sentences(text: str, min_words: int = 3) -> list[tuple[str, str]]:
    """[(kind, text)] in reading order: prose sentences and code blocks."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _CODE.finditer(text):
        out += [("sentence", s) for s in _prose(text[pos : m.start()], min_words)]
        if m.group(1).strip():
            out.append(("code", m.group(1).strip()))
        pos = m.end()
    out += [("sentence", s) for s in _prose(text[pos:], min_words)]
    return out


def _prose(chunk: str, min_words: int) -> list[str]:
    res = []
    for s in _SPLIT.split(chunk):
        s = _BULLET.sub("", s).strip()
        if len(s.split()) >= min_words:
            res.append(s)
    return res


def concepts_of(text: str, kind: str = "sentence") -> list[tuple[str, str, str]]:
    """(canonical, surface, kind) concepts of a piece; code pieces yield identifiers only."""
    plain = _MARK.sub(" ", text)
    if kind == "code":
        seen, out = set(), []
        for m in _IDENT.finditer(plain):
            c = T.canonical(m.group())
            if c and c not in seen and c not in T.STOP_ALL:
                seen.add(c)
                out.append((c, m.group(), "identifier"))
        return out[:12]
    if T.detect_lang(plain) == "en" and ENGLISH_PATH:
        from ctxgraph.v2 import english

        if english.available():
            return english.concepts_en(plain)
    out, seen = [], set()
    for e in extract_entities(plain):
        if not is_concept(e.canonical, e.kind, e.surface):
            continue
        key = phrase_key(e.canonical)
        if key not in seen:
            seen.add(key)
            out.append((key, e.surface, e.kind))
    return out


def phrase_key(phrase: str) -> str:
    """Concept identity: every word lemmatized («экрана создания кошелька» = «экран создания
    кошелька» -> «экран создание кошелек»). Identifiers keep their form."""
    words = phrase.split()
    if len(words) == 1 and not T.is_cyrillic(phrase):
        return phrase
    return " ".join(T.lemma(w) for w in words) or phrase


class NameExtractor:
    """GLiNER names (§3.2): only named entities; concepts come from our extractor."""

    LABELS = ["person", "organization", "location", "software", "product"]

    def __init__(self, model: str = "urchade/gliner_multi-v2.1", threshold: float = 0.5) -> None:
        from gliner import GLiNER

        self.m = GLiNER.from_pretrained(model)
        self.threshold = threshold

    def __call__(self, text: str) -> list[tuple[str, str, str]]:
        ents = self.m.predict_entities(text[:2000], self.LABELS, threshold=self.threshold)
        out = []
        for e in ents:
            surf = e["text"].strip()
            canon = phrase_key(" ".join(T.words(surf))) or surf.lower()
            out.append((canon, surf, e["label"]))
        return out


def merge_nested(concepts: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Within one sentence, a concept whose words are a contiguous part of a longer concept is the
    same mention: keep only the fullest form ("экран создания кошелька" absorbs "экран создания",
    "создание кошелька", "создание", "кошелек")."""
    toks = {c: c.split() for c, _, _ in concepts}
    keep = []
    for c, surf, kind in sorted(concepts, key=lambda x: (-len(toks[x[0]]), x[0])):
        t = toks[c]
        nested = any(
            len(toks[k]) > len(t) and any(toks[k][i : i + len(t)] == t for i in range(len(toks[k]) - len(t) + 1))
            for k, _, _ in keep
        )
        if not nested:
            keep.append((c, surf, kind))
    order = {c: i for i, (c, _, _) in enumerate(concepts)}
    return sorted(keep, key=lambda x: order[x[0]])


def pieces(text: str, names=None, *, nested: bool = True) -> list[Piece]:
    out = []
    for i, (kind, s) in enumerate(sentences(text)):
        p = Piece(i, kind, s, T.detect_lang(s), concepts_of(s, kind))
        if names is not None and kind == "sentence":
            have = {c for c, _, _ in p.concepts}
            p.concepts += [x for x in names(s) if x[0] not in have]
        if nested and kind == "sentence":
            p.concepts = merge_nested(p.concepts)
        out.append(p)
    return out
