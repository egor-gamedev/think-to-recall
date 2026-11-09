"""English extraction path (ARCHITECTURE_V2 §3.2): spaCy `en_core_web_sm` (MIT).

Concepts of an English sentence = lemmatized noun chunks (determiners / pronouns dropped) + named
entities. Russian keeps the pymorphy3 + rules path (`ctxgraph.v2.extract.concepts_of`).
"""

from __future__ import annotations

from functools import lru_cache

_DROP_POS = {"DET", "PRON", "PART", "PUNCT", "NUM", "CCONJ", "ADP"}


@lru_cache(maxsize=1)
def nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["textcat"])


def available() -> bool:
    try:
        nlp()
        return True
    except Exception:  # noqa: BLE001 — spaCy or the model missing: fall back to the rules path
        return False


_CACHE: dict[str, list[tuple[str, str, str]]] = {}


def precompute(texts: list[str], batch_size: int = 256, n_process: int = 1) -> None:
    """Run spaCy over many sentences at once (`nlp.pipe`) and cache the concepts: bulk corpus
    ingestion then hits the cache instead of calling spaCy sentence by sentence. Same result."""
    todo = [t for t in dict.fromkeys(texts) if t not in _CACHE]
    for t, doc in zip(todo, nlp().pipe(todo, batch_size=batch_size, n_process=n_process), strict=True):
        _CACHE[t] = _concepts_doc(doc)


def concepts_en(text: str) -> list[tuple[str, str, str]]:
    """(lemmatized key, surface, kind) for an English sentence."""
    if text in _CACHE:
        return list(_CACHE[text])
    return _concepts_doc(nlp()(text))


def _concepts_doc(doc) -> list[tuple[str, str, str]]:
    out, seen = [], set()

    def add(key: str, surf: str, kind: str) -> None:
        key = " ".join(key.lower().split())
        if len(key) >= 3 and key not in seen and not key.isdigit():
            seen.add(key)
            out.append((key, surf, kind))

    for ent in doc.ents:
        if ent.label_ in ("CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY"):
            continue
        # the possessive marker is not part of a name: «Maria Lopez's» -> «maria lopez»
        toks = [t for t in ent if t.pos_ not in ("DET", "PUNCT") and t.dep_ != "case"]
        if toks:
            add(" ".join(t.lemma_ for t in toks), ent.text, f"ent:{ent.label_}")
    for ch in doc.noun_chunks:
        # a possessor is its own concept (entity / chunk), not part of the possessed one:
        # «Maria Lopez's pottery workshop» -> «pottery workshop»
        poss = {d.i for t in ch if t.dep_ == "poss" for d in t.subtree}
        own = [t for t in ch if t.i in poss and t.pos_ not in _DROP_POS and not t.is_stop and t.dep_ != "case"]
        if own:
            add(" ".join(t.lemma_ for t in own), " ".join(t.text for t in own), "np")
        toks = [t for t in ch if t.pos_ not in _DROP_POS and not t.is_stop and t.i not in poss and t.dep_ != "case"]
        if toks:
            add(" ".join(t.lemma_ for t in toks), ch.text, "np")
    return out
