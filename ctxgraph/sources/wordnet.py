"""WordNet (английский, NLTK + OMW) и RuWordNet (русский): гипонимы, гиперонимы, братья, части."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ctxgraph.sources.base import Link, Source
from ctxgraph.store import Node

NLTK_DIR = Path(__file__).resolve().parents[2] / ".cache" / "nltk"
MAX_CHILDREN = 12
#: Узел с большим числом гипонимов — хаб (person, device, structure): его братья бесполезны,
#: а из детей берём только частотные.
HUB_DEGREE = 30


def _freq(syn) -> int:
    """Частота смысла по корпусу SemCor (счётчики лемм WordNet)."""
    return sum(lm.count() for lm in syn.lemmas())


def _named(syns) -> list:
    return sorted(syns, key=lambda s: s.name())


def _by_freq(syns, limit: int, *, drop_rare: bool = False) -> list:
    # Ничьи по частоте — по имени: порядок связей в NLTK зависит от PYTHONHASHSEED.
    ranked = sorted(syns, key=lambda s: (-_freq(s), s.name()))
    if drop_rare:
        ranked = [s for s in ranked if _freq(s) > 0]
    return ranked[:limit]


def _pretty(name: str) -> str:
    return name.replace("_", " ")


@lru_cache(maxsize=1)
def _wn():
    import nltk

    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    from nltk.corpus import wordnet

    wordnet.ensure_loaded()
    return wordnet


class WordNetSource(Source):
    name = "wordnet"
    prefix = "wn:"

    def _node(self, syn) -> Node:
        return Node(
            f"wn:{syn.name()}",
            _pretty(syn.lemma_names()[0]),
            self.name,
            "en",
            {"gloss": syn.definition()[:160]},
        )

    def link(self, term: str, lang: str) -> list[Node]:
        if lang != "en":
            return []
        wn = _wn()
        key = term.replace(" ", "_")
        syns = wn.synsets(key, pos=wn.NOUN)

        def sense_count(s) -> int:
            return sum(lm.count() for lm in s.lemmas() if lm.name().lower() == key.lower())

        counts = [sense_count(s) for s in syns]
        top = max(counts, default=0)
        # Редкие смыслы не используем: при наличии частотных — только те, что ≥ 10% от главного.
        keep = [s for s, c in zip(syns, counts, strict=True) if top == 0 or c >= max(1, 0.1 * top)]
        out = []
        for s in keep[:4] if top else keep[:2]:
            n = self._node(s)
            n.meta["freq"] = sense_count(s)
            out.append(n)
        return out

    def neighbors(self, node: Node) -> list[Link]:
        wn = _wn()
        syn = wn.synset(node.id[3:])
        out: list[Link] = []
        hypo = syn.hyponyms()
        hub = len(hypo) > HUB_DEGREE
        for h in _by_freq(hypo, 8 if hub else MAX_CHILDREN, drop_rare=hub):
            out.append(Link(self._node(h), "подвид", "deeper", 0.9, self.name))
        for m in _named(syn.part_meronyms() + syn.member_meronyms())[:4]:
            out.append(Link(self._node(m), "часть/член", "deeper", 0.7, self.name))
        for p in _named(syn.hypernyms())[:2]:
            out.append(Link(self._node(p), "обобщение", "side", 0.8, self.name))
            sibs = p.hyponyms()
            if len(sibs) > HUB_DEGREE:
                continue  # братья через хаб («adult — snuffler» через person) — шум
            for s in _by_freq(sibs, MAX_CHILDREN, drop_rare=True):
                if s != syn:
                    out.append(
                        Link(
                            self._node(s),
                            "брат (общий родитель)",
                            "side",
                            0.7,
                            self.name,
                            via=f"wn:{p.name()}",
                        )
                    )
        for h in _named(syn.part_holonyms() + syn.member_holonyms())[:3]:
            out.append(Link(self._node(h), "входит в", "assoc", 0.6, self.name))
        return out


@lru_cache(maxsize=1)
def _rwn():
    from ruwordnet import RuWordNet

    return RuWordNet()


class RuWordNetSource(Source):
    name = "ruwordnet"
    prefix = "rwn:"

    def _node(self, syn) -> Node:
        return Node(f"rwn:{syn.id}", syn.title.lower(), self.name, "ru")

    def link(self, term: str, lang: str) -> list[Node]:
        if lang != "ru":
            return []
        out, seen = [], set()
        for sense in _rwn().get_senses(term.replace("ё", "е"))[:6]:
            s = sense.synset
            if s.id not in seen and s.part_of_speech == "N":
                seen.add(s.id)
                out.append(self._node(s))
        return out

    def neighbors(self, node: Node) -> list[Link]:
        syn = _rwn().get_synset_by_id(node.id[4:])
        if syn is None:
            return []
        out: list[Link] = []

        def nouns(xs):
            return [x for x in xs if x.part_of_speech == "N"]

        for h in nouns(syn.hyponyms)[:MAX_CHILDREN]:
            out.append(Link(self._node(h), "подвид", "deeper", 0.9, self.name))
        for i in list(syn.instances)[:6]:
            out.append(Link(self._node(i), "экземпляр", "deeper", 0.8, self.name))
        for m in list(syn.meronyms)[:4]:
            out.append(Link(self._node(m), "часть", "deeper", 0.7, self.name))
        for p in nouns(syn.hypernyms)[:2]:
            out.append(Link(self._node(p), "обобщение", "side", 0.8, self.name))
            sibs = nouns(p.hyponyms)
            if len(sibs) > HUB_DEGREE:
                continue
            for s in sibs[:MAX_CHILDREN]:
                if s.id != syn.id:
                    out.append(
                        Link(
                            self._node(s),
                            "брат (общий родитель)",
                            "side",
                            0.7,
                            self.name,
                            via=f"rwn:{p.id}",
                        )
                    )
        for h in list(syn.holonyms)[:3]:
            out.append(Link(self._node(h), "входит в", "assoc", 0.6, self.name))
        for d in list(syn.domains)[:2] + list(syn.domain_items)[:3]:
            out.append(Link(self._node(d), "предметная область", "assoc", 0.6, self.name))
        for c in nouns(list(syn.causes)[:2] + list(syn.effects)[:2]):
            out.append(Link(self._node(c), "причина/следствие", "assoc", 0.6, self.name))
        for r in nouns(syn.related)[:3]:
            out.append(Link(self._node(r), "связано", "assoc", 0.5, self.name))
        return out
