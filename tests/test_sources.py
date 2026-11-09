import importlib.util

import pytest

from ctxgraph.sources.conceptnet import ConceptNetSource
from ctxgraph.sources.wikidata import WikidataSource
from ctxgraph.store import Node, Store


def _sparql_rows(slot_rows):
    return {"results": {"bindings": slot_rows}}


def _b(**kw):
    return {k: {"value": v} for k, v in kw.items()}


def test_wikidata_parses_slots_and_caches(monkeypatch):
    store = Store()
    src = WikidataSource(store)
    calls = []

    def fake_get(url, params, post=False):
        calls.append(params.get("query", params.get("search")))
        if "search" in params:
            return {
                "search": [
                    {"id": "Q4167410", "label": "рыбы", "description": "страница значений"},
                    {"id": "Q152", "label": "рыбы", "description": "группа водных позвоночных"},
                ]
            }
        q = params["query"]
        if "?x ?p wd:Q152" in q and "sitelinks" in q:
            return _sparql_rows(
                [
                    _b(
                        x="http://www.wikidata.org/entity/Q1",
                        xLabel="лососёвые",
                        p="http://www.wikidata.org/prop/direct/P279",
                        sl="50",
                    ),
                    _b(
                        x="http://www.wikidata.org/entity/Q2",
                        xLabel="Q2",
                        p="http://www.wikidata.org/prop/direct/P279",
                        sl="0",
                    ),
                    _b(
                        x="http://www.wikidata.org/entity/Q3",
                        xLabel="Какой-то музей",
                        p="http://www.wikidata.org/prop/direct/P31",
                        sl="0",
                    ),
                ]
            )
        if "?via" in q:
            return _sparql_rows(
                [
                    _b(
                        x="http://www.wikidata.org/entity/Q4",
                        xLabel="хрящевые рыбы",
                        via="http://www.wikidata.org/entity/Q5",
                        sl="30",
                    )
                ]
            )
        if "wd:Q152 ?p ?x" in q and "P366" in q:
            return _sparql_rows(
                [
                    _b(
                        x="http://www.wikidata.org/entity/Q6",
                        xLabel="пища",
                        p="http://www.wikidata.org/prop/direct/P366",
                    )
                ]
            )
        return _sparql_rows([])

    monkeypatch.setattr(src, "_get", fake_get)
    cands = src.link("рыбы", "ru")
    assert [c.id for c in cands] == ["wd:Q152"]  # страница значений отброшена
    nb = src.neighbors(cands[0])
    got = {(lk.slot, lk.rel, lk.node.label) for lk in nb}
    assert ("deeper", "подкласс", "лососёвые") in got
    assert ("side", "брат (общий родитель)", "хрящевые рыбы") in got
    assert ("assoc", "применяется для", "пища") in got
    assert not any(lk.node.label in ("Q2", "Какой-то музей") for lk in nb)
    sib = next(lk for lk in nb if lk.node.label == "хрящевые рыбы")
    assert sib.via == "wd:Q5"
    n_calls = len(calls)
    src.neighbors(cands[0])
    assert len(calls) == n_calls  # из кэша


def test_wikidata_failure_not_cached(monkeypatch):
    store = Store()
    src = WikidataSource(store)
    monkeypatch.setattr(src, "_get", lambda *a, **k: None)
    assert src.neighbors(Node("wd:Q1", "x", "wikidata")) == []
    assert store.cache_get("wd:nb:Q1") is None


def test_conceptnet_degrades_gracefully(monkeypatch):
    import requests

    store = Store()
    src = ConceptNetSource(store)

    def boom(*a, **k):
        raise requests.ConnectionError("502")

    monkeypatch.setattr(requests, "get", boom)
    node = src.link("fish", "en")[0]
    assert src.neighbors(node) == [] and src.neighbors(node) == []
    assert not src.available



@pytest.mark.skipif(importlib.util.find_spec("ruwordnet") is None, reason="ruwordnet не установлен")
def test_ruwordnet_local():
    from ctxgraph.sources.wordnet import RuWordNetSource

    src = RuWordNetSource()
    try:
        cands = src.link("рыба", "ru")
    except Exception as e:  # база не скачана
        pytest.skip(f"нет базы RuWordNet: {e}")
    fish = next(c for c in cands if c.label == "рыба")
    nb = src.neighbors(fish)
    assert any(lk.slot == "deeper" for lk in nb) and any(lk.slot == "side" for lk in nb)


def test_wordnet_local():
    from ctxgraph.sources.wordnet import WordNetSource

    src = WordNetSource()
    try:
        cands = src.link("fish", "en")
    except LookupError as e:
        pytest.skip(f"нет данных WordNet: {e}")
    nb = src.neighbors(cands[0])
    labels = {lk.node.label for lk in nb if lk.slot == "deeper"}
    assert labels and src.link("рыба", "ru") == []
