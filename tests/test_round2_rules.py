"""Правила, введённые после разметки рёбер раунда 1 (reports/labels/round1_labeled.csv)."""

import pytest

from ctxgraph import text as T
from ctxgraph.embed import HashingEncoder
from ctxgraph.graph import Engine
from ctxgraph.ingest import ingest
from ctxgraph.menu import MenuConfig
from ctxgraph.sources.base import Link, Source
from ctxgraph.sources.experience import ExperienceSource, is_concept
from ctxgraph.sources.wikidata import WikidataSource
from ctxgraph.store import Node, Store


@pytest.mark.parametrize(
    ("phrase", "canon"),
    [
        ("головной боли", "головная боль"),
        ("возвраты средств", "возврат средств"),
        ("таблицы платежей", "таблица платежей"),
        ("Северной войны", "северная война"),
        ("products", "product"),
        ("Kubernetes probes", "kubernetes probe"),
        ("parseConfig", "parse config"),
    ],
)
def test_canonical_phrases(phrase, canon):
    assert T.canonical(phrase) == canon


@pytest.mark.parametrize(
    ("phrase", "ok"),
    [
        ("литров начинающим", False),  # прилагательное не согласовано с существительным
        ("головной боли", True),
        ("температура воды", True),
        ("рыба в аквариуме", False),  # предлог
        ("pod is alive", False),
        ("Shopline Shopline", False),
        ("Compliance We process", False),
        ("CREATE TABLE products", False),
        ("r.product", False),
        ("idempotency key", True),
    ],
)
def test_noun_phrase_filter(phrase, ok):
    assert T.is_noun_phrase(phrase) is ok


def test_no_keyphrases_in_code_and_no_cross_line_phrases():
    sql = [
        {
            "id": "db/queries/products.sql",
            "kind": "file",
            "title": "db/queries/products.sql",
            "content": "SELECT p.* FROM products p JOIN regions r ON r.product_id = p.id",
        }
    ]
    ents = ingest(sql)[0].entities
    assert not any(e.kind == "keyphrase" for e in ents)
    assert any(e.kind == "identifier" and e.canonical == "product id" for e in ents)
    doc = [
        {
            "id": "docs/regions.md",
            "kind": "doc",
            "title": "docs/regions.md",
            "content": "## Regional availability\nProducts are not licensed everywhere.",
        }
    ]
    kps = {e.surface for e in ingest(doc)[0].entities if e.kind == "keyphrase"}
    assert "Regional availability Products" not in kps


def test_urls_not_stored_in_experience():
    s = Store()
    for f in ingest("Дашборд https://dashboard.smsbridge.example для аккаунта"):
        s.put_fragment(f, "past")
    kinds = {r[0] for r in s.db.execute("SELECT kind FROM mentions")}
    assert "url" not in kinds


@pytest.mark.parametrize(
    ("canon", "kind", "surface", "ok"),
    [
        ("currently", "lemma", "currently", False),
        ("plus", "lemma", "plus", False),
        ("дважды", "lemma", "дважды", False),
        ("year", "lemma", "year", False),
        ("https://x.example", "url", None, False),
        ("compliance we process", "keyphrase", "Compliance We process", False),
        ("idempotency", "lemma", "idempotency", True),
        ("неон", "lemma", "неон", True),
        ("payments", "identifier", "payments", True),
    ],
)
def test_experience_concepts(canon, kind, surface, ok):
    assert is_concept(canon, kind, surface) is ok


def test_experience_counts_sessions_not_fragments():
    s = Store()
    src = ExperienceSource(s)
    node = Node("exp:аквариум", "аквариум", "experience")
    for f in ingest("Палатка и спальник."):
        s.put_fragment(f, "other")  # чтобы PMI мог быть > 0
    for f in ingest("Неоны в аквариуме.\n\nАквариум и неоны снова."):
        s.put_fragment(f, "past-1")  # два фрагмента одной сессии — одно наблюдение
    assert src.neighbors(node) == []
    for f in ingest("Аквариум, где живут неоны."):
        s.put_fragment(f, "past-2")
    nb = src.neighbors(node)
    assert [lk.node.id for lk in nb] == ["exp:неон"]
    assert "2 сессиях" in nb[0].rel


def test_experience_requires_positive_pmi():
    s = Store()
    src = ExperienceSource(s)
    for i in range(2):  # пара есть во всех сессиях: PMI = 0 — не связь, а фон
        for f in ingest("Аквариум и неоны."):
            s.put_fragment(f, f"past-{i}")
    assert src.neighbors(Node("exp:аквариум", "аквариум", "experience")) == []


@pytest.mark.parametrize(
    ("word", "ok"),
    [
        ("run", False),
        ("add", False),
        ("deployed", False),
        ("probe", False),
        ("memory", True),
        ("price", True),
        ("redis", True),
        ("idempotency", True),
    ],
)
def test_english_part_of_speech(word, ok):
    assert T.en_noun_ok(word) is ok


@pytest.mark.parametrize("canon", ["created at", "default", "auto", "key"])
def test_column_stoplist(canon):
    assert not is_concept(canon, "identifier", canon)


def test_label_ok():
    ok = Engine._label_ok
    assert ok("рыба", "рыбы") and ok("api key", "API key access authorization")
    assert not ok("regional", "Northeast Regional")
    assert not ok("product id", "Product identification using multiple services")
    assert ok("tablet", "планшетный компьютер")  # алиас на другом языке


class _Dict(Source):
    """Внешний источник с описаниями — для проверки порогов привязки."""

    def __init__(self, name, prefix, cands):
        self.name, self.prefix, self._c = name, prefix, cands

    def link(self, term, lang):
        return [
            Node(f"{self.prefix}{i}", lab, self.name, lang, dict(meta))
            for i, (lab, meta) in enumerate(self._c.get(term, []))
        ]

    def neighbors(self, node):
        return [Link(Node(node.id + "/x", "сосед", self.name), "подвид", "deeper", 0.8, self.name)]


def _engine(sources, **cfg):
    return Engine(Store(), sources, HashingEncoder(), config=MenuConfig(**cfg), session="s")


def test_banned_wikidata_type_needs_proper_noun():
    wd = _Dict("wikidata", "wd:", {"api": [("API", {"desc": "api", "banned": True})]})
    eng = _engine([wd], link_min_sim=-1.0)
    f = ingest("the api returns json")[0]
    assert eng._link_term("api", eng.encoder.one("api"), f) == []
    f2 = ingest("We integrate with Api for fuel cards")[0]
    assert eng._link_term("api", eng.encoder.one("api"), f2)  # имя собственное в тексте


def test_external_sources_need_high_similarity_in_code():
    wn = _Dict("wordnet", "wn:", {"text": [("text", {"gloss": "the words of something written"})]})
    eng = _engine([wn], link_min_sim=-1.0, tech_dict_min_sim=0.99)
    code = ingest(
        [
            {
                "id": "sms.ts",
                "kind": "file",
                "title": "src/sms.ts",
                "content": "const text: string = message.text",
            }
        ]
    )[0]
    assert eng._is_technical(code)
    assert eng._link_term("text", eng.encoder.one("text"), code) == []
    prose = ingest("Пришлите text письма целиком")[0]
    assert not eng._is_technical(prose)
    assert eng._link_term("text", eng.encoder.one("text"), prose)


def test_wordnet_loses_ties_to_ruwordnet_and_wikidata():
    wn = _Dict("wordnet", "wn:", {"x": [("x", {"gloss": "x"})]})
    wd = _Dict("wikidata", "wd:", {"x": [("x", {"desc": "x"})]})
    eng = _engine([wn, wd], link_min_sim=-1.0)
    got = {n.source for n, _ in eng._link_term("x", eng.encoder.one("x"), ingest("x y z")[0])}
    assert got == {"wikidata"}


def test_wikidata_banned_query_and_cache(monkeypatch):
    s = Store()
    src = WikidataSource(s)
    calls = []

    def fake(q):
        calls.append(q)
        return [{"item": "http://www.wikidata.org/entity/Q646807"}]

    monkeypatch.setattr(src, "_sparql", fake)
    assert src.banned(["Q646807", "Q1092514"]) == {"Q646807"}
    assert src.banned(["Q646807", "Q1092514"]) == {"Q646807"}
    assert len(calls) == 1


def test_wordnet_hub_and_rare_senses():
    from ctxgraph.sources.wordnet import WordNetSource

    src = WordNetSource()
    try:
        adult = next(n for n in src.link("adult", "en") if n.id == "wn:adult.n.01")
    except LookupError as e:
        pytest.skip(f"нет данных WordNet: {e}")
    sibs = [lk for lk in src.neighbors(adult) if lk.rel.startswith("брат")]
    assert sibs == []  # родитель person — хаб
    person = Node("wn:person.n.01", "person", "wordnet")
    kids = [lk.node.label for lk in src.neighbors(person) if lk.rel == "подвид"]
    assert 0 < len(kids) <= 8 and "snuffer" not in kids


def test_wikidata_cooldown_instead_of_permanent_off(monkeypatch):
    import ctxgraph.sources.wikidata as wdm

    src = WikidataSource(Store())
    src._failures = 5
    src._cooldown_until = 100.0
    monkeypatch.setattr(wdm.time, "time", lambda: 50.0)
    assert not src.available
    monkeypatch.setattr(wdm.time, "time", lambda: 150.0)
    assert src.available and src._failures == 0


def test_experience_switch_off(tmp_path):
    from ctxgraph import E1_CONFIG, build_engine

    eng = build_engine(tmp_path / "g.sqlite", encoder="hashing", offline=True, experience=False)
    assert not any(s.name == "experience" for s in eng.sources)
    assert eng.config.limits["experience"] == 0
    assert E1_CONFIG.limits["experience"] == 0
    eng.new_session("past")
    eng.observe("Аквариум и неоны.")
    eng.new_session("now")
    assert not [i for i in eng.menu("Хочу аквариум").items() if i.slot == "experience"]
