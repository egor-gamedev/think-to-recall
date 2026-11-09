"""Рукава-фразы, метрика полноты на уровне факта, слой отбора по предложениям."""

import pytest
from test_e1c_sleeves import HISTORY

from ctxgraph.e1b import coverage as C
from ctxgraph.e1b import memory as M
from ctxgraph.e1b import selection as SEL
from ctxgraph.e1b import v9
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens


@pytest.fixture(scope="module")
def arc():
    turns = [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(HISTORY)]
    return M.Archive(turns, HashingEncoder())


def test_phrase_sleeves_group_dependents(arc):
    q = "Узлы блокчейна у Alchemy на бесплатном тарифе выдержат нагрузку мобильного приложения кошелька?"
    ss = v9.phrase_sleeves(arc, q)
    assert 1 <= len(ss) <= v9.MAX_SLEEVES
    assert any(len(s.query.split()) >= 2 for s in ss)
    assert all(len(s.query.split()) <= 6 + 1 for s in ss)
    assert all(s.query in q for s in ss)  # фраза — кусок исходного текста


def test_phrase_max_len_and_pp(arc):
    q = "Переводы i18next для веб-дашборда на Next.js и пуши через Firebase Cloud Messaging"
    short = v9.phrase_sleeves(arc, q, max_len=2, with_pp=False)
    assert all(len(s.query.split()) <= 3 for s in short)
    assert v9.phrase_sleeves(arc, q) == v9.phrase_sleeves(arc, q)


def test_split_sentences_drops_code_and_bullets():
    t = "Первое предложение тут. Второе тоже здесь!\n```py\nx = 1\n```\n- пункт списка номер один\nДа."
    assert C.split_sentences(t) == ["Первое предложение тут.", "Второе тоже здесь!", "пункт списка номер один"]


FACTS = {"F1": {"text": "Переводы храним в JSON-файлах i18next."}}
TURN = {"S1.0": "Привет, у нас мобильный кошелёк на React Native и бэкенд на NestJS. Переводы интерфейса храним в JSON-файлах i18next, "
        "по файлу на язык. Ещё есть веб-дашборд."}  # fmt: skip
FT = {"F1": ["S1.0"]}


def test_fact_coverage_needs_the_fact_sentence():
    head = "- [пользователь] Привет, у нас мобильный кошелёк на React Native и бэкенд на NestJS. …"
    assert C.anchor_coverage(["F1"], FT, TURN, head)["n"] == 1  # старая метрика: начало реплики
    assert C.fact_coverage(["F1"], FACTS, FT, TURN, head)["n"] == 0
    full = "- [пользователь · S1.0] Переводы интерфейса храним в JSON-файлах i18next, по файлу на язык."
    r = C.fact_coverage(["F1"], FACTS, FT, TURN, ["", full])
    assert r["all"] and r["evidence"]["F1"]["hit"].startswith("Переводы")


def test_select_budget_neighbors_and_order(arc):
    def scorer(q, texts):  # заглушка: балл — пересечение слов с запросом
        qs = set(q.lower().split())
        return [len(qs & set(t.lower().split())) for t in texts]

    q = "где храним переводы i18next"
    text, log = SEL.select(arc, q, approx_tokens, 120, scorer)
    assert approx_tokens(text) <= 120 and text.startswith(M.HEADER)
    lines = [ln for ln in text.splitlines() if ln.startswith("- [")]
    assert lines and "i18next" in lines[0]
    assert all(" · S1." in ln for ln in lines)
    ids = [p for unit in log["picked"] for p in unit]
    assert len(ids) == len(set(ids))
    assert SEL.select(arc, q, approx_tokens, 10, scorer)[0] == ""


def test_v2_plus_pool_has_no_graph(arc):
    pool = SEL.hybrid_pool(arc, "переводы i18next", 3)
    assert len(pool) == 3 and pool == SEL.hybrid_pool(arc, "переводы i18next", 3)

    def scorer(q, texts):
        return [float(len(t)) for t in texts]

    text, _ = SEL.v2_plus(arc, "переводы i18next", approx_tokens, 200, scorer, pool=3)
    ids = {ln.split(" · ")[1].split("]")[0] for ln in text.splitlines() if ln.startswith("- [")}
    assert ids and ids <= set(pool)
