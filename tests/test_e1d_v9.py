"""V9: рукава из текста, поиск по рукавам, пересечения, якорь цели, блок V9-q."""

import pytest
from test_e1c_sleeves import HISTORY, Q

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import v9
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens


@pytest.fixture(scope="module")
def arc():
    turns = [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(HISTORY)]
    return M.Archive(turns, HashingEncoder())


def test_sleeves_rare_first_and_capped(arc):
    ss = v9.text_sleeves(arc, Q)
    assert 1 <= len(ss) <= v9.MAX_SLEEVES
    ents = [s for s in ss if s.kind == "entity"]
    assert [arc.df(s.id) for s in ents] == sorted(arc.df(s.id) for s in ents)
    assert all(s.id in arc.by_ent and not arc.common(s.id) for s in ents)
    assert len({s.id for s in ss}) == len(ss)


def test_phrases_fill_when_no_entities(arc):
    ss = v9.text_sleeves(arc, "где лежат переводы")
    assert ss and all(s.kind in ("entity", "phrase") for s in ss)
    assert all(arc.dfl.get(s.id, 0) > 0 for s in ss if s.kind == "phrase")


def test_clause_mode_extends_query(arc):
    ss = v9.text_sleeves(arc, Q, query_mode="clause")
    assert all(s.query.startswith(s.surface) for s in ss)
    assert any(s.query != s.surface for s in ss)


def test_intersections_first_and_bonus(arc):
    c = v9.candidates(arc, Q)
    inter = c["intersections"]
    assert inter, "веб-дашборд связан и с JWT, и с переводами"
    assert c["ranked"][: len(inter)] == inter
    c0 = v9.candidates(arc, Q, bonus=0.0)
    assert set(c0["ranked"]) == set(c["ranked"])


def test_prev_reached_makes_cross_step_intersection(arc):
    c = v9.candidates(arc, "Пуши через Firebase")
    lone = [t for t in c["ranked"] if t not in c["intersections"]]
    assert lone
    t = lone[0]
    c2 = v9.candidates(arc, "Пуши через Firebase", prev_reached={t: {"другая тема"}})
    assert t in c2["intersections"]


def test_anchor_filter(arc):
    assert v9.candidates(arc, Q, anchor_cos=-1.0)["ranked"] == v9.candidates(arc, Q)["ranked"]
    c = v9.candidates(arc, Q, anchor_cos=1.01)
    assert c["ranked"] == [] and c["dropped_by_anchor"]


def test_exclude_already_shown(arc):
    c = v9.candidates(arc, Q)
    c2 = v9.candidates(arc, Q, exclude={c["ranked"][0]})
    assert c["ranked"][0] not in c2["ranked"]


def test_v9q_block_budget_and_full_count(arc):
    text, log = v9.v9q(arc, Q, approx_tokens, 600)
    assert approx_tokens(text) <= 600
    full = [ln for ln in text.splitlines() if ln.startswith("- [")]
    assert 1 <= len(full) <= 4 and len(log["picked"]) <= 4
    assert v9.v9q(arc, Q, approx_tokens, 600) == (text, log)


def test_empty_archive():
    arc = M.Archive([], HashingEncoder())
    assert v9.v9q(arc, Q, approx_tokens, 600)[0] == ""


def test_replaced_turn_marked():
    hist = [
        ("user", "Бэкенд пишем на Django, база PostgreSQL."),
        ("user", "Перешли с PostgreSQL на SQLite: проект маленький, сервер слабый."),
    ]
    arc = M.Archive(
        [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(hist)],
        HashingEncoder(),
    )
    assert arc.replaces == {"S1.1": ["S1.0"]}
    text, _ = v9.v9q(arc, "Какая у нас база, PostgreSQL?", approx_tokens, 600)
    assert "позже заменено" in text


def test_main_line_keeps_v6_and_reserves_one_slot(arc):
    main, _ = M.v6_candidates(arc, Q)
    text, log = v9.v9q_main(arc, Q, approx_tokens, 600)
    assert log["pool_main"] == main
    assert log["picked"][: 4 - len(log["picked_periphery"])] == main[: 4 - len(log["picked_periphery"])]
    assert len(log["picked_periphery"]) <= 1
    assert all(t not in main for t in log["picked_periphery"])
    assert set(main) <= set(log["pool"]) and approx_tokens(text) <= 600


def test_main_line_falls_back_to_v6_without_periphery(arc):
    text, log = v9.v9q_main(arc, Q, approx_tokens, 600, anchor_cos=1.01)
    assert log["picked_periphery"] == [] and text == M.v6_hybrid(arc, Q, approx_tokens, 600)


def test_bonus_norm_divides_by_entity_count(arc):
    c = v9.candidates(arc, Q, bonus=1.0)
    cn = v9.candidates(arc, Q, bonus=1.0, bonus_norm="ents")
    assert c["intersections"] and set(c["intersections"]) == set(cn["intersections"])
    assert v9.candidates(arc, Q, k_per_sleeve=20)["k_per_sleeve"] == 20


def test_rest_keeps_v6_block_and_budget(arc):
    v6 = M.v6_hybrid(arc, Q, approx_tokens, 600)
    text, log = v9.v9q_rest(arc, Q, approx_tokens, 600)
    assert text.startswith(v6) and approx_tokens(text) <= 600
    main, _ = M.v6_candidates(arc, Q)
    assert all(t not in main for t in log["periphery_shown"])
    if log["periphery_shown"]:
        assert v9.PERIPHERY_HEADER in text
    small, log2 = v9.v9q_rest(arc, Q, approx_tokens, approx_tokens(v6))
    assert small == v6 and log2["periphery_shown"] == []
