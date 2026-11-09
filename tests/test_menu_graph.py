from ctxgraph.menu import HEADER, render_arm
from ctxgraph.text import approx_tokens

CTX = "Какие рыбы подойдут для аквариума начинающему?"


def test_menu_slots_limits_and_ids(make_engine):
    eng = make_engine()
    m = eng.menu(CTX)
    assert m.arms, m.render()
    anchors = {a.anchor for a in m.arms}
    assert {"рыба", "аквариум"} <= anchors
    fish = next(a for a in m.arms if a.anchor == "рыба")
    limits = eng.config.limits
    for slot, lim in limits.items():
        assert len(fish.items.get(slot, [])) <= lim
    assert len(fish.items.get("far", [])) <= 1
    text = m.render()
    assert text.startswith(HEADER)
    assert "[f:" in text  # id для expand


def test_arm_token_limit(make_engine):
    eng = make_engine(max_tokens_arm=60)
    m = eng.menu(CTX)
    for a in m.arms:
        assert approx_tokens(render_arm(a, eng.config)) <= 60


def test_menu_without_relation_types(make_engine):
    eng = make_engine(show_rel=False)
    text = eng.menu(CTX).render()
    assert "(подвид)" not in text and "(обобщение)" not in text


def test_no_duplicates_across_arms_and_crossing(make_engine):
    eng = make_engine()
    m = eng.menu(CTX)
    ids = [i.node_id for i in m.items()]
    assert len(ids) == len(set(ids))
    # «цихлиды» достижимы и от «рыба», и от «аквариум» — это пересечение рукавов.
    assert any(c.node_id == "f:cichlids" for c in m.crossings)


def test_items_already_in_context_are_skipped(make_engine):
    eng = make_engine()
    m = eng.menu("Рыбы и вода в аквариуме")
    assert "f:water" not in {i.node_id for i in m.items()}


def test_expand_and_recursion(make_engine):
    eng = make_engine()
    eng.menu(CTX)
    arm = eng.expand("f:salmonids")
    labels = {i.label for i in arm.all_items()}
    assert {"сёмга", "форель"} & labels
    assert eng.expand("f:unknown") is None


def test_auto_depth_builds_chain(make_engine):
    eng = make_engine()
    m0 = eng.menu(CTX, auto_depth=0)
    assert all(not a.children for a in m0.arms)
    m2 = eng.menu(CTX, auto_depth=2)
    fish = next(a for a in m2.arms if a.anchor == "рыба")
    assert fish.children, m2.render()
    assert m2.tokens > m0.tokens
    kinds = [e[1] for e in eng.store.events()]
    assert "auto_expand" in kinds


def test_mark_used_reinforces_path_and_creates_shortcut(make_engine):
    eng = make_engine(shortcut_sim=0.5)
    eng.menu(CTX)
    eng.expand("f:salmonids")
    r = eng.mark_used("f:salmon")
    assert r["shortcut"] is True
    assert r["chain"][:3] == ["f:salmon", "f:salmonids", "f:fish"]
    assert r["reinforced"] >= 2
    e = eng.store.edge_between("f:salmonids", "f:salmon")
    assert e.uses == 1 and e.bonus > 0
    # Повторная отметка не усиливает второй раз.
    assert eng.mark_used("f:salmon")["reinforced"] == 0


def test_direct_item_use_makes_no_shortcut(make_engine):
    eng = make_engine()
    eng.menu(CTX)
    r = eng.mark_used("f:salmonids")
    assert r["shortcut"] is False and r["reinforced"] >= 1


def test_shortcut_applies_only_in_similar_context(make_engine):
    eng = make_engine(shortcut_sim=0.5)
    eng.menu(CTX)
    eng.expand("f:salmonids")
    eng.mark_used("f:salmon")

    eng.new_session("s2")
    m = eng.menu("Какие рыбы подойдут для аквариума новичку?")
    sc = [i for i in m.items() if i.source == "shortcut"]
    assert sc and sc[0].node_id == "f:salmon" and "лососевые" in sc[0].rel

    eng.new_session("s3")
    m = eng.menu("Рецепт: рыба в духовке с лимоном и чесноком, запекать сорок минут")
    assert not [i for i in m.items() if i.source == "shortcut"]


def test_reinforce_from_answer(make_engine):
    eng = make_engine()
    eng.menu(CTX)
    used = eng.reinforce_from_answer("Начинающему подойдут цихлиды, но им нужен простор.")
    assert "f:cichlids" in used
    assert eng.store.edge_between("f:fish", "f:cichlids") or eng.store.edge_between(
        "f:aquarium", "f:cichlids"
    )


def test_experience_slot_from_past_session(make_engine):
    eng = make_engine()
    # «аквариум»+«неон» — в двух разных сессиях; третья сессия про другое (иначе PMI = 0).
    eng.new_session("past-1")
    eng.observe("В аквариуме живут неоны, дважды меняли воду.")
    eng.new_session("past-2")
    eng.observe("Для аквариума с неонами купили фильтр.")
    eng.new_session("past-3")
    eng.observe("Отпуск в горах, палатка и спальник.")
    eng.new_session("now")
    m = eng.menu("Хочу аквариум")
    exp = [i for i in m.items() if i.source == "experience"]
    assert exp, m.render()
    # Только то, что встречалось вместе в ≥ 2 сессиях: неон. Вода и фильтр — по одному разу.
    assert {i.node_id for i in exp} == {"exp:неон"}, m.render()
    # Текущая сессия в опыт не попадает.
    eng.observe("Хочу аквариум с барбусами")
    m2 = eng.menu("Хочу аквариум")
    assert not any("барбус" in i.label for i in m2.items())


def test_diversity_metric(make_engine):
    eng = make_engine()
    eng.menu(CTX)
    eng.expand("f:salmonids")
    eng.expand("f:salmonids")
    eng.expand("f:chordates")
    d = eng.diversity()
    assert d["n"] == 3 and abs(d["unique_ratio"] - 2 / 3) < 1e-9
