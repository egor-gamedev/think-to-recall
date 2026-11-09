"""ctxgraph v2: append-only store, decision changes, PPR, selection, sleeves, privacy.

Small debug tests (seconds), not quality measurements.
"""

import pytest

from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens
from ctxgraph.v2 import sleeves as SL
from ctxgraph.v2.graph import Graph, cid
from ctxgraph.v2.store import Store

HISTORY = [
    ("user", "Делаем мобильный кошелёк на React Native, бэкенд на NestJS, база PostgreSQL."),
    ("assistant", "Хорошо. Для PostgreSQL возьмём TypeORM, миграции храним в репозитории бэкенда."),
    ("user", "Для веб-дашборда бэкенд выдаёт JWT-сессию после проверки подписи SIWE."),
    ("assistant", "JWT-сессию держим в httpOnly cookie, срок неделя, обновление по подписи."),
    ("user", "Переводы интерфейса храним в JSON-файлах i18next, общие для мобилки и веб-дашборда."),
    ("user", "Перешли с PostgreSQL на SQLite: проект маленький, сервер слабый, админить некому."),
]


def build(path=":memory:", turns=HISTORY):
    g = Graph(Store(path), HashingEncoder())
    for i, (r, t) in enumerate(turns):
        g.add_episode(t, r, "s1", eid=f"S1.{i}")
    return g


def dump(store: Store) -> dict:
    t = ("episodes", "fragments", "concepts", "mentions", "edges", "supersedes")
    cols = {"episodes": "id,seq,session_id,role,text", "fragments": "id,episode_id,idx,seq,text,kind",
            "concepts": "id,label,kind,first_seq,n_fragments", "mentions": "fragment_id,concept_id",
            "edges": "src,dst,rel,weight,t_valid,t_invalid,episodes", "supersedes": "new_fragment,old_fragment,seq"}  # fmt: skip
    return {x: sorted(store.q(f"SELECT {cols[x]} FROM {x}")) for x in t}


def test_incremental_equals_reopened_and_never_deletes(tmp_path):
    statements = []
    a = Graph(Store(), HashingEncoder())
    a.s.db.set_trace_callback(statements.append)
    sizes = []
    for i, (r, t) in enumerate(HISTORY):
        a.add_episode(t, r, "s1", eid=f"S1.{i}")
        sizes.append(sum(a.s.counts()[k] for k in ("episodes", "fragments", "concepts", "mentions", "edges")))
    assert sizes == sorted(sizes)  # rows only grow
    assert not [s for s in statements if s.lstrip().upper().startswith(("DELETE", "DROP"))]
    # half, close, reopen, rest == all in one go
    p = tmp_path / "g.sqlite"
    b = build(p, HISTORY[:3])
    b.s.db.close()
    b = Graph(Store(p), HashingEncoder())
    for i, (r, t) in enumerate(HISTORY[3:], start=3):
        b.add_episode(t, r, "s1", eid=f"S1.{i}")
    assert dump(a.s) == dump(b.s)


def test_same_episode_twice_is_noop():
    g = build()
    before = g.s.counts()
    assert g.add_episode(HISTORY[0][1], "user", "s1", eid="S1.0")["new"] is False
    assert g.s.counts() == before


def test_decision_change_invalidates_not_deletes():
    g = build()
    sup = g.s.q("SELECT new_fragment, old_fragment FROM supersedes")
    assert sup and all(n.startswith("f:S1.5:") and o.startswith("f:S1.0:") for n, o in sup)
    inv = g.s.q("SELECT src, dst, t_invalid FROM edges WHERE t_invalid IS NOT NULL")
    assert inv and all(t == 5 for *_, t in inv)
    assert any("postgresql" in (s + d) for s, d, _ in inv)  # merged phrase: «база postgresql»
    # invalid edges are out of PPR but the old sentence is still retrievable and marked
    nodes, ix, P = g._adjacency()
    for s, d, _ in inv:
        assert P[ix[s], ix[d]] == 0 if s in ix and d in ix else True
    mem, _ = g.select("Какая у нас база данных, PostgreSQL?", 600, approx_tokens, candidates=40)
    assert "(later replaced)" in mem and "SQLite" in mem


def test_time_travel_view():
    g = build()
    early = g._adjacency(upto_seq=1)[0]
    assert not any(n.startswith("f:S1.4:") for n in early)
    assert any(n.startswith("f:S1.4:") for n in g._adjacency()[0])


def test_ppr_mass_and_neighbours():
    g = build()
    pr = g.ppr({cid("i18next"): 1.0})
    assert abs(sum(pr.values()) - 1.0) < 1e-6
    assert any(k.startswith("f:S1.4:") for k, _ in sorted(pr.items(), key=lambda x: -x[1])[:5])


def test_select_budget_and_sentences():
    g = build()
    mem, info = g.select("Где храним переводы интерфейса?", 120, approx_tokens)
    assert approx_tokens(mem) <= 120 and "i18next" in mem
    ids = [i for u in info["picked"] for i in u]
    assert len(ids) == len(set(ids))


def test_privacy_before_storage():
    g = Graph(Store(), HashingEncoder())
    key = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"
    r = g.add_episode(f"Ключ от API: {key}, не потеряй. Почта ivan@example.com.", "user", "s1", eid="P1")
    assert {x["type"] for x in r["redactions"]} >= {"api_key", "email"}
    blob = " ".join(str(v) for t in dump(g.s).values() for v in t)
    assert key not in blob and "ivan@example.com" not in blob


def test_sleeves_and_intersections():
    g = build()
    step = "Нужно вспомнить, как устроена JWT-сессия после подписи SIWE и где лежат переводы для веб-дашборда."
    mgs = SL.mini_graphs(g, step)
    assert 1 <= len(mgs) <= 4 and all(m.nodes for m in mgs)
    fake = [SL.MiniGraph("a", "a", nodes={"x": 1.0, "y": 0.5}), SL.MiniGraph("b", "b", nodes={"x": 0.3, "z": 1.0})]
    assert SL.intersections(fake) == {"x": ["a", "b"]}
    tr = SL.sleeves_trace(g, step, "Как сделать вход в веб-дашборд?", approx_tokens, 200, anchor_cos=None)
    assert approx_tokens(tr["periphery"]) <= 200



def test_subphrase_edges_link_not_merge():
    g = build(turns=[("user", "Сегодня делаем экран создания кошелька с мнемоникой."),
                     ("user", "Экран создания ещё раз проверим на маленьком телефоне."),
                     ("user", "Экраны создания кошелька теперь в тёмной теме.")])  # fmt: skip
    ids = {r[0] for r in g.s.q("SELECT id FROM concepts")}
    assert cid("экран создание кошелек") in ids and cid("экран создание") in ids  # not merged
    sub = g.s.q("SELECT src, dst, weight FROM edges WHERE rel='subphrase'")
    assert (cid("экран создание"), cid("экран создание кошелек"), 0.3) in sub
    # one concept for the same phrase after lemmatizing the whole phrase
    n = g.s.q("SELECT n_fragments FROM concepts WHERE id=?", (cid("экран создание кошелек"),))[0][0]
    assert n == 2


def test_periphery_never_repeats_main_block():
    g = build()
    step = "Нужно вспомнить, как устроена JWT-сессия после подписи SIWE и где лежат переводы для веб-дашборда."
    tr = SL.sleeves_trace(g, step, "Как сделать вход в веб-дашборд?", approx_tokens, 400, anchor_cos=None)
    shown = set(tr["final_picked"])
    assert shown and not shown & {it["fragment"] for it in tr["periphery_items"]}


def test_assistant_turns_down_weighted():
    g = build()
    g.cfg.w_assistant = 1.0
    _, a = g.select("JWT-сессия в cookie", 600, approx_tokens)
    g.cfg.w_assistant = 0.01
    _, b = g.select("JWT-сессия в cookie", 600, approx_tokens)
    first = lambda info: info["lines"][0]["role"]  # noqa: E731
    assert first(b) == "user" and [x["role"] for x in b["lines"]].count("user") >= [x["role"] for x in a["lines"]].count("user")


def test_min_score_cuts_instead_of_filling():
    g = build()

    def scorer(q, texts):
        return [1.0 if "i18next" in t else 0.0 for t in texts]

    full, _ = g.select("Где храним переводы?", 600, approx_tokens, scorer=scorer)
    cut, info = g.select("Где храним переводы?", 600, approx_tokens, scorer=scorer, min_score=0.5)
    assert len(cut) < len(full) and all(x["score"] >= 0.5 for x in info["lines"])


def test_hybrid_splits_by_threshold_without_duplicates():
    g = build()

    def scorer(q, texts):
        return [0.9 if "i18next" in t or "JWT" in t else 1e-5 for t in texts]

    step = "Где лежат переводы для веб-дашборда и как устроена JWT-сессия?"
    fill = SL.sleeves_trace(g, step, "Как сделать вход?", approx_tokens, 400, anchor_cos=None, scorer=scorer, mode="fill")
    cut = SL.sleeves_trace(g, step, "Как сделать вход?", approx_tokens, 400, anchor_cos=None, scorer=scorer, mode="cutoff")
    hyb = SL.sleeves_trace(g, step, "Как сделать вход?", approx_tokens, 400, anchor_cos=None, scorer=scorer)
    assert all(x["score"] >= g.cfg.min_score for x in cut["final_lines"])
    assert hyb["final_lines"] == cut["final_lines"]  # same main block as the cutoff
    assert hyb["low_lines"] and SL.LOW_MARK in hyb["periphery"]
    main = {f for x in hyb["final_lines"] for f in x["fragments"]}
    per = [it["fragment"] for it in hyb["periphery_items"]]
    assert not main & set(per) and len(per) == len(set(per))  # no duplicates anywhere
    assert len(fill["final_lines"]) >= len(cut["final_lines"])


def test_informative_filter_and_no_code_in_memory():
    g = Graph(Store(), HashingEncoder())
    assert not g.informative("Структура в монорепе:")
    assert not g.informative("Бэкапы как сделать?")
    assert not g.informative("Прокрутку добавил, зелёное.")
    assert g.informative("По сетям: поддерживаем Ethereum и Polygon, Solana отложили до весны.")
    g.add_episode("Библиотека для сети — viem, легче ethers и лучше типизирована.\n\n```ts\nimport { mainnet } from 'viem/chains';\n```",
                  "assistant", "s1", eid="A1")  # fmt: skip
    mem, info = g.select("какая библиотека viem для сети", 600, approx_tokens)
    assert "import" not in mem and all(f.endswith(":0") for u in info["picked"] for f in u)


def test_planner_parses_steps_and_builds_sleeves():
    from ctxgraph.v2 import planner as PL

    assert PL.parse_steps("1. Вспомнить, какие сети поддерживаем.\n2) Где храним ключи на устройстве.\n3. Сроки релиза и размер команды.\n4. Лишний шаг тут.") == [
        "Вспомнить, какие сети поддерживаем.", "Где храним ключи на устройстве.", "Сроки релиза и размер команды."]

    class LLM:
        calls = []

        def chat(self, messages, max_tokens, tools=None):
            self.calls.append(messages[-1]["content"])
            return {"content": "1. Как устроена JWT-сессия для веб-дашборда.\n2. Где лежат переводы i18next."}

    g, llm = build(), LLM()
    p = PL.plan_and_recall(g, llm, "Как сделать вход в веб-дашборд?", approx_tokens)
    assert len(p.steps) == 2 and len(llm.calls) == 1 and "Do NOT answer" in llm.calls[0]
    assert p.trace["sleeves"] and p.memory_for_model


def test_bulk_synonyms_equal_incremental():
    from ctxgraph.v2.graph import Config

    turns = HISTORY + [("user", "JWT-сессии и JWT-токены храним в httpOnly cookie, сессия неделю.")]
    cfg = Config(synonym_thr=0.5)  # low threshold so the hashing encoder yields some synonym edges
    a = Graph(Store(), HashingEncoder(), config=cfg)
    b = Graph(Store(), HashingEncoder(), config=cfg, defer_synonyms=True)
    for i, (r, t) in enumerate(turns):
        a.add_episode(t, r, "s1", eid=f"S1.{i}")
        b.add_episode(t, r, "s1", eid=f"S1.{i}")
    assert b.s.q("SELECT COUNT(*) FROM edges WHERE rel='synonym'")[0][0] == 0
    b.build_synonyms(block=3)
    q = "SELECT src, dst, rel, round(weight, 5), t_valid, episodes FROM edges WHERE rel='synonym' ORDER BY 1, 2"
    assert a.s.q(q) and a.s.q(q) == b.s.q(q)


def test_gate_equals_dense_without_signal_and_only_inserts():
    from ctxgraph.v2.gate import GateConfig, PassageIndex, gated

    g = Graph(Store(), HashingEncoder())
    docs = ["Alice Brown founded Zorvex Labs in Oslo.", "Zorvex Labs builds quiet electric ferries.",
            "Electric ferries cross the fjord daily.", "Bob Green likes tea.", "Carol White paints murals."]  # fmt: skip
    for i, t in enumerate(docs):
        g.add_episode(t, "document", "corpus", eid=f"p{i}")
    idx = PassageIndex(g)
    dense = [0, 3, 4, 2, 1]
    score = {p: 1.0 - 0.1 * r for r, p in enumerate(dense)}
    # no signal (nothing rare linked to the anchors) -> identical to dense
    same, info = gated(dense, score, idx, set(), GateConfig(max_df=0))
    assert same == dense and not info["changed"]
    # bridge «zorvex labs» from p0 raises p1; dense top-keep untouched, nothing removed
    out, info = gated(dense, score, idx, set(), GateConfig(max_df=3, anchor=1, keep=1, slots=1))
    assert out[0] == dense[0] and out[1] == 1 and sorted(out) == sorted(dense) and info["changed"]


def test_ppr_gate_no_signal_is_base_and_only_raises():
    from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated

    g = Graph(Store(), HashingEncoder())
    docs = ["Alice Brown founded Zorvex Labs in Oslo.", "Bob Green likes tea.", "Carol White paints murals.",
            "Dan Black sells bikes.", "Eve Gray reads novels.", "Zorvex Labs was sold to Kestrel Holdings.",
            "Frank Blue plays chess."]  # fmt: skip
    for i, t in enumerate(docs):
        g.add_episode(t, "document", "corpus", eid=f"p{i}")
    idx = PassageIndex(g)
    base = [0, 1, 2, 3, 4, 5, 6]
    calls = []

    def ppr(anchors, rare_q):
        calls.append(1)
        return {5: 1.0, 4: 0.1}

    kestrel = {c for c in idx.p_con[5] if "kestrel" in c}
    # no rare question concept -> p5 meets only one start (anchor p0) -> no signal, no PPR call, output is base
    out, info = ppr_gated(base, idx, set(), ppr, PprGateConfig(max_df=3, anchor=1))
    assert out == base and not info["signal"] and not calls
    # question names Kestrel: p5 meets anchor p0 (via «zorvex labs») and the rare question concept -> raised to 5th
    out, info = ppr_gated(base, idx, kestrel, ppr, PprGateConfig(max_df=3, anchor=1))
    assert info["changed"] and out[:5] == [0, 1, 2, 3, 5] and sorted(out) == sorted(base)
    # PPR margin not met -> base
    out, _ = ppr_gated(base, idx, kestrel, lambda a, q: {5: 0.1, 4: 0.1}, PprGateConfig(max_df=3, anchor=1))
    assert out == base
    # stage 2 reranker check: refused -> base; accepted -> same as without it; offline replay reproduces the output
    from ctxgraph.v2.gate import apply_picked, pick

    cfg = PprGateConfig(max_df=3, anchor=1)
    rr = {5: 0.30, 4: 0.50}
    for delta, moved in ((0.1, False), (0.2, True)):
        out, info = ppr_gated(base, idx, kestrel, ppr, cfg, admit=lambda c, d: rr[c] >= rr[d] - delta)
        assert info["changed"] is moved and (out == base) is not moved
        assert apply_picked(base, pick(base, info["order"], info["pr"], cfg, lambda c, d: rr[c] >= rr[d] - delta), cfg) == out


def test_ppr_gate_idf_and_rel_modes_keep_invariant():
    from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated, q_weights

    g = Graph(Store(), HashingEncoder())
    docs = ["Alice Brown founded Zorvex Labs in Oslo.", "Bob Green likes tea.", "Carol White paints murals.",
            "Dan Black sells bikes.", "Eve Gray reads novels.", "Zorvex Labs was sold to Kestrel Holdings."]  # fmt: skip
    for i, t in enumerate(docs):
        g.add_episode(t, "document", "corpus", eid=f"p{i}")
    idx = PassageIndex(g)
    base = [0, 1, 2, 3, 4, 5]
    kestrel = {c for c in idx.p_con[5] if "kestrel" in c}
    for mode in ("rel", "idf"):
        cfg = PprGateConfig(rarity=mode, max_df=3, ref_n=idx.n, anchor=1)
        out, info = ppr_gated(base, idx, set(), lambda a, q: {5: 1.0}, cfg)  # one start only -> no signal
        assert out == base and not info["signal"]
        out, info = ppr_gated(base, idx, kestrel, lambda a, q: {5: 1.0, 4: 0.1}, cfg)
        assert info["changed"] and out[:5] == [0, 1, 2, 3, 5]
    w = q_weights(idx, kestrel, PprGateConfig(rarity="idf"))
    assert all(v > 0 for v in w.values()) and w


def test_ppr_gate_only_adds_or_raises_property():
    """Random gate outputs: a permutation-preserving insertion — base order kept, at most `slots` newcomers in the
    top, the first `top - slots` base items never move, displaced base items only shift down by the insertions."""
    import random

    from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated

    g = Graph(Store(), HashingEncoder())
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"]
    rng = random.Random(0)
    for i in range(30):
        g.add_episode(f"Item {i} mentions {rng.choice(words)} station and {rng.choice(words)} harbor today.",
                      "document", "corpus", eid=f"p{i}")  # fmt: skip
    idx = PassageIndex(g)
    qc = set(list(idx.df)[:6])
    for trial in range(50):
        base = list(range(30))
        rng.shuffle(base)
        cfg = PprGateConfig(rarity="idf", anchor=3, slots=2, margin=rng.choice([0.0, 3.0]))
        pr = {p: rng.random() for p in range(30)}
        out, info = ppr_gated(base, idx, qc, lambda a, q, pr=pr: pr, cfg)
        if not info["changed"]:
            assert out == base
            continue
        k = len(info["picked"])
        assert 1 <= k <= cfg.slots and sorted(out) == sorted(base)
        assert out[: cfg.top - k] == base[: cfg.top - k] and out[cfg.top - k : cfg.top] == info["picked"]
        assert [p for p in out if p not in info["picked"]] == [p for p in base if p not in info["picked"]]


def test_english_possessor_not_glued_to_concept():
    from ctxgraph.v2 import english

    if not english.available():
        pytest.skip("spaCy model missing")
    keys = [k for k, _, _ in english.concepts_en("Maria Lopez's pottery workshop fits her weekend plans.")]
    assert "maria lopez" in keys and "pottery workshop" in keys
    assert not any("'s" in k or k.startswith("maria lopez ") for k in keys)
    keys = [k for k, _, _ in english.concepts_en("The company's new CEO visited Oslo.")]
    assert "company" in keys and "new ceo" in keys
