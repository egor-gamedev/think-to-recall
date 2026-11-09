import numpy as np

from ctxgraph.ingest import ingest
from ctxgraph.store import DAY, Edge, Node, Store, diversity


def test_reinforce_and_decay():
    s = Store()
    t = [1000.0]
    s.now = lambda: t[0]
    s.half_life = 10 * DAY
    s.put_node(Node("a", "A", "x"))
    s.put_node(Node("b", "B", "x"))
    s.put_edge(Edge("a", "b", "rel", "x", "deeper", 0.5))
    assert s.reinforce("a", "b", 1.0)
    e = s.edge_between("a", "b")
    assert e.uses == 1 and abs(s.weight(e) - 1.5) < 1e-9
    t[0] += 10 * DAY  # один период полураспада
    assert abs(s.weight(s.edge_between("a", "b")) - 1.0) < 1e-6
    t[0] += 1000 * DAY  # бонус почти исчез, но ребро осталось с базовым весом
    assert abs(s.weight(s.edge_between("a", "b")) - 0.5) < 1e-3
    assert not s.reinforce("a", "zzz")


def test_put_edge_keeps_max_base_and_bonus():
    s = Store()
    s.put_edge(Edge("a", "b", "r", "x", "side", 0.3))
    s.reinforce("a", "b", 0.5)
    s.put_edge(Edge("a", "b", "r", "x", "side", 0.7))
    e = s.edge_between("a", "b")
    assert e.base == 0.7 and e.bonus == 0.5


def test_shortcut_merges_in_similar_context():
    s = Store()
    v = np.ones(4, dtype=np.float32) / 2
    s.add_shortcut("ent:рыба", "t1", "b1", ["рыба"], v, 1.0, "s1")
    s.add_shortcut("ent:рыба", "t1", "b1", ["рыба"], v, 1.0, "s2")
    other = np.array([1, 0, 0, 0], dtype=np.float32)
    s.add_shortcut("ent:рыба", "t1", "b1", ["кухня"], other, 1.0, "s3")
    scs = s.shortcuts_for("ent:рыба")
    assert len(scs) == 2
    assert abs(max(sc["weight"] for sc in scs) - 2.0) < 1e-6


def test_cooccurrence_excludes_current_session():
    s = Store()
    for f in ingest("Неоны и гуппи в аквариуме."):
        s.put_fragment(f, "past")
    for f in ingest("Аквариум и цихлиды."):
        s.put_fragment(f, "now")
    past = {r[0] for r in s.cooccurring("аквариум", exclude_session="now")}
    assert "гуппи" in past and "цихлида" not in past
    assert s.doc_freq("аквариум") == 2 and s.doc_freq("аквариум", "now") == 1


def test_diversity():
    assert diversity([])["n"] == 0
    d = diversity(["a", "a", "a", "a"])
    assert d["unique_ratio"] == 0.25 and d["entropy"] == 0.0
    assert diversity(["a", "b", "c", "d"])["entropy"] == 1.0
