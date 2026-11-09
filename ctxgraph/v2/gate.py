"""Gated graph boost (Egor's invariant): dense is the main line; the graph only adds / raises candidates
on a STRONG signal, otherwise the ranking equals dense exactly.

Strong signal for a passage p outside the dense top-`keep`:
- **bridge**: p shares a rare concept (in <= `max_df` sentences of the corpus, absent from the question)
  with one of the dense top-`anchor` passages — the classic multi-hop hop "question -> passage 1 ->
  bridge entity -> passage 2";
- **intersection**: p is linked by rare concepts to >= 2 different dense top-`anchor` passages.
The best `slots` such passages (intersections first, then by dense score) are inserted right after the
dense top-`keep`; everything else keeps the dense order. No signal -> output == dense.
"""

from __future__ import annotations

from collections import defaultdict
import math
from dataclasses import dataclass

from ctxgraph.v2.graph import Graph


@dataclass
class GateConfig:
    max_df: int = 3  # "rare": concept in at most this many sentences
    anchor: int = 3  # dense passages that can emit a bridge
    keep: int = 3  # dense passages never moved
    slots: int = 2  # at most this many graph candidates inserted
    need_intersection: bool = False  # True: only intersections count as strong


class PassageIndex:
    """passage <-> concepts over sentence mentions (passage id = episode `p{i}`)."""

    def __init__(self, g: Graph) -> None:
        self.df = dict(g.s.q("SELECT id, n_fragments FROM concepts"))
        self.n = max(1, g.s.q("SELECT COUNT(*) FROM fragments WHERE kind='sentence'")[0][0])
        self.p_con: dict[int, set[str]] = defaultdict(set)
        self.c_pas: dict[str, set[int]] = defaultdict(set)
        for f, c in g.s.q("SELECT fragment_id, concept_id FROM mentions"):
            p = int(f.split(":")[1][1:])
            self.p_con[p].add(c)
            self.c_pas[c].add(p)


@dataclass
class PprGateConfig:
    max_df: int = 3  # rare concept: in <= max_df sentences (node specificity 1/df >= 1/max_df)
    # "abs": df <= max_df; "rel": df <= max_df * N / ref_n (threshold scaled to the corpus size N, in sentences);
    # "idf": no rarity cut — any shared concept links, question concepts seed PPR with weight log(N/df) (smooth)
    rarity: str = "abs"
    ref_n: int = 7599  # sentences in the dev-100 corpus, where max_df was chosen
    anchor: int = 3  # base top passages used as starts
    margin: float = 0.0  # candidate PPR must exceed the displaced base passage's PPR by this fraction
    slots: int = 1  # at most this many base top-5 passages displaced (from the bottom of the five)
    top: int = 5


def df_limit(idx: PassageIndex, cfg: PprGateConfig) -> float:
    if cfg.rarity == "idf":
        return math.inf
    return cfg.max_df * idx.n / cfg.ref_n if cfg.rarity == "rel" else cfg.max_df


def q_weights(idx: PassageIndex, question_concepts: set[str], cfg: PprGateConfig) -> dict[str, float]:
    """Seed weights of the question concepts that act as starts: 1/df (HippoRAG node specificity) under a rarity cut,
    log(N/df) (IDF) without one."""
    lim = df_limit(idx, cfg)
    qs = sorted(c for c in question_concepts if 0 < idx.df.get(c, 0) <= lim)
    if cfg.rarity == "idf":
        return {c: math.log(idx.n / idx.df[c]) for c in qs if idx.df[c] < idx.n}
    return {c: 1 / idx.df[c] for c in qs}


def signal_candidates(base_rank: list[int], idx: PassageIndex, question_concepts: set[str],
                      cfg: PprGateConfig, extra_anchors: tuple[int, ...] = ()) -> dict[int, set[str]]:  # fmt: skip
    """Passages outside base top-`top` that intersect >= 2 distinct starts through RARE concepts only.
    Starts = base top-`anchor` passages and rare question concepts; a passage meets an anchor by sharing a rare
    concept with it (not a question concept), and meets a question concept by containing it."""
    anchors = list(dict.fromkeys([*base_rank[: cfg.anchor], *extra_anchors]))  # stage 2: + sleeve tops
    shown = set(base_rank[: cfg.top])
    lim = df_limit(idx, cfg)
    rare_q = {c for c in question_concepts if 0 < idx.df.get(c, 0) <= lim}
    starts: dict[int, set[str]] = defaultdict(set)
    for a in anchors:
        for c in idx.p_con.get(a, ()):
            if c in question_concepts or idx.df.get(c, 0) > lim:
                continue
            for p in idx.c_pas.get(c, ()):
                if p not in shown:
                    starts[p].add(f"a{a}")
    for c in rare_q:
        for p in idx.c_pas.get(c, ()):
            if p not in shown:
                starts[p].add(c)
    return {p: s for p, s in starts.items() if len(s) >= 2}


def start_ppr(g: Graph, sent_of: dict[int, list[str]], dense_norm: dict[str, float], anchors: list[int],
              qw: dict[str, float]) -> dict[int, float]:  # fmt: skip
    """PPR from the gate's starts: half the mass on the anchor passages' sentences (in proportion to their min-max
    dense score), half on the question concepts (in proportion to `qw`); passage score = max over its sentences."""
    seeds: dict[str, float] = {}
    fs = [f for a in anchors for f in sent_of[a]]
    ws = [max(dense_norm.get(f, 0.0), 1e-6) for f in fs]
    for f, w in zip(fs, ws, strict=True):
        seeds[f] = (0.5 if qw else 1.0) * w / sum(ws)
    tot = sum(qw.values())
    for c, w in qw.items():
        seeds[c] = 0.5 * w / tot
    best: dict[int, float] = {}
    for k, v in g.ppr(seeds).items():
        if k.startswith("f:"):
            p = int(k.split(":")[1][1:])
            best[p] = max(best.get(p, 0.0), v)
    return best


def pick(base_rank: list[int], order: list[int], pr: dict[int, float], cfg: PprGateConfig,
         admit=None) -> list[int]:  # fmt: skip
    """The gate's displacement loop: candidates in PPR order replace base top-`top` passages from the bottom up,
    each only if its PPR beats the displaced one's by `margin` and `admit(candidate, displaced)` (stage 2: the
    reranker check) holds; the first refusal stops the loop."""
    top = list(base_rank[: cfg.top])
    picked: list[int] = []
    for p in order[: cfg.slots]:
        pos = cfg.top - 1 - len(picked)
        if pos < cfg.anchor:
            break
        if pr.get(p, 0.0) > (1 + cfg.margin) * pr.get(top[pos], 0.0) and (admit is None or admit(p, top[pos])):
            picked.append(p)
        else:
            break
    return picked


def apply_picked(base_rank: list[int], picked: list[int], cfg: PprGateConfig) -> list[int]:
    if not picked:
        return list(base_rank)
    keep = list(base_rank[: cfg.top - len(picked)])
    return keep + picked + [p for p in base_rank if p not in keep and p not in picked]


def ppr_gated(base_rank: list[int], idx: PassageIndex, question_concepts: set[str], ppr_fn,
              cfg: PprGateConfig | None = None, extra_anchors: tuple[int, ...] = (), admit=None) -> tuple[list[int], dict]:  # fmt: skip
    """Egor's invariant: base ranking unchanged unless a strong signal; then a candidate replaces the lowest of the
    base top-5 only if it is an intersection of starts via rare concepts AND its PPR (seeded from the same starts)
    beats the displaced passage by `margin` (AND, stage 2, `admit(candidate, displaced)` holds). No candidate -> no
    PPR call, output is the base ranking itself. `ppr_fn(anchors, {question concept: seed weight}) -> {passage:
    score}` runs the PPR. `info["order"]` / `info["pr"]` let the displacement loop be replayed offline (`pick`)."""
    cfg = cfg or PprGateConfig()
    cand = signal_candidates(base_rank, idx, question_concepts, cfg, extra_anchors)
    if not cand:
        return list(base_rank), {"changed": False, "signal": False}
    anchors = list(dict.fromkeys([*base_rank[: cfg.anchor], *extra_anchors]))
    pr = ppr_fn(anchors, q_weights(idx, question_concepts, cfg))
    order = sorted(cand, key=lambda p: (-pr.get(p, 0.0), p))
    picked = pick(base_rank, order, pr, cfg, admit)
    trace = {"order": order[: cfg.slots],
             "pr": {p: pr.get(p, 0.0) for p in [*order[: cfg.slots], *base_rank[cfg.anchor : cfg.top]]}}  # fmt: skip
    if not picked:
        return list(base_rank), {"changed": False, "signal": True, **trace}
    return apply_picked(base_rank, picked, cfg), {"changed": True, "signal": True, "picked": picked,
                                                  "starts": {p: sorted(cand[p]) for p in picked}, **trace}  # fmt: skip


def gated(dense_rank: list[int], dense_score: dict[int, float], idx: PassageIndex, question_concepts: set[str],
          cfg: GateConfig | None = None) -> tuple[list[int], dict]:  # fmt: skip
    cfg = cfg or GateConfig()
    anchors = dense_rank[: cfg.anchor]
    protected = set(dense_rank[: cfg.keep])
    links: dict[int, set[int]] = defaultdict(set)  # candidate -> anchors it bridges to
    via: dict[int, set[str]] = defaultdict(set)
    for a in anchors:
        for c in idx.p_con.get(a, ()):
            if idx.df.get(c, 0) > cfg.max_df or c in question_concepts:
                continue
            for p in idx.c_pas.get(c, ()):
                if p != a and p not in protected:
                    links[p].add(a)
                    via[p].add(c)
    strong = [p for p, a in links.items() if len(a) >= 2 or not cfg.need_intersection]
    strong.sort(key=lambda p: (-len(links[p]), -dense_score.get(p, -1e9), p))
    picked = strong[: cfg.slots]
    if not picked:
        return list(dense_rank), {"changed": False}
    head = dense_rank[: cfg.keep]
    rest = [p for p in dense_rank if p not in head and p not in picked]
    out = head + picked + rest
    return out, {"changed": out[:5] != dense_rank[:5], "picked": picked,
                 "via": {p: sorted(via[p])[:3] for p in picked}, "intersection": [len(links[p]) >= 2 for p in picked]}  # fmt: skip
