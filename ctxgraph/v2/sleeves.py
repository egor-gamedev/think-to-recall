"""Thinking sleeves on the v2 graph (ARCHITECTURE_V2 §3.9) — our own part.

sleeve = a group of words of a thinking step (noun group around a rare concept, 2–6 words;
`ctxgraph.e1b.v9.phrase_sleeves`, reused through a small adapter) -> its own seeds (dense + BM25 of
the phrase + concepts of the phrase) -> its **mini-graph** = PPR from those seeds (concepts and
sentences with weights). Intersections = nodes present (in the top of) >= 2 mini-graphs.
Periphery: <= `periphery_share` of the step context, intersections first, weight 0.5**depth,
anchor filter = closeness to the original question, block marked as optional.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ctxgraph.e1b import v9
from ctxgraph.v2.extract import phrase_key
from ctxgraph.v2.graph import Graph, cid

PERIPHERY_HEADER = "Nearby (optional associations, not requirements; use only if relevant):"
LOW_MARK = "[less certain, optional]"


class _Keyed(dict):
    """Concept-id dict looked up by any surface canonical: keys are normalized with `phrase_key`."""

    def __contains__(self, k) -> bool:
        return dict.__contains__(self, phrase_key(k))

    def get(self, k, default=None):
        return dict.get(self, phrase_key(k), default)


class _ArcAdapter:
    """Duck-typed stand-in for `e1b.memory.Archive`, enough for `v9.phrase_sleeves`. Lookups go
    through `phrase_key`, the v2 concept identity (whole phrase lemmatized)."""

    def __init__(self, g: Graph) -> None:
        rows = g.s.q("SELECT id, n_fragments FROM concepts")
        self.df_ = _Keyed({r[0][2:]: r[1] for r in rows})
        self.n = max(1, g.s.q("SELECT COUNT(*) FROM fragments")[0][0])
        self.by_ent = _Keyed({k: [None] * v for k, v in self.df_.items() if v})
        self.dfl = self.df_

    def df(self, e: str) -> int:
        return self.df_.get(e, 0)

    def common(self, e: str) -> bool:
        return self.df(e) > max(3, 0.15 * self.n)


@dataclass
class MiniGraph:
    sleeve: str  # phrase text
    key: str
    seeds: dict[str, float] = field(default_factory=dict)
    nodes: dict[str, float] = field(default_factory=dict)  # top nodes with normalized weight
    stats: dict = field(default_factory=dict)  # spread of weights without the seed nodes


def spread(pr: dict[str, float], seeds: dict[str, float], top: int = 15) -> dict:
    """Does the mini-graph discriminate? Weights of the top non-seed nodes: share of PPR mass,
    max/min ratio and coefficient of variation (≈0 means a flat spread, e.g. from a hub)."""
    rest = sorted((v for k, v in pr.items() if k not in seeds), reverse=True)[:top]
    if not rest:
        return {"n": 0}
    a = np.array(rest)
    return {"n": len(rest), "mass": round(float(a.sum()), 4), "top": round(float(a[0]), 4),
            "max_min": round(float(a[0] / a[-1]), 2) if a[-1] > 0 else None,
            "cv": round(float(a.std() / a.mean()), 3) if a.mean() > 0 else None}  # fmt: skip


def mini_graphs(g: Graph, text: str, *, top: int = 15, upto_seq: int | None = None) -> list[MiniGraph]:
    arc = _ArcAdapter(g)
    out = []
    for s in v9.phrase_sleeves(arc, text):
        tr = g.seeds(s.query, upto_seq)
        seeds = dict(tr.concept_seeds)
        for lem in s.lemmas:
            if lem in arc.by_ent:
                seeds.setdefault(cid(phrase_key(lem)), g.specificity(arc.df(lem), arc.n))
        csum = sum(seeds.values()) or 1.0
        fw = g.cfg.passage_weight if seeds else 1.0
        tot = sum(tr.fragment_seeds.values()) or 1.0
        for f, v in tr.fragment_seeds.items():
            seeds[f] = seeds.get(f, 0.0) + fw * csum * v / tot
        pr = g.ppr(seeds, upto_seq)
        # hide only the seed *concepts* (they dominate the PPR mass); seed sentences are the most
        # relevant content of the sleeve and must stay in its mini-graph
        cseeds = {k for k in seeds if k.startswith("c:")}
        best = sorted(((k, v) for k, v in pr.items() if k not in cseeds), key=lambda x: (-x[1], x[0]))[:top]
        mx = best[0][1] if best else 1.0
        out.append(MiniGraph(s.query, s.id, seeds, {k: v / mx for k, v in best}, spread(pr, cseeds, top)))
    return out


def intersections(mgs: list[MiniGraph], g: Graph | None = None, *, min_weight: float | None = None,
                  ) -> dict[str, list[str]]:  # fmt: skip
    """Nodes reached from >= 2 sleeves. Without `g` — the plain rule (weight >= 0.05 in each).
    With `g` — the strict rule (Egor): weight >= 0.2 in each mini-graph; hubs excluded (a sentence with
    more concepts than the 90th percentile, a concept more frequent than the 95th percentile); a concept
    found in a single sentence counts as that sentence — two sleeves that merely found the same
    sentence must not turn its every word into an "intersection"."""
    thr = (0.2 if g is not None else 0.05) if min_weight is None else min_weight
    seen: dict[str, set[str]] = defaultdict(set)
    for m in mgs:
        for n, w in m.nodes.items():
            if w >= thr:
                seen[n].add(m.sleeve)
    if g is None:
        return {n: sorted(ss) for n, ss in seen.items() if len(ss) >= 2}
    df = dict(g.s.q("SELECT id, n_fragments FROM concepts"))
    per_frag = dict(g.s.q("SELECT fragment_id, COUNT(*) FROM mentions GROUP BY fragment_id"))
    frag_hub = float(np.percentile(list(per_frag.values()), 90)) if per_frag else 1e9
    con_hub = float(np.percentile(list(df.values()), 95)) if df else 1e9
    out: dict[str, set[str]] = defaultdict(set)
    for n, ss in seen.items():
        if len(ss) < 2:
            continue
        if n.startswith("c:"):
            if df.get(n, 0) > con_hub:
                continue
            if df.get(n, 0) == 1:
                f = g.s.q("SELECT fragment_id FROM mentions WHERE concept_id=?", (n,))
                n = f[0][0] if f else n
        if n.startswith("f:") and per_frag.get(n, 0) > frag_hub:
            continue
        out[n] |= ss
    return {n: sorted(ss) for n, ss in out.items()}


def periphery(g: Graph, mgs: list[MiniGraph], question: str, count: Callable[[str], int], budget: int, *,
              depth: int = 0, anchor_cos: float | None = 0.76, exclude: set[str] | frozenset = frozenset(),
              ) -> tuple[str, list[dict]]:  # fmt: skip
    """Periphery block for the next step. Order: intersections, then by weight × 0.5**depth."""
    inter = intersections(mgs, g)
    score: dict[str, float] = defaultdict(float)
    for m in mgs:
        for n, w in m.nodes.items():
            if n.startswith("f:") and n not in exclude:
                score[n] = max(score[n], w * 0.5**depth)
    if anchor_cos is not None and score:
        qv = g.enc.one(question) if hasattr(g.enc, "one") else g.enc.encode([question])[0]
        ix = {f: i for i, f in enumerate(g._fids)}
        score = {f: w for f, w in score.items() if f in ix and float(g._fmat[ix[f]] @ qv) >= anchor_cos}
    order = sorted(score, key=lambda f: (f not in inter, -score[f], f))
    lines, items = [PERIPHERY_HEADER], []
    for f in order:
        r = g.s.q("SELECT text FROM fragments WHERE id=? AND kind='sentence'", (f,))
        if not r or not g.informative(r[0][0]):
            continue
        mark = " [links: " + " + ".join(f"«{x}»" for x in sorted(set(inter[f]))) + "]" if f in inter else ""
        ln = f"- {r[0][0]}{mark}"
        if count("\n".join(lines + [ln])) > budget:
            continue
        lines.append(ln)
        items.append({"fragment": f, "weight": round(score[f], 3), "intersection": f in inter})
    return ("\n".join(lines) if items else ""), items


def final_block(g: Graph, question: str, mgs: list[MiniGraph], count, budget: int = 600, *,
                scorer=None, candidates: int = 30, min_score: float | None = None,
                score_query: str | None = None, split_low: bool = False) -> tuple[str, dict]:  # fmt: skip
    """Memory for the answer (§3.9 step 6): selection over the query ranking fused with the
    sentences of every sleeve mini-graph (intersections first within each sleeve)."""
    inter = intersections(mgs, g)
    extra = []
    for m in mgs:
        fr = [n for n in m.nodes if n.startswith("f:")]
        extra.append(sorted(fr, key=lambda f: (f not in inter, -m.nodes[f], f)))
    return g.select(question, budget, count, scorer=scorer, candidates=candidates, extra=extra, min_score=min_score,
                    score_query=score_query, split_low=split_low)


def sleeves_trace(g: Graph, text: str, question: str, count, budget: int, **kw) -> dict:
    """Everything the demo shows for one step: sleeves, mini-graphs, intersections, periphery,
    and the final memory block (query + sleeves)."""
    mgs = mini_graphs(g, text, upto_seq=kw.pop("upto_seq", None))
    scorer = kw.pop("scorer", None)
    # mode: "hybrid" (default) — main block above the threshold, the rest within budget goes to the
    # periphery marked "less certain, optional"; "cutoff" — stop at the threshold; "fill" — no threshold
    mode = kw.pop("mode", "hybrid")
    min_score = kw.pop("min_score", g.cfg.min_score) if scorer is not None and mode != "fill" else None
    final, finfo = final_block(g, question, mgs, count, scorer=scorer, min_score=min_score,
                               score_query=f"{question} {text}", split_low=mode == "hybrid")
    shown = {f for u in finfo.get("picked", []) for f in u}  # what the main block shows is not repeated
    low = finfo.get("low_lines", [])
    low_ids = {f for x in low for f in x["fragments"]}
    block, items = periphery(g, mgs, question, count, budget, exclude=shown | low_ids | set(kw.pop("exclude", ())), **kw)
    if low:  # below-threshold lines join the periphery, marked, without duplicates
        head = block.splitlines() if block else [PERIPHERY_HEADER]
        block = "\n".join(head + [x["line"].replace("- ", f"- {LOW_MARK} ", 1) for x in low])
        items = items + [{"fragment": f, "weight": x["score"], "intersection": False, "low": True}
                         for x in low for f in x["fragments"]]  # fmt: skip
    return {
        "sleeves": [{"phrase": m.sleeve, "key": m.key, "seeds": dict(sorted(m.seeds.items(), key=lambda x: -x[1])[:8]),
                     "nodes": m.nodes, "stats": m.stats} for m in mgs],  # fmt: skip
        "intersections": intersections(mgs, g),
        "intersections_plain": intersections(mgs),
        "final_lines": finfo.get("lines", []),
        "final_picked": sorted(shown),
        "low_lines": low,
        "periphery": block,
        "periphery_items": items,
        "final": final,
    }


__all__ = ["MiniGraph", "PERIPHERY_HEADER", "intersections", "mini_graphs", "periphery", "sleeves_trace", "np"]
