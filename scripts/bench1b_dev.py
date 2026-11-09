"""Bench 1b exploration on MuSiQue dev-100 (disjoint from the test 1000) — no LLM calls, CPU.

    python scripts/bench1b_dev.py   # -> reports/bench1b_dev.md, out/bench1b_dev/rows.json

Variants (passage Recall@2/@5, passage score = max over its sentences):
  a_dense     e5 dense (the bench1 control)
  c_ppr       bench1 graph variant (unchanged)
  r_dense     dense top-40 sentences -> cross-encoder (no graph): separates the reranker from the graph
  h_ppr       PPR seeded like HippoRAG 2 (`graph_search_with_fact_entities`, checked against their code):
              "facts" = top-5 sentences by dense (link_top_k = 5), recognition-memory filter = cross-encoder on
              (question, sentence) keeping scores >= Config.min_score; concept seeds = concepts of kept facts,
              weight = fact score / number of sentences with the concept (node specificity), averaged over
              facts, top-5 concepts, hub concepts (> 95th percentile frequency, e.g. "the country") dropped;
              ALL sentences seeded with min-max-normalised dense score x 0.05; damping 0.5; no fact kept ->
              dense ranking (their fallback).
  fuse_a_h    RRF of the a_dense and h_ppr passage rankings (k = 60) — dense is the main line
  d3g2_h      first 3 passages of a_dense, then the 2 best h_ppr passages not among them, then a_dense
  fuse_a_c / d3g2_c — the same with the bench1 c_ppr
  c_fix       bench1 PPR with the seeding bug fixed: ALL sentences seeded with min-max dense x 0.05 (bench1 used only
              the top-30 by flat RRF weights 1/(60+r), so the dense order was nearly erased), question concepts as before
  c_noconc    PPR from dense sentence seeds only (no concept seeds) — how much the concept seeds hurt
  g_gate*     Egor's invariant (`ctxgraph/v2/gate.py`): dense order unless a strong graph signal (a rare concept
              shared with a dense top-3 passage = bridge, or links to >= 2 of them = intersection); then <= 2
              passages inserted after dense top-3, nothing removed. No signal -> exactly dense.
  r_gate      the same gate over r_dense (dense + rerank) instead of dense
Cross-encoder here: bge-reranker-v2-m3 int8 ONNX on CPU (exploration only; a test run would use fp32 on GPU).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bench1_run as B  # noqa: E402

from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2.gate import GateConfig, PassageIndex, gated  # noqa: E402
from ctxgraph.v2.graph import Config, Graph, cid  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

OUT = ROOT / "out" / "bench1b_dev"
LINK_TOP_K, PASSAGE_W = 5, 0.05
GATES = {"g_gate": GateConfig(), "g_gate_df2": GateConfig(max_df=2), "g_gate_df5": GateConfig(max_df=5),
         "g_gate_int": GateConfig(max_df=5, need_intersection=True), "g_gate_1": GateConfig(slots=1, keep=4)}
VARIANTS = ["a_dense", "c_ppr", "c_fix", "c_noconc", "r_dense", "h_ppr", "fuse_a_h", "d3g2_h", "fuse_a_c", "d3g2_c",
            *GATES, "r_gate"]


def rrf_rank(*rankings: list[int], k: int = 60) -> list[int]:
    s: dict[int, float] = defaultdict(float)
    for rk in rankings:
        for r, p in enumerate(rk[:500]):
            s[p] += 1 / (k + r)
    return [p for p, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))]


def d3g2(a: list[int], g: list[int]) -> list[int]:
    head = a[:3]
    extra = [p for p in g if p not in head][:2]
    return head + extra + [p for p in a if p not in head and p not in extra]


def fixed_ppr(g: Graph, q: str, dense: np.ndarray, ids: list[str], fallback: list[int], concepts: bool) -> list[int]:
    lo, hi = float(dense.min()), float(dense.max())
    norm = (dense - lo) / (hi - lo) if hi > lo else np.zeros_like(dense)
    seeds = {ids[i]: float(norm[i]) * PASSAGE_W for i in range(len(ids)) if norm[i] > 0}
    if concepts:
        for c, v in g.seeds(q).concept_seeds.items():
            seeds[c] = seeds.get(c, 0.0) + v
    pr = g.ppr(seeds)
    return B.rank_passages({k: v for k, v in pr.items() if k.startswith("f:")}, fallback)


class HippoSeeds:
    def __init__(self, g: Graph, scorer) -> None:
        self.g, self.scorer = g, scorer
        df = dict(g.s.q("SELECT id, n_fragments FROM concepts"))
        self.df = df
        self.hub = float(np.percentile(list(df.values()), 95))
        self.text = dict(g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"))
        self.ment: dict[str, list[str]] = defaultdict(list)
        for f, c in g.s.q("SELECT fragment_id, concept_id FROM mentions"):
            self.ment[f].append(c)

    def rank(self, q: str, dense: np.ndarray, ids: list[str], fallback: list[int]) -> tuple[list[int], dict]:
        order = np.argsort(-dense, kind="stable")
        facts = [ids[i] for i in order if ids[i] in self.text][:LINK_TOP_K]
        sc = self.scorer(q, [self.text[f] for f in facts])
        kept = [(f, s) for f, s in zip(facts, sc, strict=True) if s >= self.g.cfg.min_score]
        info = {"facts": len(facts), "kept": len(kept)}
        if not kept:
            return fallback, info | {"fallback": True}
        w: dict[str, list[float]] = defaultdict(list)
        for f, s in kept:
            for c in self.ment.get(f, []):
                if self.df.get(c, 0) > self.hub:
                    continue
                w[c].append(s / max(self.df.get(c, 1), 1))
        concepts = sorted(((c, float(np.mean(v))) for c, v in w.items()), key=lambda x: -x[1])[:LINK_TOP_K]
        lo, hi = float(dense.min()), float(dense.max())
        norm = (dense - lo) / (hi - lo) if hi > lo else np.zeros_like(dense)
        seeds = {ids[i]: float(norm[i]) * PASSAGE_W for i in range(len(ids)) if norm[i] > 0}
        for c, v in concepts:
            seeds[c] = seeds.get(c, 0.0) + v
        pr = self.g.ppr(seeds)
        frag = {k: v for k, v in pr.items() if k.startswith("f:")}
        return B.rank_passages(frag, fallback), info | {"concepts": [c for c, _ in concepts], "fallback": False}


def main() -> None:
    from ctxgraph.e1b.backends import E5Encoder

    sys.path.insert(0, str(ROOT / "scripts"))
    from v2_onnx_reranker import OnnxReranker

    qs, _ = B.load("musique_dev100", ROOT / ".cache" / "bench", None)
    g = Graph(Store(OUT / "musique_dev100" / "graph.sqlite"), E5Encoder(B.E5, revision=B.E5_REV), config=Config())
    ce = OnnxReranker()
    hs = HippoSeeds(g, ce.score)
    ids, mat = g._fids, g._fmat
    text = hs.text
    pidx = PassageIndex(g)
    gate_info: dict[str, list] = defaultdict(list)
    rows, t0 = [], time.time()
    times = defaultdict(float)
    for qi, q in enumerate(qs):
        t = time.time()
        base = B.run_question(g, q["question"], None, ids, mat)  # a_dense, b_rrf, c_ppr (bench1 code)
        times["a+b+c"] += time.time() - t
        qv = g.enc.one(q["question"])
        dense = mat @ qv
        t = time.time()
        top = [ids[i] for i in np.argsort(-dense, kind="stable") if ids[i] in text][: B.RERANK_K]
        r = B.rank_passages(dict(zip(top, ce.score(q["question"], [text[f] for f in top]), strict=True)), base["a_dense"])
        times["r_dense"] += time.time() - t
        t = time.time()
        h, hinfo = hs.rank(q["question"], dense, ids, base["a_dense"])
        times["h_ppr"] += time.time() - t
        a, c = base["a_dense"], base["c_ppr"]
        t = time.time()
        cf = fixed_ppr(g, q["question"], dense, ids, a, True)
        times["c_fix"] += time.time() - t
        cn = fixed_ppr(g, q["question"], dense, ids, a, False)
        ranks = {"a_dense": a, "c_ppr": c, "c_fix": cf, "c_noconc": cn, "r_dense": r, "h_ppr": h,
                 "fuse_a_h": rrf_rank(a, h), "d3g2_h": d3g2(a, h), "fuse_a_c": rrf_rank(a, c), "d3g2_c": d3g2(a, c)}  # fmt: skip
        qc = {cid(cn_) for cn_, _, _ in X.concepts_of(q["question"])}
        pscore = defaultdict(lambda: -1e9)
        for i in range(len(ids)):
            if ids[i].startswith("f:"):
                p = int(ids[i].split(":")[1][1:])
                pscore[p] = max(pscore[p], float(dense[i]))
        t = time.time()
        for name, gc in GATES.items():
            ranks[name], gi = gated(a, pscore, pidx, qc, gc)
            gate_info[name].append(gi)
        times["g_gate"] += (time.time() - t) / len(GATES)
        ranks["r_gate"], gi = gated(r, pscore, pidx, qc, GateConfig())
        gate_info["r_gate"].append(gi)
        row = {"id": q["id"], "type": q["type"], "gold": q["gold"], "h_info": hinfo}
        for v, rk in ranks.items():
            row[v] = {"r2": B.recall(rk, q["gold"], 2), "r5": B.recall(rk, q["gold"], 5), "top5": rk[:5]}
        rows.append(row)
        if qi % 20 == 0:
            print(f"  q {qi}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows), "utf-8")

    def found(r, v):
        return set(r[v]["top5"]) & set(r["gold"])

    wins_c = [r for r in rows if found(r, "c_ppr") - found(r, "a_dense")]  # graph found what dense missed
    L = ["# Bench 1b — разведка на 100 вопросах dev MuSiQue (не тестовые; без вызовов LLM)\n",
         "Корпус — собственные абзацы этих 100 вопросов (как в HippoRAG: dev-вопросы со своими абзацами), поэтому уровни выше, "
         "чем на тестовом корпусе из 11 656 абзацев. Переранжировщик — bge-reranker-v2-m3 int8 (ONNX, CPU); в тестовом прогоне — fp32 на GPU.\n",
         f"| вариант | R@2 | R@5 | R@5 − dense, п.п. [95 %] | находит «граф-победы» c_ppr (из {len(wins_c)}) | проигрыши dense (dense нашёл, вариант нет) | время на вопрос, с |",
         "|---|---|---|---|---|---|---|"]  # fmt: skip
    tq = {"a_dense": times["a+b+c"], "c_ppr": times["a+b+c"], "r_dense": times["r_dense"], "h_ppr": times["h_ppr"],
          "c_fix": times["c_fix"], "g_gate": times["g_gate"]}
    for v in VARIANTS:
        r2 = 100 * np.mean([r[v]["r2"] for r in rows])
        r5 = 100 * np.mean([r[v]["r5"] for r in rows])
        m, lo, hi = B.bootstrap([r[v]["r5"] - r["a_dense"]["r5"] for r in rows])
        kept = sum(bool((found(r, "c_ppr") - found(r, "a_dense")) & found(r, v)) for r in wins_c)
        loss = sum(bool(found(r, "a_dense") - found(r, v)) for r in rows)
        tt = tq.get(v)
        L.append(f"| {v} | {r2:.1f} | {r5:.1f} | {100 * m:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}] | {kept} | {loss} | "
                 f"{tt / len(rows):.2f} |" if tt is not None else
                 f"| {v} | {r2:.1f} | {r5:.1f} | {100 * m:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}] | {kept} | {loss} | — |")  # fmt: skip
    fb = sum(r["h_info"].get("fallback", False) for r in rows)
    nc = np.mean([len(r["h_info"].get("concepts", [])) for r in rows])
    L.append(f"\nh_ppr: фильтр фактов оставил ≥ 1 предложение в {100 - fb} вопросах из 100 (иначе — dense, как у HippoRAG 2); "
             f"понятий-стартов в среднем {nc:.1f}.")  # fmt: skip
    L.append("\n**Шлюз (инвариант Егора).** Изменённые вопросы — где граф сработал и первая пятёрка отличается от базы; "
             "точность вставок — доля вставленных абзацев, которые являются опорными.\n")
    L.append("| вариант | изменено вопросов | вставлено абзацев | из них опорных | R@5 на изменённых: база → шлюз | R@5 на остальных == база |")
    L.append("|---|---|---|---|---|---|")
    for name, infos in gate_info.items():
        base_v = "r_dense" if name == "r_gate" else "a_dense"
        ch = [i for i, gi in enumerate(infos) if gi["changed"]]
        ins = [(i, p) for i in ch for p in infos[i]["picked"]]
        hit = sum(p in rows[i]["gold"] for i, p in ins)
        b5 = np.mean([rows[i][base_v]["r5"] for i in ch]) * 100 if ch else float("nan")
        g5 = np.mean([rows[i][name]["r5"] for i in ch]) * 100 if ch else float("nan")
        same = all(rows[i][name]["top5"] == rows[i][base_v]["top5"] for i in range(len(rows)) if i not in set(ch))
        L.append(f"| {name} | {len(ch)} | {len(ins)} | {hit} | {b5:.1f} → {g5:.1f} | {'да' if same else 'НЕТ'} |")
    (ROOT / "reports" / "bench1b_dev.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
