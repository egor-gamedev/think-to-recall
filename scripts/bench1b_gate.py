"""Bench 1b: PPR gate (`ctxgraph.v2.gate.ppr_gated`) — threshold sweep on dev-100, confirmation on dev-200.

    python scripts/bench1b_gate.py --dataset musique_dev100            # sweep -> out/bench1b_dev/gate_sweep_dev100.json
    python scripts/bench1b_gate.py --dataset musique_dev200 --only CFG # confirm one config (and its neighbours)

Criterion (Egor): only questions where the gate changes the top-5 count. Win = a gold passage in the gate top-5 that
the base top-5 lacks; loss = the reverse. A config qualifies if wins > losses. Bases: a_dense (e5) and r_dense
(dense top-40 sentences -> bge-reranker-v2-m3 int8 ONNX, CPU; cached per dataset).
"""

from __future__ import annotations

import argparse
import itertools
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
from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated, start_ppr  # noqa: E402
from ctxgraph.v2.graph import Config, Graph, cid  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

OUT = ROOT / "out" / "bench1b_dev"
GRID = {"max_df": [2, 3, 5, 10, 20], "anchor": [2, 3], "margin": [0.0, 0.25, 1.0, 3.0, 10.0], "slots": [1, 2]}


def cfg_name(base: str, c: PprGateConfig) -> str:
    rar = {"abs": f"df{c.max_df}", "rel": f"rel{c.max_df}", "idf": "idf"}[c.rarity]
    return f"{base}|{rar}|a{c.anchor}|m{c.margin:g}|s{c.slots}"


def r_dense_cache(ds: str, g: Graph, qs: list[dict], ids, mat, text, a_ranks) -> list[list[int]]:
    path = OUT / ds / "r_dense_top20.json"
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    if ds == "musique_dev100" and (OUT / "rows.json").exists():  # computed by bench1b_dev.py (top-5 is all R@5 needs)
        rows = {r["id"]: r for r in json.loads((OUT / "rows.json").read_text("utf-8"))}
        out = [rows[q["id"]]["r_dense"]["top5"] for q in qs]
    else:
        from v2_onnx_reranker import OnnxReranker

        ce, out, t0 = OnnxReranker(), [], time.time()
        for qi, q in enumerate(qs):
            dense = mat @ g.enc.one(q["question"])
            top = [ids[i] for i in np.argsort(-dense, kind="stable") if ids[i] in text][: B.RERANK_K]
            sc = ce.score(q["question"], [text[f] for f in top])
            out.append(B.rank_passages(dict(zip(top, sc, strict=True)), a_ranks[qi])[:20])
            if qi % 20 == 0:
                print(f"  rerank q {qi}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
    path.write_text(json.dumps(out), "utf-8")
    return out


def main() -> None:
    from ctxgraph.e1b.backends import E5Encoder

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="musique_dev100", choices=["musique_dev100", "musique_dev200", "musique_dev200b"])
    ap.add_argument("--only", nargs="*", help="config names to evaluate (default: the whole grid)")
    ap.add_argument("--rarity", default="abs", choices=["abs", "rel", "idf"])
    ap.add_argument("--tag", default="", help="suffix of the output file")
    args = ap.parse_args()
    ds = args.dataset
    qs, _ = B.load(ds, ROOT / ".cache" / "bench", None)
    g = Graph(Store(OUT / ds / "graph.sqlite"), E5Encoder(B.E5, revision=B.E5_REV), config=Config())
    ids, mat = g._fids, g._fmat
    text = dict(g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"))
    idx = PassageIndex(g)
    sent_of: dict[int, list[str]] = defaultdict(list)
    for f in ids:
        sent_of[B.passage_of(f)].append(f)

    a_ranks, dense_all, qcs = [], [], []
    for q in qs:
        dense = mat @ g.enc.one(q["question"])
        order = np.argsort(-dense, kind="stable")
        a_ranks.append(B.rank_passages({ids[i]: float(dense[i]) for i in order[: B.TOPN_DENSE]}))
        lo, hi = float(dense.min()), float(dense.max())
        dense_all.append({ids[i]: (float(dense[i]) - lo) / (hi - lo) for i in range(len(ids))} if hi > lo else {})
        qcs.append({cid(c) for c, _, _ in X.concepts_of(q["question"])} & set(idx.df))
    need_r = not args.only or any(n.startswith("r_dense") for n in args.only)
    bases = {"a_dense": a_ranks} | ({"r_dense": r_dense_cache(ds, g, qs, ids, mat, text, a_ranks)} if need_r else {})

    ppr_cache: dict[tuple, dict[int, float]] = {}

    def make_ppr(qi: int):
        def run(anchors: list[int], qw: dict[str, float]) -> dict[int, float]:
            key = (qi, tuple(anchors), tuple(sorted(qw.items())))
            if key not in ppr_cache:
                ppr_cache[key] = start_ppr(g, sent_of, dense_all[qi], anchors, qw)
            return ppr_cache[key]

        return run

    configs = [(b, PprGateConfig(**dict(zip(GRID, vals, strict=True)), rarity=args.rarity)) for b in bases
               for vals in itertools.product(*GRID.values())]  # fmt: skip
    if args.only:
        configs = [(b, c) for b, c in configs if cfg_name(b, c) in set(args.only)]
    configs = list({cfg_name(b, c): (b, c) for b, c in configs}.values())  # "idf" ignores max_df: drop duplicates
    res, t0 = [], time.time()
    for b, c in configs:
        wins = losses = qw = ql = changed = signal = 0
        r5 = []
        for qi, q in enumerate(qs):
            base = bases[b][qi]
            out, info = ppr_gated(base, idx, qcs[qi], make_ppr(qi), c)
            gold = set(q["gold"])
            got, had = set(out[:5]) & gold, set(base[:5]) & gold
            signal += info["signal"]
            if out[:5] != base[:5]:
                changed += 1
                wins += len(got - had)
                losses += len(had - got)
                qw += len(got) > len(had)
                ql += len(got) < len(had)
            else:
                assert out[:5] == base[:5]
            r5.append((B.recall(out, q["gold"], 5), B.recall(base, q["gold"], 5)))
        m, lo, hi = B.bootstrap([x - y for x, y in r5])
        res.append({"config": cfg_name(b, c), "base": b, "signal": signal, "changed": changed, "wins": wins,
                    "losses": losses, "q_wins": qw, "q_losses": ql, "r5_base": float(np.mean([y for _, y in r5])),
                    "r5_gate": float(np.mean([x for x, _ in r5])), "diff": [m, lo, hi]})  # fmt: skip
    print(f"{len(configs)} configs, {time.time() - t0:.0f}s, {len(ppr_cache)} PPR runs", flush=True)
    tag = ds.replace("musique_", "")
    (OUT / f"gate_sweep_{tag}{args.tag}.json").write_text(json.dumps(res, indent=1), "utf-8")
    for r in sorted(res, key=lambda r: (-(r["wins"] - r["losses"]), r["changed"]))[:25]:
        print(f"{r['config']:34s} signal {r['signal']:3d} changed {r['changed']:3d}  wins {r['wins']:3d} losses "
              f"{r['losses']:3d}  R@5 {100 * r['r5_base']:.1f}->{100 * r['r5_gate']:.1f}")  # fmt: skip


if __name__ == "__main__":
    main()
