"""Benchmark 1b (second attempt), per `reports/prereg_bench1b.md` — no LLM calls.

    python scripts/bench1b_run.py --dataset musique --device cuda --out /kaggle/working/out1b

Same data and graph build as benchmark 1 (`scripts/bench1_run.py`). Variants (passage Recall@2/@5):
  d0   e5 dense (= bench1 variant a)                          — base of the gate
  K    dense top-40 sentences -> bge-reranker-v2-m3            — mandatory control
  G    d0 + gate `ctxgraph.v2.gate.ppr_gated`                  — H1: G vs d0
  KG   K + the same gate                                       — H2: KG vs K
Gate configuration: `scripts/bench1b_config.json` (frozen with the prereg). Invariant checked on every question:
where the gate changes nothing, the output IS the base ranking (else the run aborts).
H1, H2: one-sided paired sign-flip permutation test on per-question R@5 differences (10 000, seed 0), Holm over
the two (MuSiQue only); 95 % paired bootstrap intervals; wins / losses on the changed questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bench1_run as B  # noqa: E402

from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated, start_ppr  # noqa: E402
from ctxgraph.v2.graph import cid  # noqa: E402

VARIANTS = ["d0", "K", "G", "KG"]
HYPS = {"H1": ("G", "d0"), "H2": ("KG", "K")}


def perm_p(diff: list[float], n: int = 10000, seed: int = 0) -> float:
    """One-sided (mean > 0) paired sign-flip permutation test."""
    d = np.asarray(diff, dtype=float)
    obs = d.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    return float((1 + np.sum((signs * d).mean(1) >= obs - 1e-12)) / (n + 1))


def holm(ps: dict[str, float]) -> dict[str, float]:
    out, run = {}, 0.0
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
        run = max(run, min(1.0, (len(ps) - i) * p))
        out[k] = run
    return out


def wins_losses(rows: list[dict], x: str, y: str) -> dict:
    w = lo = qw = ql = ch = 0
    for r in rows:
        if r[x]["top5"] == r[y]["top5"]:
            continue
        ch += 1
        got, had = set(r[x]["top5"]) & set(r["gold"]), set(r[y]["top5"]) & set(r["gold"])
        w, lo = w + len(got - had), lo + len(had - got)
        qw, ql = qw + (len(got) > len(had)), ql + (len(got) < len(had))
    return {"changed": ch, "wins": w, "losses": lo, "q_wins": qw, "q_losses": ql}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["musique", "2wiki", "musique_dev100", "musique_dev200", "musique_dev200b"])  # fmt: skip
    ap.add_argument("--out", default="out/bench1b")
    ap.add_argument("--data", default=str(ROOT / ".cache" / "bench"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reranker", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--config", default=str(ROOT / "scripts" / "bench1b_config.json"))
    ap.add_argument("--first", type=int, default=0, help="debug only: search just the first N questions")
    a = ap.parse_args()
    from ctxgraph.e1b.backends import E5Encoder

    conf = json.loads(Path(a.config).read_text("utf-8"))
    gcfg = PprGateConfig(**conf["gate"])
    out = Path(a.out) / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    qs, corpus = B.load(a.dataset, Path(a.data), None)
    qs = qs[: a.first] if a.first else qs
    print(f"{a.dataset}: {len(qs)} questions, {len(corpus)} passages; gate {gcfg}", flush=True)
    enc = B.PrecomputedEncoder(E5Encoder(B.E5, device=a.device, revision=B.E5_REV))
    g, binfo = B.build(corpus, enc, out, a.device)
    print("graph:", binfo, flush=True)
    if a.reranker == "fp32":
        from ctxgraph.e1b.selection import Reranker

        scorer = Reranker(B.BGE, revision=B.BGE_REV, device=a.device).score
    else:
        from v2_onnx_reranker import OnnxReranker

        scorer = OnnxReranker().score
    ids, mat = g._fids, g._fmat
    text = dict(g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"))
    idx = PassageIndex(g)
    sent_of: dict[int, list[str]] = {}
    for f in ids:
        sent_of.setdefault(B.passage_of(f), []).append(f)
    rows, t0, tk, tg = [], time.time(), 0.0, 0.0
    with (out / "recall.jsonl").open("w", encoding="utf-8") as fh:
        for i, q in enumerate(qs):
            dense = mat @ g.enc.one(q["question"])
            order = np.argsort(-dense, kind="stable")
            d0 = B.rank_passages({ids[j]: float(dense[j]) for j in order[: B.TOPN_DENSE]})
            t = time.time()
            top = [ids[j] for j in order if ids[j] in text][: B.RERANK_K]
            k = B.rank_passages(dict(zip(top, scorer(q["question"], [text[f] for f in top]), strict=True)), d0)
            tk += time.time() - t
            t = time.time()
            lo, hi = float(dense.min()), float(dense.max())
            norm = {ids[j]: (float(dense[j]) - lo) / (hi - lo) for j in range(len(ids))} if hi > lo else {}
            qc = {cid(c) for c, _, _ in X.concepts_of(q["question"])} & set(idx.df)
            cache: dict = {}

            def ppr(anchors, qw, norm=norm, cache=cache):
                key = (tuple(anchors), tuple(sorted(qw.items())))
                if key not in cache:
                    cache[key] = start_ppr(g, sent_of, norm, anchors, qw)
                return cache[key]

            ranks, info = {"d0": d0, "K": k}, {}
            for v, base in (("G", d0), ("KG", k)):
                ranks[v], info[v] = ppr_gated(base, idx, qc, ppr, gcfg)
                if not info[v]["changed"] and ranks[v] != base:  # the invariant
                    raise SystemExit(f"INVARIANT BROKEN: {q['id']} {v}")
            tg += time.time() - t
            row = {"id": q["id"], "type": q["type"], "gold": q["gold"],
                   "gate": {v: {kk: vv for kk, vv in info[v].items()} for v in ("G", "KG")}}  # fmt: skip
            for v in VARIANTS:
                row[v] = {"r2": B.recall(ranks[v], q["gold"], 2), "r5": B.recall(ranks[v], q["gold"], 5), "top5": ranks[v][:5]}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"  q {i}/{len(qs)}  {time.time() - t0:.0f}s", flush=True)

    summ: dict = {"dataset": a.dataset, "n": len(rows), "gate": conf, "build": binfo, "recall": {}, "diffs": {},
                  "hypotheses": {}, "by_type": {}, "seconds": {"rerank": round(tk, 1), "gate": round(tg, 1),
                  "search": round(time.time() - t0, 1)}, "reranker": a.reranker,
                  "models": {"e5": f"{B.E5}@{B.E5_REV}", "bge": f"{B.BGE}@{B.BGE_REV}"}}  # fmt: skip
    for kk in (2, 5):
        for v in VARIANTS:
            summ["recall"][f"{v}@{kk}"] = float(np.mean([r[v][f"r{kk}"] for r in rows]))
    for x, y in (("G", "d0"), ("KG", "K"), ("K", "d0"), ("G", "K"), ("KG", "d0")):
        summ["diffs"][f"{x}-{y}@5"] = B.bootstrap([r[x]["r5"] - r[y]["r5"] for r in rows])
    ps = {h: perm_p([r[x]["r5"] - r[y]["r5"] for r in rows]) for h, (x, y) in HYPS.items()}
    adj = holm(ps)
    for h, (x, y) in HYPS.items():
        summ["hypotheses"][h] = {"compare": f"{x}-{y}", "diff_r5": summ["diffs"][f"{x}-{y}@5"], "p": ps[h],
                                 "p_holm": adj[h], **wins_losses(rows, x, y),
                                 "signal": sum(r["gate"][x]["signal"] for r in rows)}  # fmt: skip
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        summ["by_type"][t] = {"n": len(rs), **{f"{v}@5": float(np.mean([r[v]["r5"] for r in rs])) for v in VARIANTS}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
    print(json.dumps({k: summ[k] for k in ("recall", "hypotheses")}, indent=1))


if __name__ == "__main__":
    main()
