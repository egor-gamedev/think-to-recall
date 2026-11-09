"""ImplexConv (MIT; HF `Kaylee0501/ImplexConv` @ baf436948f), session-level retrieval — no LLM calls.

    python scripts/implexconv_run.py --split dev --reranker none     # local debug: d0 / G only
    python scripts/implexconv_run.py --split main --device cuda      # main run (after the prereg is frozen)

One graph per example: each session of `conversation` is an episode `p{i}` (sentences inside). Per question: rank the
example's sessions (session score = max over its sentences) and report Recall@k (k = 1, 3, 5, 10) of the evidence
sessions (`retrieved_conv` / `retrieved_conv_ids`, given as session texts — matched exactly).
Variants and gate as in bench1b (`scripts/bench1b_config.json`, unchanged): d0 | K | G | KG.
Sample (seed 0): main = 100 opposed + 100 supportive examples; dev = 5 + 5 other examples (debug only).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bench1_run as B  # noqa: E402
from bench1b_run import holm, perm_p  # noqa: E402

from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated, start_ppr  # noqa: E402
from ctxgraph.v2.graph import cid  # noqa: E402

DATA = ROOT / ".cache" / "bench" / "implexconv" / "data"
FILES = {"opposed": ["opposed-00000-of-00002.parquet", "opposed-00001-of-00002.parquet"],
         "supportive": ["supportive-00000-of-00001.parquet"]}  # fmt: skip
KS = (1, 3, 5, 10)


def sample(split: str) -> dict[str, list[int]]:
    """Example indices per kind; main and dev disjoint (both from one seeded shuffle)."""
    sizes = {"opposed": 1550, "supportive": 814}
    out = {}
    for kind, n in sizes.items():
        order = list(range(n))
        random.Random(0).shuffle(order)
        out[kind] = sorted(order[:100]) if split == "main" else sorted(order[100:105])
    return out


def load(split: str, data: Path) -> list[dict]:
    import pyarrow.parquet as pq

    exs = []
    for kind, idxs in sample(split).items():
        rows = [r for f in FILES[kind] for r in pq.read_table(data / f).to_pylist()]
        assert len(rows) == {"opposed": 1550, "supportive": 814}[kind], (kind, len(rows))
        for i in idxs:
            r = rows[i]
            # supportive: list of session texts, evidence given as texts; opposed: {session id: text} (a fixed-width
            # struct, unused ids are None), evidence given as ids
            raw = r["conversation"]
            items = sorted(((k, s) for k, s in raw.items() if s), key=lambda x: int(x[0])) if isinstance(raw, dict) \
                else list(enumerate(raw))
            conv = [s for _, s in items]
            key: dict = {}
            for j, (k, s) in enumerate(items):
                key.setdefault(str(k) if isinstance(raw, dict) else s, j)
            qs = []
            for n, q in enumerate(r["qa"]):
                ev = q.get("retrieved_conv") or [str(x) for x in (q.get("retrieved_conv_ids") or [])]
                gold = sorted({key[s] for s in ev if s in key})
                if gold:
                    qs.append({"id": f"{kind[:3]}{i}_{n}", "question": q["question"], "gold": gold})
            exs.append({"id": f"{kind}{i}", "kind": kind, "corpus": [{"title": "", "text": s} for s in conv], "qs": qs})
    return exs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "main"], required=True)
    ap.add_argument("--out", default="out/implexconv")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reranker", choices=["fp32", "int8", "none"], default="fp32")
    ap.add_argument("--config", default=str(ROOT / "scripts" / "bench1b_config.json"))
    a = ap.parse_args()
    from ctxgraph.e1b.backends import E5Encoder

    gcfg = PprGateConfig(**json.loads(Path(a.config).read_text("utf-8"))["gate"])
    scorer = None
    if a.reranker == "fp32":
        from ctxgraph.e1b.selection import Reranker

        scorer = Reranker(B.BGE, revision=B.BGE_REV, device=a.device).score
    elif a.reranker == "int8":
        from v2_onnx_reranker import OnnxReranker

        scorer = OnnxReranker().score
    variants = ["d0", "G"] + (["K", "KG"] if scorer else [])
    e5 = E5Encoder(B.E5, device=a.device, revision=B.E5_REV)
    out_root = Path(a.out) / a.split
    out_root.mkdir(parents=True, exist_ok=True)
    exs = load(a.split, Path(a.data))
    print(f"{len(exs)} examples, {sum(len(e['qs']) for e in exs)} questions", flush=True)
    rows, t0 = [], time.time()
    for ex in exs:
        g, binfo = B.build(ex["corpus"], B.PrecomputedEncoder(e5), out_root / ex["id"], a.device)
        print(f"{ex['id']}: {len(ex['corpus'])} sessions, {len(ex['qs'])} q; {binfo['fragments']} sentences  {time.time() - t0:.0f}s", flush=True)
        ids, mat = g._fids, g._fmat
        text = dict(g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"))
        idx = PassageIndex(g)
        sent_of: dict[int, list[str]] = {}
        for f in ids:
            sent_of.setdefault(B.passage_of(f), []).append(f)
        for q in ex["qs"]:
            dense = mat @ g.enc.one(q["question"])
            order = np.argsort(-dense, kind="stable")
            d0 = B.rank_passages({ids[j]: float(dense[j]) for j in order})
            ranks = {"d0": d0}
            if scorer:
                top = [ids[j] for j in order if ids[j] in text][: B.RERANK_K]
                ranks["K"] = B.rank_passages(dict(zip(top, scorer(q["question"], [text[f] for f in top]), strict=True)), d0)
            lo, hi = float(dense.min()), float(dense.max())
            norm = {ids[j]: (float(dense[j]) - lo) / (hi - lo) for j in range(len(ids))} if hi > lo else {}
            qc = {cid(c) for c, _, _ in X.concepts_of(q["question"])} & set(idx.df)
            cache: dict = {}

            def ppr(anchors, qw, norm=norm, cache=cache, g=g, sent_of=sent_of):
                key = (tuple(anchors), tuple(sorted(qw.items())))
                if key not in cache:
                    cache[key] = start_ppr(g, sent_of, norm, anchors, qw)
                return cache[key]

            info = {}
            for v, b in (("G", "d0"), ("KG", "K")):
                if b not in ranks:
                    continue
                ranks[v], info[v] = ppr_gated(ranks[b], idx, qc, ppr, gcfg)
                if not info[v]["changed"] and ranks[v] != ranks[b]:  # the invariant
                    raise SystemExit(f"INVARIANT BROKEN: {q['id']} {v}")
            row = {"id": q["id"], "kind": ex["kind"], "gold": q["gold"],
                   "gate": {v: {"signal": i_["signal"], "changed": i_["changed"]} for v, i_ in info.items()}}  # fmt: skip
            for v in variants:
                row[v] = {f"r{k}": B.recall(ranks[v], q["gold"], k) for k in KS} | {"top10": ranks[v][:10]}
            rows.append(row)
    with (out_root / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ: dict = {"split": a.split, "n": len(rows), "gate": gcfg.__dict__, "reranker": a.reranker, "recall": {},
                  "seconds": round(time.time() - t0, 1)}  # fmt: skip
    for kind in ("all", "opposed", "supportive"):
        rs = [r for r in rows if kind in ("all", r["kind"])]
        for v in variants:
            for k in KS:
                summ["recall"][f"{kind}:{v}@{k}"] = float(np.mean([r[v][f"r{k}"] for r in rs])) if rs else None
    hyps = {"H1": ("KG", "K"), "H2": ("G", "d0")} if scorer else {"H2": ("G", "d0")}
    ps, diffs = {}, {}
    for h, (x, y) in hyps.items():
        d = [r[x]["r5"] - r[y]["r5"] for r in rows]
        ps[h], diffs[h] = perm_p(d), B.bootstrap(d)
    adj = holm(ps)
    summ["hypotheses"] = {h: {"compare": f"{x}-{y}", "metric": "evidence-session Recall@5", "diff": diffs[h], "p": ps[h],
                              "p_holm": adj[h], "changed": sum(r[x]["top10"][:5] != r[y]["top10"][:5] for r in rows),
                              "wins": sum(r[x]["r5"] > r[y]["r5"] for r in rows),
                              "losses": sum(r[x]["r5"] < r[y]["r5"] for r in rows)}
                          for h, (x, y) in hyps.items()}  # fmt: skip
    (out_root / "summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
    print(json.dumps({k: summ[k] for k in ("recall", "hypotheses")}, indent=1))


if __name__ == "__main__":
    main()
