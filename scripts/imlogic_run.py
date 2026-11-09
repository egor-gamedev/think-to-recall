"""IMLogic, memory-level retrieval (internal research only: no license in the repo, source HaluMem is CC BY-NC-ND 4.0).

    python scripts/imlogic_run.py --split dev --reranker none     # local debug: d0 / G only
    python scripts/imlogic_run.py --split main --device cuda      # main run (after the prereg is frozen)

Unit = memory records of one user (all `memory_points[].memory_content` of HaluMem-Medium, deduplicated), as in the
IMLogic memory-level setting; one graph per user, each record an episode `p{i}`. Per question: rank that user's
records and report Recall@k (k = 1, 5, 10, 30) of the decisive memory `memory_l` and of the similar one `memory_s`.
Variants as in bench1b (`scripts/bench1b_run.py`), gate config `scripts/bench1b_config.json` unchanged:
  d0 dense e5 | K dense top-40 -> bge-reranker-v2-m3 | G d0 + gate | KG K + gate
Split: users 0..19; dev = 2 users drawn with seed 0 (debug only), main = the other 18. No LLM calls.
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

DATA = ROOT / ".cache" / "bench" / "imlogic"
KS = (1, 5, 10, 30)
DEV_USERS = sorted(random.Random(0).sample(range(20), 2))


def load_user(i: int, data: Path) -> tuple[list[dict], list[dict]]:
    with (data / "HaluMem-Medium.jsonl").open(encoding="utf-8") as f:
        for j, line in enumerate(f):
            if j == i:
                u = json.loads(line)
                break
    mem = list(dict.fromkeys(m["memory_content"] for s in u["sessions"] for m in s["memory_points"]))
    key = {m: k for k, m in enumerate(mem)}
    qa = json.loads((data / "qa" / f"qa_user{i:02d}.json").read_text("utf-8"))
    qs = [{"id": f"u{i:02d}_{n}", "question": q["query"], "type": q["category"], "l": key[q["memory_l"]],
           "s": key[q["memory_s"]]} for n, q in enumerate(qa)]  # fmt: skip
    return qs, [{"title": "", "text": m} for m in mem]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "main"], required=True)
    ap.add_argument("--users", type=int, nargs="*", help="debug only: override the user list")
    ap.add_argument("--out", default="out/imlogic")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reranker", choices=["fp32", "int8", "none"], default="fp32")
    ap.add_argument("--config", default=str(ROOT / "scripts" / "bench1b_config.json"))
    a = ap.parse_args()
    from ctxgraph.e1b.backends import E5Encoder

    gcfg = PprGateConfig(**json.loads(Path(a.config).read_text("utf-8"))["gate"])
    users = a.users if a.users is not None else (DEV_USERS if a.split == "dev" else [u for u in range(20) if u not in DEV_USERS])
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
    rows, t0 = [], time.time()
    for u in users:
        qs, corpus = load_user(u, Path(a.data))
        g, binfo = B.build(corpus, B.PrecomputedEncoder(e5), out_root / f"user{u:02d}", a.device)
        print(f"user {u}: {len(corpus)} records, {len(qs)} questions; graph {binfo}  {time.time() - t0:.0f}s", flush=True)
        ids, mat = g._fids, g._fmat
        text = dict(g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"))
        idx = PassageIndex(g)
        sent_of: dict[int, list[str]] = {}
        for f in ids:
            sent_of.setdefault(B.passage_of(f), []).append(f)
        for q in qs:
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
            row = {"id": q["id"], "user": u, "type": q["type"], "l": q["l"], "s": q["s"],
                   "gate": {v: {"signal": i_["signal"], "changed": i_["changed"]} for v, i_ in info.items()}}  # fmt: skip
            for v in variants:
                rk = ranks[v]
                row[v] = {"rank_l": rk.index(q["l"]) + 1 if q["l"] in rk else None,
                          "rank_s": rk.index(q["s"]) + 1 if q["s"] in rk else None, "top5": rk[:5]}  # fmt: skip
            rows.append(row)
    with (out_root / "rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def hit(r, v, m, k):
        x = r[v][f"rank_{m}"]
        return float(x is not None and x <= k)

    summ: dict = {"split": a.split, "users": users, "n": len(rows), "gate": gcfg.__dict__, "reranker": a.reranker,
                  "recall": {f"{v}_{m}@{k}": float(np.mean([hit(r, v, m, k) for r in rows]))
                             for v in variants for m in ("l", "s") for k in KS},
                  "seconds": round(time.time() - t0, 1)}  # fmt: skip
    hyps = {"H1": ("KG", "K"), "H2": ("G", "d0")} if scorer else {"H2": ("G", "d0")}
    ps, diffs = {}, {}
    for h, (x, y) in hyps.items():
        d = [hit(r, x, "l", 5) - hit(r, y, "l", 5) for r in rows]
        ps[h], diffs[h] = perm_p(d), B.bootstrap(d)
    adj = holm(ps)
    summ["hypotheses"] = {h: {"compare": f"{x}-{y}", "metric": "memory_l Recall@5", "diff": diffs[h], "p": ps[h],
                              "p_holm": adj[h],
                              "wins": sum(hit(r, x, "l", 5) > hit(r, y, "l", 5) for r in rows),
                              "losses": sum(hit(r, x, "l", 5) < hit(r, y, "l", 5) for r in rows),
                              "changed": sum(r[x]["top5"] != r[y]["top5"] for r in rows)}
                          for h, (x, y) in hyps.items()}  # fmt: skip
    summ["by_type"] = {t: {"n": sum(r["type"] == t for r in rows),
                           **{f"{v}_l@5": float(np.mean([hit(r, v, "l", 5) for r in rows if r["type"] == t])) for v in variants}}
                       for t in sorted({r["type"] for r in rows})}  # fmt: skip
    (out_root / "summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
    print(json.dumps({k: summ[k] for k in ("recall", "hypotheses")}, indent=1))


if __name__ == "__main__":
    main()
