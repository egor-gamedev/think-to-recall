"""Benchmark 1 (prereg `reports/prereg_bench1.md`): passage Recall@2/@5 on MuSiQue and 2WikiMultiHopQA.

    python scripts/bench1_run.py --dataset musique --out out/bench1 [--device cuda] [--limit N]

No LLM calls. Data = the HippoRAG 2 release (`osunlp/HippoRAG_2`, revision 5ec05b38de): the same 1000
questions and corpora as the paper. `musique_dev100` = 100 MuSiQue answerable-dev questions disjoint from
the test 1000 (seed 0), with a corpus of their own paragraphs — the only set allowed for tuning;
`musique_dev200` = 200 more, disjoint from both (seed 1) — confirmation only; `musique_dev200b` = 200 more
(seed 2), disjoint from all three — second confirmation.

Variants (all rank the corpus passages; passage score = max over its sentences, as Recall in HippoRAG 2
is per passage):
  a_dense    — e5-base over sentences;
  b_rrf      — dense + BM25 (FTS5 over lemmas) fused by RRF (k = 60);
  c_ppr      — v2 graph: hybrid seeds + concepts of the question -> personalized PageRank;
  d_rerank   — c_ppr top-40 sentences re-scored by bge-reranker-v2-m3 (query = question).
Ties / passages not reached fall back to the next simpler variant's order (c -> b, d -> c).
`--limit N` (debug only): first N questions with a corpus of just their paragraphs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import text as T  # noqa: E402
from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

HIPPO_REPO, HIPPO_REV = "osunlp/HippoRAG_2", "5ec05b38de"
DEV_REPO, DEV_REV, DEV_FILE = "dgslibisey/MuSiQue", "c8f4f8c946", "musique_ans_v1.0_dev.jsonl"
E5, E5_REV = "intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a"
BGE, BGE_REV = "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
VARIANTS = ["a_dense", "b_rrf", "c_ppr", "d_rerank"]
TOPN_DENSE = TOPN_BM25 = 500
RERANK_K = 40


def fetch(name: str, data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    p = data_dir / name
    if p.exists():
        return p
    repo, rev = (DEV_REPO, DEV_REV) if name == DEV_FILE else (HIPPO_REPO, HIPPO_REV)
    return Path(hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=str(data_dir)))


def load(dataset: str, data_dir: Path, limit: int | None) -> tuple[list[dict], list[dict]]:
    """questions [{id, question, type, gold: [corpus idx]}], corpus [{title, text}]."""
    if dataset in ("musique", "2wiki"):
        qf, cf = ("musique.json", "musique_corpus.json") if dataset == "musique" else ("2wikimultihopqa.json", "2wikimultihopqa_corpus.json")
        raw = json.loads(fetch(qf, data_dir).read_text("utf-8"))
        corpus = json.loads(fetch(cf, data_dir).read_text("utf-8"))
    else:  # musique_dev100: tuning set; musique_dev200: confirmation set; both disjoint from the test 1000 and each other
        test = {q["id"] for q in json.loads(fetch("musique.json", data_dir).read_text("utf-8"))}
        dev = [json.loads(ln) for ln in fetch(DEV_FILE, data_dir).read_text("utf-8").splitlines() if ln.strip()]
        pool = [d for d in dev if d["id"] not in test]
        raw = random.Random(0).sample(pool, 100)
        if dataset in ("musique_dev200", "musique_dev200b"):
            used = {d["id"] for d in raw}
            raw = random.Random(1).sample([d for d in pool if d["id"] not in used], 200)
        if dataset == "musique_dev200b":
            used |= {d["id"] for d in raw}
            raw = random.Random(2).sample([d for d in pool if d["id"] not in used], 200)
        corpus = list({(p["title"], p["paragraph_text"]): {"title": p["title"], "text": p["paragraph_text"]}
                       for d in raw for p in d["paragraphs"]}.values())  # fmt: skip
    key = {(c["title"], c["text"]): i for i, c in enumerate(corpus)}
    qs, missing = [], 0
    for x in raw:
        if dataset == "2wiki":
            sup = {t for t, _ in x["supporting_facts"]}
            gold_keys = [(t, " ".join(s)) for t, s in x["context"] if t in sup]
            qid, qtype = x["_id"], x["type"]
        else:
            gold_keys = [(p["title"], p["paragraph_text"]) for p in x["paragraphs"] if p["is_supporting"]]
            qid, qtype = x["id"], x["id"].split("__")[0]
        gold = [key[k] for k in gold_keys if k in key]
        missing += len(gold_keys) - len(gold)
        qs.append({"id": qid, "question": x["question"], "type": qtype, "gold": sorted(set(gold)),
                   "paragraphs": x.get("paragraphs") or x.get("context")})  # fmt: skip
    if missing:
        print(f"WARNING: {missing} gold passages not found in the corpus", flush=True)
    if limit:  # debug: own small corpus from the questions' paragraphs
        qs = qs[:limit]
        if dataset != "2wiki":
            keep = {key[(p["title"], p["paragraph_text"])] for q in qs for p in q["paragraphs"] if (p["title"], p["paragraph_text"]) in key}
        else:
            keep = {key[(t, " ".join(s))] for q in qs for t, s in q["paragraphs"] if (t, " ".join(s)) in key}
        keep |= {g for q in qs for g in q["gold"]}
        remap = {old: new for new, old in enumerate(sorted(keep))}
        corpus = [corpus[i] for i in sorted(keep)]
        for q in qs:
            q["gold"] = [remap[g] for g in q["gold"]]
    for q in qs:
        q.pop("paragraphs", None)
    return qs, corpus


class PrecomputedEncoder:
    """e5 with a cache filled in large batches (GPU-friendly); graph calls hit the cache."""

    def __init__(self, enc, batch: int = 256) -> None:
        self.enc, self.batch, self.name = enc, batch, getattr(enc, "name", "enc")

    def precompute(self, texts: list[str]) -> None:
        todo = list(dict.fromkeys(texts))
        for i in range(0, len(todo), self.batch * 16):
            self.enc.encode(todo[i : i + self.batch * 16])

    def encode(self, texts):
        return self.enc.encode(texts)

    def one(self, text):
        return self.enc.one(text)


def build(corpus: list[dict], enc, out: Path, device: str) -> tuple[Graph, dict]:
    t0 = time.time()
    g = Graph(Store(out / "graph.sqlite"), enc, config=Config(), defer_synonyms=True)
    todo = g.s.counts()["episodes"] < len(corpus)
    if todo:  # embeddings in large batches first (GPU-friendly); graph calls then hit the cache
        enc.precompute([s for c in corpus for kind, s in X.sentences(f"{c['title']}\n{c['text']}") if kind == "sentence"])
    t_emb = time.time() - t0
    if todo:  # resumable: append-only, already ingested passages are skipped
        from ctxgraph.v2 import english

        # spaCy over all English sentences at once (same text as the graph will see: scrubbed, marks removed)
        plain = [X._MARK.sub(" ", s) for c in corpus for kind, s in X.sentences(X.scrub(f"{c['title']}\n{c['text']}")[0])
                 if kind == "sentence"]  # fmt: skip
        english.precompute([p for p in plain if T.detect_lang(p) == "en"], n_process=2)
        print(f"  spaCy precomputed {len(plain)} sentences  {time.time() - t0:.0f}s", flush=True)
        for i, c in enumerate(corpus):
            g.add_episode(f"{c['title']}\n{c['text']}", "document", "corpus", eid=f"p{i}", commit=False)
            if i % 500 == 0:
                g.s.commit()
                print(f"  ingested {i}/{len(corpus)}  {time.time() - t0:.0f}s", flush=True)
        g.s.commit()
    t_ing = time.time() - t0 - t_emb
    n_syn = g.build_synonyms(device=device)
    info = {"seconds_embed": round(t_emb, 1), "seconds_ingest": round(t_ing, 1),
            "seconds_synonyms": round(time.time() - t0 - t_emb - t_ing, 1), "synonym_edges": n_syn,
            **g.s.counts()}  # fmt: skip
    return g, info


def passage_of(fid: str) -> int:
    return int(fid.split(":")[1][1:])


def rank_passages(frag_scores: dict[str, float], fallback: list[int] | None = None) -> list[int]:
    best: dict[int, float] = {}
    for f, s in frag_scores.items():
        p = passage_of(f)
        if s > best.get(p, -1e9):
            best[p] = s
    ranked = [p for p, _ in sorted(best.items(), key=lambda x: (-x[1], x[0]))]
    if fallback:
        seen = set(ranked)
        ranked += [p for p in fallback if p not in seen]
    return ranked


def recall(ranked: list[int], gold: list[int], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold) if gold else 0.0


def run_question(g: Graph, q: str, scorer, ids: list[str], mat: np.ndarray) -> dict[str, list[int]]:
    qv = g.enc.one(q)
    dense = mat @ qv
    order = np.argsort(-dense, kind="stable")
    a = rank_passages({ids[i]: float(dense[i]) for i in order[:TOPN_DENSE]})
    rrf: dict[str, float] = defaultdict(float)
    for r, i in enumerate(order[:TOPN_DENSE]):
        rrf[ids[i]] += 1 / (60 + r)
    for r, (f, _) in enumerate(g.s.bm25(T.content_lemmas(q), TOPN_BM25)):
        rrf[f] += 1 / (60 + r)
    b = rank_passages(rrf, a)
    tr = g.retrieve(q)
    ppr = {k: v for k, v in tr.ppr.items() if k.startswith("f:")}
    c = rank_passages(ppr, b)
    out = {"a_dense": a, "b_rrf": b, "c_ppr": c}
    if scorer is not None:
        top = [f for f, _ in sorted(ppr.items(), key=lambda x: (-x[1], x[0]))
               if g.s.q("SELECT kind FROM fragments WHERE id=?", (f,))[0][0] == "sentence"][:RERANK_K]  # fmt: skip
        texts = [g.s.q("SELECT text FROM fragments WHERE id=?", (f,))[0][0] for f in top]
        sc = scorer(q, texts) if top else []
        out["d_rerank"] = rank_passages(dict(zip(top, sc, strict=True)), c)
    return out


def bootstrap(diff: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    a = np.array(diff)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(rows: list[dict], variants: list[str]) -> dict:
    out = {"n": len(rows), "recall": {}, "by_type": {}, "diffs": {}}
    for k in (2, 5):
        for v in variants:
            out["recall"][f"{v}@{k}"] = float(np.mean([r[v][f"r{k}"] for r in rows]))
        for x, y in (("c_ppr", "a_dense"), ("b_rrf", "a_dense"), ("c_ppr", "b_rrf"), ("d_rerank", "a_dense"), ("d_rerank", "c_ppr")):
            if x in variants and y in variants:
                out["diffs"][f"{x}-{y}@{k}"] = bootstrap([r[x][f"r{k}"] - r[y][f"r{k}"] for r in rows])
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        out["by_type"][t] = {"n": len(rs), **{f"{v}@5": float(np.mean([r[v]["r5"] for r in rs])) for v in variants}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["musique", "2wiki", "musique_dev100", "musique_dev200", "musique_dev200b"], required=True)
    ap.add_argument("--out", default="out/bench1")
    ap.add_argument("--data", default=str(ROOT / ".cache" / "bench"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N questions, own small corpus")
    ap.add_argument("--reranker", choices=["fp32", "int8", "none"], default="fp32")
    a = ap.parse_args()
    from ctxgraph.e1b.backends import E5Encoder

    out = Path(a.out) / (a.dataset + (f"_limit{a.limit}" if a.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    qs, corpus = load(a.dataset, Path(a.data), a.limit or None)
    print(f"{a.dataset}: {len(qs)} questions, {len(corpus)} passages", flush=True)
    enc = PrecomputedEncoder(E5Encoder(E5, device=a.device, revision=E5_REV))
    g, binfo = build(corpus, enc, out, a.device)
    print("graph:", binfo, flush=True)
    scorer, variants = None, VARIANTS[:3]
    if a.reranker == "fp32":
        from ctxgraph.e1b.selection import Reranker

        scorer = Reranker(BGE, revision=BGE_REV, device=a.device).score
    elif a.reranker == "int8":
        sys.path.insert(0, str(ROOT / "scripts"))
        from v2_onnx_reranker import OnnxReranker

        scorer = OnnxReranker().score
    if scorer is not None:
        variants = VARIANTS
    ids, mat = g._fids, g._fmat
    t0, rows = time.time(), []
    with (out / "recall.jsonl").open("w", encoding="utf-8") as fh:
        for i, q in enumerate(qs):
            ranks = run_question(g, q["question"], scorer, ids, mat)
            row = {"id": q["id"], "type": q["type"], "gold": q["gold"]}
            for v in variants:
                row[v] = {"r2": recall(ranks[v], q["gold"], 2), "r5": recall(ranks[v], q["gold"], 5), "top5": ranks[v][:5]}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"  q {i}/{len(qs)}  {time.time() - t0:.0f}s", flush=True)
    summ = summarize(rows, variants) | {"build": binfo, "seconds_search": round(time.time() - t0, 1),
                                        "dataset": a.dataset, "limit": a.limit, "reranker": a.reranker,
                                        "models": {"e5": f"{E5}@{E5_REV}", "bge": f"{BGE}@{BGE_REV}"}}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=1), "utf-8")
    print(json.dumps({k: summ[k] for k in ("recall", "diffs")}, indent=1))


if __name__ == "__main__":
    main()
