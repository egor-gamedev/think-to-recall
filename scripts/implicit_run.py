"""Implicit-links benchmark (`reports/prereg_implicit.md`): IMLogic memory-level + ImplexConv-200, six variants.

    python scripts/implicit_run.py --dataset imlogic --split dev --stage graph     # e5 + spaCy graph, A / B / C
    python scripts/implicit_run.py --dataset imlogic --split dev --stage embed     # Qwen3-Embedding-4B (then unloaded)
    python scripts/implicit_run.py --dataset imlogic --split dev --stage stack     # Qwen3-Reranker-4B: D0 / D
    python scripts/implicit_run.py --dataset imlogic --split dev --stage dplus     # LLM sufficiency loop: D+ (internet)
    python scripts/implicit_run.py --dataset imlogic --split dev --stage summary

Stages run one model at a time (the two 4B models never share the GPU) and checkpoint per corpus (IMLogic user /
ImplexConv example): `<out>/<dataset>_<split>/<corpus>/{graph.sqlite, qwen_facts.npy, qwen_queries.npy}` and one
JSON-lines file per variant `<out>/<dataset>_<split>/<variant>.jsonl` ({"id", "corpus", "gold", "rank": top-30}); a
rerun skips corpora already written. Variants:
  A  dense e5 (as bench1b (0))               B  graph only: PPR from the question's concepts (IDF), no dense
  C  A + gate (bench1b (Ш), code 590126d)    D0 EverMemOS-style search stack without LLM memory construction
  D  D0 + gate                               D+ D + their LLM sufficiency / multi-query loop (`evermem_stack.py`)
Gate config: `scripts/bench1b_config.json`, unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bench1_run as B  # noqa: E402
import evermem_stack as S  # noqa: E402

from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2.gate import PassageIndex, PprGateConfig, ppr_gated, q_weights, start_ppr  # noqa: E402
from ctxgraph.v2.graph import Config, Graph, cid  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

TOP = 30
VARIANTS = ["A", "B", "C", "D0", "D", "D+"]


# ── data ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def imlogic(split: str, data: Path) -> list[dict]:
    import imlogic_run as I

    users = I.DEV_USERS if split == "dev" else [u for u in range(20) if u not in I.DEV_USERS]
    lines = (data / "imlogic" / "HaluMem-Medium.jsonl").read_text("utf-8").splitlines()
    out = []
    for u in users:
        rec = json.loads(lines[u])
        ts: dict[str, int] = {}
        for s in rec["sessions"]:
            for m in s["memory_points"]:
                # HaluMem has malformed stamps such as "Aug 12, 2040, 2040, 13:33:11": take month, day, first year, time
                mo, d, y, hms = re.match(r"(\w{3}) (\d+), (\d{4}),(?: \d{4},)* (\d+:\d+:\d+)", m["timestamp"]).groups()
                t = dt.datetime.strptime(f"{mo} {d}, {y}, {hms}", "%b %d, %Y, %H:%M:%S").replace(tzinfo=dt.UTC)
                ts.setdefault(m["memory_content"], int(t.timestamp() * 1000))
        qs, corpus = I.load_user(u, data / "imlogic")
        docs = [S.Doc(text=c["text"], facts=[c["text"]], ts_ms=ts[c["text"]]) for c in corpus]
        out.append({"id": f"user{u:02d}", "corpus": corpus, "docs": docs,
                    "qs": [{"id": q["id"], "question": q["question"], "gold": [q["l"]], "s": q["s"], "type": q["type"]}
                           for q in qs]})  # fmt: skip
    return out


def implexconv(split: str, data: Path) -> list[dict]:
    import implexconv_run as R

    out = []
    for ex in R.load(split, data / "implexconv" / "data"):
        docs = [S.Doc(text=c["text"], facts=[s for k, s in X.sentences("\n" + c["text"]) if k == "sentence"] or [c["text"]])
                for c in ex["corpus"]]  # fmt: skip
        out.append({"id": ex["id"], "corpus": ex["corpus"], "docs": docs,
                    "qs": [q | {"type": ex["kind"]} for q in ex["qs"]]})  # fmt: skip
    return out


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def done_corpora(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(x)["corpus"] for x in path.read_text("utf-8").splitlines() if x.strip()}


def emit(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class GraphCtx:
    """Our graph over one corpus + the gate (bench1b code), for A / B / C / D / D+."""

    def __init__(self, path: Path, enc, gcfg: PprGateConfig) -> None:
        self.g = Graph(Store(path), enc, config=Config())
        self.ids, self.mat = self.g._fids, self.g._fmat
        self.idx = PassageIndex(self.g)
        self.sent_of: dict[int, list[str]] = {}
        for f in self.ids:
            self.sent_of.setdefault(B.passage_of(f), []).append(f)
        self.gcfg = gcfg

    def dense(self, q: str) -> tuple[list[int], dict[str, float]]:
        d = self.mat @ self.g.enc.one(q)
        order = np.argsort(-d, kind="stable")
        lo, hi = float(d.min()), float(d.max())
        norm = {self.ids[j]: (float(d[j]) - lo) / (hi - lo) for j in range(len(self.ids))} if hi > lo else {}
        return B.rank_passages({self.ids[j]: float(d[j]) for j in order}), norm

    def qc(self, q: str) -> set[str]:
        return {cid(c) for c, _, _ in X.concepts_of(q)} & set(self.idx.df)

    def gate(self, base: list[int], q: str, norm: dict[str, float], steps: tuple[str, ...] = (),
             extra_anchors: tuple[int, ...] = ()) -> tuple[list[int], dict]:  # fmt: skip
        """Gate of bench1b; stage 2 adds the plan's steps: their concepts as question concepts, their top
        results as extra anchors (`extra_anchors`). Without steps: exactly the stage-1 gate."""
        cache: dict = {}

        def ppr(anchors, qw):
            key = (tuple(anchors), tuple(sorted(qw.items())))
            if key not in cache:
                cache[key] = start_ppr(self.g, self.sent_of, norm, anchors, qw)
            return cache[key]

        qc = self.qc(q).union(*(self.qc(t) for t in steps))
        out, info = ppr_gated(base, self.idx, qc, ppr, self.gcfg, extra_anchors)
        if not info["changed"] and out != base:
            raise SystemExit("INVARIANT BROKEN")
        return out, info

    def graph_only(self, q: str) -> list[int]:
        qw = q_weights(self.idx, self.qc(q), self.gcfg)
        if not qw:
            return []
        pr = start_ppr(self.g, self.sent_of, {}, [], qw)
        return [p for p, _ in sorted(pr.items(), key=lambda x: (-x[1], x[0]))]


class DenseCtx:
    """Variant A without the graph (ImplexConv main on CPU, Egor 2026-11-08): the same fragments the graph stores
    (sentences of the scrubbed "title + newline + text", `X.pieces` order, ids `f:p{i}:{j}`), the same e5 "passage:" vectors,
    the same max-over-sentences ranking as `GraphCtx.dense`."""

    def __init__(self, d: Path, enc) -> None:
        self.ids = json.loads((d / "e5_fids.json").read_text("utf-8"))
        self.mat = np.load(d / "e5.npy")
        self.enc = enc

    @staticmethod
    def fragments(corpus: list[dict]) -> tuple[list[str], list[str]]:
        ids, texts = [], []
        for i, c in enumerate(corpus):
            clean, _ = X.scrub(f"{c['title']}\n{c['text']}")
            for j, (kind, sent) in enumerate(X.sentences(clean)):
                if kind == "sentence":
                    ids.append(f"f:p{i}:{j}")
                    texts.append(sent)
        return ids, texts

    def dense(self, q: str) -> tuple[list[int], dict]:
        d = self.mat @ self.enc.one(q)
        order = np.argsort(-d, kind="stable")
        return B.rank_passages({self.ids[j]: float(d[j]) for j in order}), {}


def stage_embed_e5(cs, out: Path, device: str) -> None:
    from ctxgraph.e1b.backends import E5Encoder

    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in cs:
        d = out / c["id"]
        if (d / "e5_fids.json").exists():
            continue
        t0 = time.time()
        d.mkdir(parents=True, exist_ok=True)
        ids, texts = DenseCtx.fragments(c["corpus"])
        np.save(d / "e5.npy", np.asarray(e5.encode(texts), dtype=np.float32))
        (d / "e5_fids.json").write_text(json.dumps(ids), "utf-8")  # written last: marks the corpus done
        emit(out / "timing.jsonl", [{"corpus": c["id"], "stage": "embed_e5", "build_s": time.time() - t0, "sentences": len(ids)}])
        print(f"embed_e5 {c['id']}: {len(ids)} sentences {time.time() - t0:.0f}s", flush=True)


def row(c: dict, q: dict, rank: list[int], **extra) -> dict:
    return {"id": q["id"], "corpus": c["id"], "gold": q["gold"], "type": q["type"], "rank": rank[:TOP], **extra}


# ── stages ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def stage_graph(cs, out: Path, device: str, gcfg) -> None:
    from ctxgraph.e1b.backends import E5Encoder

    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    todo = [c for c in cs if c["id"] not in done_corpora(out / "C.jsonl")]
    for c in todo:
        t0 = time.time()
        g, binfo = B.build(c["corpus"], B.PrecomputedEncoder(e5), out / c["id"], device)
        tb = time.time() - t0
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", g.enc, gcfg)
        ra, rb, rc, t = [], [], [], {"A": 0.0, "B": 0.0, "C": 0.0}
        for q in c["qs"]:
            t1 = time.time()
            a, norm = ctx.dense(q["question"])
            t2 = time.time()
            b = ctx.graph_only(q["question"])
            t3 = time.time()
            cc, info = ctx.gate(a, q["question"], norm)
            t4 = time.time()
            t["A"] += t2 - t1
            t["B"] += t3 - t2
            t["C"] += t4 - t3 + t2 - t1
            ra.append(row(c, q, a))
            rb.append(row(c, q, b))
            rc.append(row(c, q, cc, signal=info["signal"], changed=info["changed"]))
        emit(out / "A.jsonl", ra)
        emit(out / "B.jsonl", rb)
        emit(out / "C.jsonl", rc)
        emit(out / "timing.jsonl", [{"corpus": c["id"], "stage": "graph", "build_s": tb, "sentences": binfo["fragments"],
                                     "search_s": t, "n_q": len(c["qs"])}])  # fmt: skip
        print(f"graph {c['id']}: {binfo['fragments']} sentences, build {tb:.0f}s", flush=True)


def stage_embed(cs, out: Path, device: str) -> None:
    todo = [c for c in cs if not (out / c["id"] / "qwen_queries.npy").exists()]
    if not todo:
        return
    emb = S.QwenEmbedder(device=device)
    for c in todo:
        t0 = time.time()
        facts = [f for d in c["docs"] for f in (d.facts or [d.text])]
        (out / c["id"]).mkdir(parents=True, exist_ok=True)
        fe = emb.docs(facts).astype(np.float32)
        np.save(out / c["id"] / "qwen_facts.npy", fe)
        qv = [emb.query(q["question"]) for q in c["qs"]]  # an example may have 0 questions
        np.save(out / c["id"] / "qwen_queries.npy", np.stack(qv).astype(np.float32) if qv else np.zeros((0, fe.shape[1]), np.float32))
        emit(out / "timing.jsonl", [{"corpus": c["id"], "stage": "embed", "build_s": time.time() - t0, "facts": len(facts)}])
        print(f"embed {c['id']}: {len(facts)} facts {time.time() - t0:.0f}s", flush=True)
    emb.close()


def stack_for(c: dict, out: Path, rr, live_embed=None) -> S.Stack:
    """Stack over the cached Qwen embeddings; queries not in the cache (D+ round-2 sub-queries) go to `live_embed`."""
    fe = np.load(out / c["id"] / "qwen_facts.npy")
    qe = np.load(out / c["id"] / "qwen_queries.npy")
    idx = S.StackIndex.build(c["docs"], lambda texts: fe)
    qmap = {q["question"]: qe[i] for i, q in enumerate(c["qs"])}

    def embed_query(q: str) -> np.ndarray:
        if q in qmap:
            return qmap[q]
        if live_embed is None:
            raise KeyError(q)
        return live_embed(q)

    return S.Stack(idx, embed_query, rr.score)


def stage_stack(cs, out: Path, device: str, gcfg) -> None:
    from ctxgraph.e1b.backends import E5Encoder

    todo = [c for c in cs if c["id"] not in done_corpora(out / "D.jsonl")]
    if not todo:
        return
    rr = S.QwenReranker(device=device)
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in todo:
        t0 = time.time()
        st = stack_for(c, out, rr)
        tb = time.time() - t0
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, gcfg)
        r0, rd, t = [], [], {"D0": 0.0, "D": 0.0}
        for q in c["qs"]:
            t1 = time.time()
            d0 = asyncio.run(st.d0(q["question"]))
            t2 = time.time()
            _, norm = ctx.dense(q["question"])
            d, info = ctx.gate(d0, q["question"], norm)
            t3 = time.time()
            t["D0"] += t2 - t1
            t["D"] += t3 - t1
            r0.append(row(c, q, d0))
            rd.append(row(c, q, d, signal=info["signal"], changed=info["changed"]))
        emit(out / "D0.jsonl", r0)
        emit(out / "D.jsonl", rd)
        emit(out / "timing.jsonl", [{"corpus": c["id"], "stage": "stack", "build_s": tb, "clusters": len(st.idx.clusters),
                                     "search_s": t, "n_q": len(c["qs"])}])  # fmt: skip
        print(f"stack {c['id']}: {len(st.idx.clusters)} scenes, {time.time() - t0:.0f}s", flush=True)
    rr.close()


def nvidia_llm() -> S.OpenAICompatLLM:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("NVIDIA_API_KEY")
    return S.OpenAICompatLLM("https://integrate.api.nvidia.com/v1", key, os.environ.get("DPLUS_MODEL", "mistralai/mistral-large-4"))


def stage_dplus(cs, out: Path, device: str, gcfg, limit: int | None) -> None:
    from ctxgraph.e1b.backends import E5Encoder

    todo = [c for c in cs if c["id"] not in done_corpora(out / "D+.jsonl")]
    if not todo:
        return
    llm = nvidia_llm()
    rr = S.QwenReranker(device=device)
    emb = S.QwenEmbedder(device="cpu")  # round-2 sub-queries only; the GPU holds the reranker
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    n = 0
    for c in todo:
        st = stack_for(c, out, rr, emb.query)
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, gcfg)
        rows = []
        for q in c["qs"]:
            if limit is not None and n >= limit:
                break
            _, norm = ctx.dense(q["question"])
            t1 = time.time()
            rank, info = asyncio.run(st.dplus(q["question"], llm, lambda base, q=q, norm=norm: ctx.gate(base, q["question"], norm)[0]))
            rows.append(row(c, q, rank, llm=info, search_s=time.time() - t1))
            n += 1
        emit(out / "D+.jsonl", rows)
        print(f"dplus {c['id']}: {len(rows)} q, calls {llm.calls}, failed {llm.failed}", flush=True)
        if limit is not None and n >= limit:
            break


def rrf(*ranks: list[int], k: int = 60) -> list[int]:
    sc: dict[int, float] = {}
    for r in ranks:
        for i, p in enumerate(r):
            sc[p] = sc.get(p, 0.0) + 1 / (k + i + 1)
    return [p for p, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]


def stage_sleeves(cs, out: Path, device: str, gcfg, plans: Path) -> None:
    """Stage 2 (dev): plan steps -> sleeves. For each base (A = e5 dense, D0 = EverMemOS-style stack):
    S0 = base; SR = RRF(base, step retrievals) — sleeves without the graph; SG = base + gate whose starts also
    include the steps (concepts) and each step's top result (anchor)."""
    from ctxgraph.e1b.backends import E5Encoder

    plan = {json.loads(x)["id"]: json.loads(x)["steps"] for x in plans.read_text("utf-8").splitlines() if x.strip()}
    todo = [c for c in cs if c["id"] not in done_corpora(out / "SG_D0.jsonl")]
    if not todo:
        return
    rr = S.QwenReranker(device=device)
    emb = S.QwenEmbedder(device="cpu")  # step queries only
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in todo:
        st = stack_for(c, out, rr, emb.query)
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, gcfg)
        rows: dict[str, list] = {f"{v}_{b}": [] for v in ("S0", "SR", "SG") for b in ("A", "D0")}
        for q in c["qs"]:
            steps = tuple(plan.get(q["id"], []))
            a, norm = ctx.dense(q["question"])
            d0 = asyncio.run(st.d0(q["question"]))
            for bname, base, retr in (("A", a, lambda t: ctx.dense(t)[0]), ("D0", d0, lambda t: asyncio.run(st.d0(t)))):
                sl = [retr(t) for t in steps]
                rows[f"S0_{bname}"].append(row(c, q, base, steps=len(steps)))
                rows[f"SR_{bname}"].append(row(c, q, rrf(base, *sl) if sl else base))
                g, info = ctx.gate(base, q["question"], norm, steps, tuple(r[0] for r in sl if r))
                rows[f"SG_{bname}"].append(row(c, q, g, signal=info["signal"], changed=info["changed"]))
        for v, rs in rows.items():
            emit(out / f"{v}.jsonl", rs)
        print(f"sleeves {c['id']}: {len(c['qs'])} q", flush=True)
    rr.close()


def stage_rrtrace(cs, out: Path, device: str, gcfg, plans: Path) -> None:
    """Stage 2 dev: reranker-aware gate over D0. Per question, the D0 base, the old gate (question starts) and the
    sleeve gate (+ plan steps and each step's D0 top-1 as starts) are run WITHOUT the reranker check, and their
    displacement traces (PPR of the first `slots` candidates and of the base top-5 tail) are saved together with
    the reranker scores of those passages (against the question; and the max over plan steps, descriptive). The
    check "candidate rr >= displaced rr - delta" is then replayed offline for any delta (`gate.pick`): the trace
    holds every quantity the loop reads, so the replay equals running the gate with the check."""
    from ctxgraph.e1b.backends import E5Encoder

    plan = {json.loads(x)["id"]: json.loads(x)["steps"] for x in plans.read_text("utf-8").splitlines() if x.strip()}
    todo = [c for c in cs if c["id"] not in done_corpora(out / "rrtrace.jsonl")]
    if not todo:
        return
    rr = S.QwenReranker(device=device)
    emb = S.QwenEmbedder(device="cpu")  # step queries only
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in todo:
        t0 = time.time()
        st = stack_for(c, out, rr, emb.query)
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, gcfg)
        rows = []
        for q in c["qs"]:
            steps = tuple(plan.get(q["id"], []))
            d0 = asyncio.run(st.d0(q["question"]))
            _, norm = ctx.dense(q["question"])
            sl = [asyncio.run(st.d0(t)) for t in steps]
            _, old = ctx.gate(d0, q["question"], norm)
            _, sg = ctx.gate(d0, q["question"], norm, steps, tuple(r[0] for r in sl if r))
            need = sorted({int(p) for t in (old, sg) for p in t.get("pr", {})})
            texts = [c["docs"][p].text for p in need]
            rq = dict(zip(need, rr.score(q["question"], texts), strict=True)) if need else {}
            rs = {p: max(v) for p, v in zip(need, zip(*(rr.score(t, texts) for t in steps), strict=True), strict=True)} \
                if need and steps else {}  # fmt: skip
            tr = lambda t: {"order": t.get("order", []), "pr": {str(p): v for p, v in t.get("pr", {}).items()}}  # noqa: E731
            rows.append(row(c, q, d0, steps=len(steps), sr=(rrf(d0, *sl) if sl else d0)[:TOP], old=tr(old), sg=tr(sg),
                            rr_q={str(p): v for p, v in rq.items()}, rr_steps={str(p): v for p, v in rs.items()}))  # fmt: skip
        emit(out / "rrtrace.jsonl", rows)
        print(f"rrtrace {c['id']}: {len(c['qs'])} q, {time.time() - t0:.0f}s", flush=True)
    rr.close()


def stage_sc(cs, out: Path, device: str, plans: Path) -> None:
    """Stage 2 control SC (Egor, 2026-11-08): the same plan steps, but ONE retrieval by their joined text instead of
    one per step; fused with the base by the same RRF as SR -> SC = RRF(base, retr(" ".join(steps))). SR vs SC
    isolates the contribution of splitting the plan into sleeves. No steps -> SC = base (as SR)."""
    plan = {json.loads(x)["id"]: json.loads(x)["steps"] for x in plans.read_text("utf-8").splitlines() if x.strip()}
    todo = [c for c in cs if c["id"] not in done_corpora(out / "SC_D0.jsonl")]
    if not todo:
        return
    from ctxgraph.e1b.backends import E5Encoder

    rr = S.QwenReranker(device=device)
    emb = S.QwenEmbedder(device="cpu")  # step queries only
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in todo:
        st = stack_for(c, out, rr, emb.query)
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, PprGateConfig())
        ra, rd = [], []
        for q in c["qs"]:
            joined = " ".join(plan.get(q["id"], []))
            a, d0 = ctx.dense(q["question"])[0], asyncio.run(st.d0(q["question"]))
            ra.append(row(c, q, rrf(a, ctx.dense(joined)[0]) if joined else a))
            rd.append(row(c, q, rrf(d0, asyncio.run(st.d0(joined))) if joined else d0))
        emit(out / "SC_A.jsonl", ra)
        emit(out / "SC_D0.jsonl", rd)
        print(f"sc {c['id']}: {len(c['qs'])} q", flush=True)
    rr.close()


MIN_IMPLEXCONV = 300


def analysis_ids(cs: list[dict], plan: dict) -> set[str]:
    """Amendment 1 (prereg_stage2): which questions enter the analysis. ImplexConv: the longest prefix of the
    seed-0 question order (`planner.question_order`) whose questions all have a plan, at least MIN_IMPLEXCONV;
    IMLogic: whole users only (every question of the user has a plan). All answered -> all questions."""
    import planner as P

    if cs and cs[0]["id"].startswith("user"):
        return {q["id"] for c in cs if all(q["id"] in plan for q in c["qs"]) for q in c["qs"]}
    order = P.question_order("implexconv", [q for c in cs for q in c["qs"]])
    n = next((i for i, q in enumerate(order) if q["id"] not in plan), len(order))
    if n < min(MIN_IMPLEXCONV, len(order)):
        raise SystemExit(f"ImplexConv: answered prefix {n} < minimum {MIN_IMPLEXCONV} (amendment 1)")
    return {q["id"] for q in order[:n]}


def stage_sleeves_a(cs, out: Path, device: str, gcfg, plans: Path) -> None:
    """Stage 2 main (`prereg_stage2.md`, frozen 2026-11-08): base A only, no Qwen models (runs on CPU).
    S0 = A; SC = RRF(A, A(joined steps)); SR = RRF(A, A(step 1), ..., A(step n)); SG = A + gate with the steps'
    concepts and each step's A top-1 as extra starts. Same definitions as `stage_sleeves` / `stage_sc` (A part)."""
    from ctxgraph.e1b.backends import E5Encoder

    plan = {json.loads(x)["id"]: json.loads(x)["steps"] for x in plans.read_text("utf-8").splitlines() if x.strip()}
    keep = analysis_ids(cs, plan)
    cs = [c | {"qs": [q for q in c["qs"] if q["id"] in keep]} for c in cs]
    cs = [c for c in cs if c["qs"]]
    print(f"sleeves_a: {sum(len(c['qs']) for c in cs)} questions in {len(cs)} corpora (amendment 1 selection)", flush=True)
    todo = [c for c in cs if c["id"] not in done_corpora(out / "SC_A.jsonl")]
    e5 = E5Encoder(B.E5, device=device, revision=B.E5_REV)
    for c in todo:
        t0 = time.time()
        has_graph = (out / c["id"] / "graph.sqlite").exists()  # ImplexConv main: e5 only, SG postponed
        ctx = GraphCtx(out / c["id"] / "graph.sqlite", e5, gcfg) if has_graph else DenseCtx(out / c["id"], e5)
        rows: dict[str, list] = {k: [] for k in ("S0_A", "SC_A", "SR_A", "SG_A")}
        for q in c["qs"]:
            steps = tuple(plan[q["id"]])
            a, norm = ctx.dense(q["question"])
            sl = [ctx.dense(t)[0] for t in steps]
            joined = " ".join(steps)
            rows["S0_A"].append(row(c, q, a, steps=len(steps)))
            rows["SC_A"].append(row(c, q, rrf(a, ctx.dense(joined)[0]) if joined else a))
            rows["SR_A"].append(row(c, q, rrf(a, *sl) if sl else a))
            if has_graph:
                g, info = ctx.gate(a, q["question"], norm, steps, tuple(r[0] for r in sl if r))
                rows["SG_A"].append(row(c, q, g, signal=info["signal"], changed=info["changed"]))
        for v, rs in rows.items():
            if rs:
                emit(out / f"{v}.jsonl", rs)
        emit(out / "timing.jsonl", [{"corpus": c["id"], "stage": "sleeves_a", "search_s": time.time() - t0, "n_q": len(c["qs"])}])
        print(f"sleeves_a {c['id']}: {len(c['qs'])} q {time.time() - t0:.0f}s", flush=True)


def stage_summary(out: Path) -> dict:
    import bench1b_run as BB

    res: dict = {"recall": {}, "tests": {}, "timing": {}}
    rows = {v: {json.loads(x)["id"]: json.loads(x) for x in (out / f"{v}.jsonl").read_text("utf-8").splitlines() if x}
            for v in VARIANTS + [f"{t}_{b}" for t in ("S0", "SR", "SG") for b in ("A", "D0")]
            if (out / f"{v}.jsonl").exists()}  # fmt: skip
    common = sorted(set.intersection(*(set(r) for r in rows.values()))) if rows else []
    for v, rs in rows.items():
        for k in (1, 3, 5, 10, 30):
            res["recall"][f"{v}@{k}"] = float(np.mean([B.recall(rs[i]["rank"], rs[i]["gold"], k) for i in common]))
    pairs = {"S1": ("C", "A"), "S2": ("D", "D0")} | {f"{t}_{b}": (x, y) for b in ("A", "D0") for t, x, y in
             (("SG-S0", f"SG_{b}", f"S0_{b}"), ("SR-S0", f"SR_{b}", f"S0_{b}"), ("SG-SR", f"SG_{b}", f"SR_{b}"))}
    for name, (x, y) in pairs.items():
        if x in rows and y in rows:
            d = [B.recall(rows[x][i]["rank"], rows[x][i]["gold"], 5) - B.recall(rows[y][i]["rank"], rows[y][i]["gold"], 5) for i in common]
            res["tests"][name] = {"compare": f"{x}-{y}", "diff": B.bootstrap(d), "p": BB.perm_p(d),
                                  "p_reverse": BB.perm_p([-v for v in d]), "wins": sum(v > 0 for v in d),
                                  "losses": sum(v < 0 for v in d), "n": len(d)}  # fmt: skip
    res["n"] = len(common)
    if (out / "timing.jsonl").exists():
        res["timing"] = [json.loads(x) for x in (out / "timing.jsonl").read_text("utf-8").splitlines() if x]
    (out / "summary.json").write_text(json.dumps(res, indent=1), "utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imlogic", "implexconv"], required=True)
    ap.add_argument("--split", choices=["dev", "main"], required=True)
    ap.add_argument("--stage", choices=["graph", "embed", "stack", "dplus", "sleeves", "rrtrace", "sc", "sleeves_a", "embed_e5", "summary"], required=True)
    ap.add_argument("--plans", default=None, help="sleeves only: plans .jsonl (scripts/planner.py)")
    ap.add_argument("--out", default="out/implicit")
    ap.add_argument("--data", default=str(ROOT / ".cache" / "bench"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="dplus only: stop after N questions (smoke test)")
    a = ap.parse_args()
    out = Path(a.out) / f"{a.dataset}_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "summary":
        print(json.dumps(stage_summary(out)["tests"], indent=1))
        return
    gcfg = PprGateConfig(**json.loads((ROOT / "scripts" / "bench1b_config.json").read_text("utf-8"))["gate"])
    cs = [c for c in (imlogic if a.dataset == "imlogic" else implexconv)(a.split, Path(a.data)) if c["qs"]]  # 0-question examples add nothing
    print(f"{a.dataset} {a.split}: {len(cs)} corpora, {sum(len(c['qs']) for c in cs)} questions", flush=True)
    {"graph": lambda: stage_graph(cs, out, a.device, gcfg), "embed": lambda: stage_embed(cs, out, a.device),
     "stack": lambda: stage_stack(cs, out, a.device, gcfg),
     "dplus": lambda: stage_dplus(cs, out, a.device, gcfg, a.limit),
     "sleeves": lambda: stage_sleeves(cs, out, a.device, gcfg, Path(a.plans)),
     "rrtrace": lambda: stage_rrtrace(cs, out, a.device, gcfg, Path(a.plans)),
     "sc": lambda: stage_sc(cs, out, a.device, Path(a.plans)),
     "sleeves_a": lambda: stage_sleeves_a(cs, out, a.device, gcfg, Path(a.plans)),
     "embed_e5": lambda: stage_embed_e5(cs, out, a.device)}[a.stage]()  # fmt: skip


if __name__ == "__main__":
    main()
