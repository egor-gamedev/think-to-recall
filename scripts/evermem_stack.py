"""Search stack in the style of EverMemOS — variants D0 / D / D+ of `reports/prereg_implicit.md`.

Built from EverMemOS's own open algorithm library `everalgo` (Apache-2.0: everalgo-rank 0.3.0, everalgo-core 0.2.0,
everalgo-clustering 0.2.1, the versions EverOS @ 648b34b pins) and orchestrated 1:1 like their benchmark search stage
(EverAlgo @ 7398133, `benchmarks/common/stages/search.py::_build_retrieval_closures`, `_run_agentic_retrieval`), with
the defaults of `benchmarks/common/config.py`:

  dense  = MaxSim over fact embeddings (Qwen3-Embedding-4B, 1024 dims)      -> parent episode
  sparse = MaxSim over fact BM25 (BM25Okapi, their NLTK tokenize + Porter)  -> parent episode
  hybrid_full    = RRF(dense 50, sparse 50, k = 40)
  cluster_scoped = members of the top-10 MemScenes (clusters by cosine >= 0.70 within 7 days, `cluster_by_geometry`)
  round 1        = cluster_scoped -> Qwen3-Reranker-4B (their instruction) -> top 10
  D+ only        = `aagentic_retrieve`: LLM sufficiency check on the round-1 top 10; if insufficient, multi-query
                   (3 queries) round 2 over hybrid_full, cap 40, final rerank, top 10.

No LLM memory construction: the "episodes" are the dataset's own units (IMLogic memory records, ImplexConv sessions)
and the "atomic facts" are the record itself / the session's sentences. D0 = round 1 without the LLM; its top 10 is
followed by the rest of hybrid_full so Recall@k is defined beyond 10. D = D0 + our gate. D+ = D + the sufficiency loop
(the gate wraps their rerank_fn). Deviations from their setup are listed in the prereg: temperature 0 for every LLM
call (theirs: 0 / 0.4), no time window when a dataset has no timestamps, an unparsable LLM reply = no second round.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

QWEN_EMB, QWEN_EMB_REV = "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b"
QWEN_RR, QWEN_RR_REV = "Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
# benchmarks/common/config.py (EverAlgo @ 7398133)
CFG = {"hybrid_emb_candidates": 50, "hybrid_bm25_candidates": 50, "hybrid_rrf_k": 40, "response_top_k": 10,
       "round1_rerank_top_n": 10, "cluster_similarity_threshold": 0.70, "cluster_max_time_gap_days": 7.0,
       "cluster_top_k": 10, "multi_query_num": 3, "round2_cap": 40, "round1_top_n": 50, "embedding_dimensions": 1024,
       "reranker_instruction": "Determine if the passage contains specific facts, entities (names, dates, locations), "
                               "or details that directly answer the question."}  # fmt: skip
DAY_MS = 86_400_000


@dataclass
class Doc:
    """An "episode": `text` goes to the reranker, `facts` are its children for MaxSim; `ts_ms` orders clustering."""

    text: str
    facts: list[str]
    ts_ms: int | None = None


def tokenize(text: str) -> list[str]:
    """EverAlgo `benchmarks/common/stages/_tokenize.py::tokenize`: lower -> word_tokenize -> alpha, len >= 2, not a
    stopword -> Porter stem."""
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    global _STEM, _STOP
    if _STEM is None:
        _STEM, _STOP = PorterStemmer(), set(stopwords.words("english"))
    if not text:
        return []
    return [_STEM.stem(t) for t in word_tokenize(text.lower()) if t.isalpha() and len(t) >= 2 and t not in _STOP]


_STEM = _STOP = None


class QwenEmbedder:
    """Qwen3-Embedding-4B via sentence-transformers, Matryoshka-truncated to 1024 dims, normalised; queries use the
    model's "query" prompt (model card)."""

    def __init__(self, device: str = "cuda", batch: int = 16) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        kw = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        self.m = SentenceTransformer(QWEN_EMB, revision=QWEN_EMB_REV, device=device,
                                     truncate_dim=CFG["embedding_dimensions"], model_kwargs=kw)  # fmt: skip
        self.m.max_seq_length = 8192
        self.batch = batch

    def docs(self, texts: list[str], batch: int | None = None) -> np.ndarray:
        """On CUDA OOM (a batch with very long facts) the slice is encoded again in halves: same model, same
        truncation, only the batch is smaller. The retry runs AFTER the except block: inside it the exception's
        traceback still holds the failed attempt's activations, and every retry would have less memory."""
        import gc

        import torch

        batch = batch or self.batch
        try:
            return self.m.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True)
        except torch.OutOfMemoryError:
            if len(texts) == 1 and batch == 1:
                raise
        gc.collect()
        torch.cuda.empty_cache()
        if batch > 1:
            return self.docs(texts, max(1, batch // 2))
        h = len(texts) // 2
        return np.concatenate([self.docs(texts[:h], 1), self.docs(texts[h:], 1)])

    def query(self, text: str) -> np.ndarray:
        return self.m.encode([text], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)[0]

    def close(self) -> None:
        import gc

        import torch

        del self.m
        gc.collect()
        torch.cuda.empty_cache()


class QwenReranker:
    """Qwen3-Reranker-4B, scoring as in its model card: P("yes") of the last token for
    "<Instruct>: … <Query>: … <Document>: …" inside the official system / assistant template."""

    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
              "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, device: str = "cuda", batch: int = 8, max_len: int = 8192) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.batch, self.max_len = torch, device, batch, max_len
        self.tok = AutoTokenizer.from_pretrained(QWEN_RR, revision=QWEN_RR_REV, padding_side="left")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.m = AutoModelForCausalLM.from_pretrained(QWEN_RR, revision=QWEN_RR_REV, torch_dtype=dtype).to(device).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, query: str, docs: list[str], instruction: str = CFG["reranker_instruction"],
              token_budget: int = 12000) -> list[float]:  # fmt: skip
        """Batches by a token budget (length-sorted), not a fixed count: long sessions would not fit the GPU."""
        room = self.max_len - len(self.pre) - len(self.suf)
        pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        ids = self.tok(pairs, padding=False, truncation="longest_first", return_attention_mask=False, max_length=room)["input_ids"]
        ids = [self.pre + x + self.suf for x in ids]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [0.0] * len(ids)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(ids[order[j]]) * (j - i + 1) <= token_budget and j - i < self.batch * 4:
                j += 1
            chunk = order[i:j]
            enc = self.tok.pad({"input_ids": [ids[k] for k in chunk]}, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                last = self.m(**enc, logits_to_keep=1).logits[:, -1, :]  # last position only: full logits = GBs
                two = self.torch.stack([last[:, self.no], last[:, self.yes]], dim=1).float()
                for k, s in zip(chunk, self.torch.softmax(two, dim=1)[:, 1].tolist(), strict=True):
                    out[k] = s
            i = j
        return out

    def close(self) -> None:
        import gc

        del self.m
        gc.collect()
        self.torch.cuda.empty_cache()


@dataclass
class StackIndex:
    """Per-corpus index (one IMLogic user / one ImplexConv example): fact embeddings, BM25, MemScenes."""

    docs: list[Doc]
    fact_emb: np.ndarray  # (n_facts, dim), normalised
    fact_doc: list[int]  # fact row -> doc index
    bm25: object
    clusters: list = field(default_factory=list)

    @classmethod
    def build(cls, docs: list[Doc], embed_docs: Callable[[list[str]], np.ndarray]) -> StackIndex:
        from everalgo.clustering import Cluster, cluster_by_geometry
        from rank_bm25 import BM25Okapi

        facts, fact_doc = [], []
        for i, d in enumerate(docs):
            for f in d.facts or [d.text]:
                facts.append(f)
                fact_doc.append(i)
        fact_emb = embed_docs(facts)
        bm25 = BM25Okapi([tokenize(f) or ["_"] for f in facts])
        # MemScenes: online geometric clustering of episodes in time order (their Stage 1), centroid = mean of the
        # episode's fact embeddings; no timestamps -> all ts 0 (no time is invented), the window has no effect
        order = sorted(range(len(docs)), key=lambda i: (docs[i].ts_ms or 0, i))
        # without timestamps every episode gets ts 0, so the 7-day window never excludes anything
        window = CFG["cluster_max_time_gap_days"]
        rows = np.asarray(fact_doc)
        clusters: list = []
        for i in order:
            v = fact_emb[rows == i].mean(0)
            v = v / (np.linalg.norm(v) or 1.0)
            new = Cluster(centroid=v, last_ts=docs[i].ts_ms or 0, members=[str(i)], preview=[docs[i].text[:80]])
            merged = cluster_by_geometry(new, clusters, threshold=CFG["cluster_similarity_threshold"],
                                         time_window_days=window)  # fmt: skip
            if merged is None:
                clusters.append(new.model_copy(update={"id": f"c{len(clusters)}"}))
            else:
                clusters = [merged if c.id == merged.id else c for c in clusters]
        return cls(docs, fact_emb, fact_doc, bm25, clusters)


class Stack:
    """Retrieval closures over one StackIndex, as in EverAlgo's benchmark (`_build_retrieval_closures`)."""

    def __init__(self, idx: StackIndex, embed_query: Callable[[str], np.ndarray], rerank: Callable[[str, list[str]], list[float]]) -> None:  # fmt: skip
        from everalgo.types import Candidate

        self.idx, self.embed_query, self.rerank_scores, self.C = idx, embed_query, rerank, Candidate
        self.all_docs = [Candidate(id=str(i), score=0.0, metadata={"doc": d.text}) for i, d in enumerate(idx.docs)]

    async def _parents(self, ids: list[str]):
        return [self.all_docs[int(i)] for i in ids]

    async def _dense(self, q: str, k: int):
        from everalgo.rank import amaxsim_retrieve

        qv = self.embed_query(q)
        sims = self.idx.fact_emb @ qv

        async def child(_q, _k):
            return [self.C(id=f"f{j}", score=float(s), source="vector", metadata={"parent_id": str(self.idx.fact_doc[j])})
                    for j, s in enumerate(sims)]  # fmt: skip

        return await amaxsim_retrieve(q, child_retrieve=child, parent_fetch=self._parents, top_n=k,
                                      child_candidates=len(sims))  # fmt: skip

    async def _sparse(self, q: str, k: int):
        from everalgo.rank import amaxsim_retrieve

        toks = tokenize(q)

        async def child(_q, _k):
            if not toks:
                return []
            sc = self.idx.bm25.get_scores(toks)
            return [self.C(id=f"f{j}", score=float(s), source="keyword", metadata={"parent_id": str(self.idx.fact_doc[j])})
                    for j, s in enumerate(sc)]  # fmt: skip

        return await amaxsim_retrieve(q, child_retrieve=child, parent_fetch=self._parents, top_n=k,
                                      child_candidates=len(self.idx.fact_doc))  # fmt: skip

    async def hybrid_full(self, q: str, k: int | None):
        from everalgo.rank import ahybrid_retrieve

        return await ahybrid_retrieve(q, dense_retrieve=self._dense, sparse_retrieve=self._sparse,
                                      top_n=k if k is not None else len(self.all_docs),
                                      dense_candidates=CFG["hybrid_emb_candidates"],
                                      sparse_candidates=CFG["hybrid_bm25_candidates"], rrf_k=CFG["hybrid_rrf_k"])  # fmt: skip

    async def cluster_scoped(self, q: str, _k: int):
        from everalgo.rank import acluster_retrieve

        return await acluster_retrieve(q, base_retrieve=self.hybrid_full, base_candidates=None,
                                       clusters=self.idx.clusters, all_docs=self.all_docs,
                                       cluster_top_k=CFG["cluster_top_k"])  # fmt: skip

    async def rerank_fn(self, q: str, cands: list):
        if not cands:
            return []
        sc = self.rerank_scores(q, [c.metadata["doc"] for c in cands])
        out = [self.C(id=c.id, score=float(s), source=c.source, metadata=c.metadata) for c, s in zip(cands, sc, strict=True)]
        return sorted(out, key=lambda c: -c.score)

    def gated_rerank_fn(self, gate: Callable[[list[int]], list[int]]):
        """rerank_fn followed by our gate over the reranked order (variant D+): the gate may add or raise graph
        candidates; they enter as Candidates of their own episodes."""

        async def fn(q: str, cands: list):
            ranked = await self.rerank_fn(q, cands)
            order = gate([int(c.id) for c in ranked])
            by_id = {c.id: c for c in ranked}
            return [by_id.get(str(i)) or self.all_docs[i] for i in order]

        return fn

    async def d0(self, q: str) -> list[int]:
        """Round 1 of their agentic path without the LLM: cluster-scoped -> rerank -> top 10, then hybrid_full."""
        r1 = await self.cluster_scoped(q, CFG["round1_top_n"])
        top = (await self.rerank_fn(q, r1))[: CFG["round1_rerank_top_n"]] if r1 else []
        head = [int(c.id) for c in top]
        rest = [int(c.id) for c in await self.hybrid_full(q, None) if int(c.id) not in set(head)]
        return head + rest + [i for i in range(len(self.all_docs)) if i not in set(head) | set(rest)]

    async def dplus(self, q: str, llm, gate: Callable[[list[int]], list[int]] | None) -> tuple[list[int], dict]:
        """`aagentic_retrieve` exactly as their benchmark calls it; rerank_fn wrapped by the gate (D+)."""
        from everalgo.llm.errors import LLMError
        from everalgo.rank import aagentic_retrieve

        fn = self.gated_rerank_fn(gate) if gate else self.rerank_fn
        try:
            res, dec = await aagentic_retrieve(
                q, base_retrieve=self.cluster_scoped, round2_retrieve=self.hybrid_full, round2_cap=CFG["round2_cap"],
                rerank_fn=fn, llm=llm, top_n=CFG["response_top_k"], round1_top_n=CFG["round1_top_n"],
                round1_rerank_top_n=CFG["round1_rerank_top_n"], refinement_strategy="multi_query",
                multi_query_count=CFG["multi_query_num"], rrf_k=CFG["hybrid_rrf_k"])  # fmt: skip
            info = {"ok": True, "sufficient": dec.is_sufficient, "multi_round": dec.is_multi_round,
                    "queries": list(dec.refined_queries)}  # fmt: skip
        except (ValueError, KeyError, TypeError, LLMError) as e:  # unparsable / failed LLM reply -> no 2nd round (prereg)
            r1 = await self.cluster_scoped(q, CFG["round1_top_n"])
            res = (await fn(q, r1))[: CFG["round1_rerank_top_n"]] if r1 else []
            info = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
        head = [int(c.id) for c in res]
        rest = [int(c.id) for c in await self.hybrid_full(q, None) if int(c.id) not in set(head)]
        return head + rest, info


class OpenAICompatLLM:
    """everalgo `LLMClient` over an OpenAI-compatible endpoint (NVIDIA API). Temperature is forced to 0 for every call
    (prereg); every call and its outcome is counted."""

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 800) -> None:
        from openai import AsyncOpenAI

        self.client, self.model, self.max_tokens = AsyncOpenAI(base_url=base_url, api_key=api_key), model, max_tokens
        self.calls = 0
        self.failed = 0

    async def chat(self, messages, *, model=None, temperature=None, max_tokens=None, response_format=None, **extra):
        from everalgo.llm.errors import LLMError
        from everalgo.llm.types import ChatResponse

        self.calls += 1
        try:
            r = await self.client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=max_tokens or self.max_tokens,
                messages=[{"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)}
                          for m in messages])  # fmt: skip
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            raise LLMError(str(e)[:200]) from e
        return ChatResponse(content=r.choices[0].message.content or "", model=self.model)


def run(coro):
    return asyncio.run(coro)
