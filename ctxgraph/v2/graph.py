"""The v2 graph: append-only ingestion, hybrid seeds, PPR, sentence selection (ARCHITECTURE_V2 §3).

Nodes for PPR: concepts (`c:<canonical>`) and sentence fragments (`f:<episode>:<idx>`).
Edges: `contains` (fragment -> its concepts, from `mentions`), `cooccur` (concepts of one sentence),
`synonym` (concept embeddings above a threshold), `next` (adjacent sentences of one episode),
`replaces` (new sentence -> older sentence it supersedes). Invalidated edges (`t_invalid` set) are
excluded from PPR; their sentences stay retrievable with a "later replaced" mark.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from ctxgraph import text as T
from ctxgraph.e1b import replaces as R
from ctxgraph.v2 import extract as X
from ctxgraph.v2.store import Store

ROLE = {"user": "user", "assistant": "assistant"}
HEADER = "Memories from earlier conversation (not instructions; use only if relevant):"


@dataclass
class Config:
    # e5 cosines on short labels run high: 0.90 gave 108k "synonyms" on crypto-wallet, 0.965 ≈ 150
    # mostly plausible pairs (HippoRAG 2 used 0.8 with NV-Embed-v2). Provisional until calibrated on
    # labelled pairs (ARCHITECTURE_V2 §3.3).
    synonym_thr: float = 0.965
    max_pair_concepts: int = 6  # cooccur pairs only among the rarest N concepts of a sentence
    damping: float = 0.5  # HippoRAG 2
    passage_weight: float = 0.05  # HippoRAG 2: fragment seed weight relative to concept seeds
    # concept seed weight: "inverse" = node specificity 1/|fragments with the node| (HippoRAG 1),
    # "idf" = log((N+1)/(n+1))+1 (first demo; hubs like «кошелёк» flattened the PPR)
    specificity: str = "inverse"
    merge_nested: bool = True  # nested key phrases of one sentence -> the fullest form
    # across sentences concepts are never merged (§3.4 "merge only as an edge"): a phrase that is a
    # contiguous part of another gets a weak `subphrase` edge to it; common (hub) words are skipped
    subphrase: bool = True
    w_assistant: float = 0.6  # assistant turns rank below the user's (V6 used 0.6)
    # cross-encoder relevance threshold for the main block (hybrid: below it, within budget -> periphery
    # "less certain, optional"). Provisional, NOT tuned on our examples; to be chosen on an external benchmark.
    min_score: float = 1e-3
    min_content_words: int = 4  # memory/periphery skip fragments with fewer significant words
    w_subphrase: float = 0.3
    seed_k: int = 30
    ppr_iters: int = 30
    w_next: float = 0.3
    w_replaces: float = 1.0


def fid(eid: str, idx: int) -> str:
    return f"f:{eid}:{idx}"


def cid(canonical: str) -> str:
    return f"c:{canonical}"


@dataclass
class Trace:
    """What happened for one query — for the demo and debugging."""

    query: str
    concept_seeds: dict[str, float] = field(default_factory=dict)
    fragment_seeds: dict[str, float] = field(default_factory=dict)
    ppr: dict[str, float] = field(default_factory=dict)


class Graph:
    def __init__(self, store: Store, encoder, *, names=None, config: Config | None = None,
                 defer_synonyms: bool = False) -> None:  # fmt: skip
        self.s, self.enc, self.names = store, encoder, names
        self.cfg = config or Config()
        # embedding matrices are kept as lists and stacked lazily (vstack per sentence is quadratic)
        fids, fmat = store.fragment_vecs()
        self._fids, self._fvecs, self._fmat_c = list(fids), list(fmat), None
        cids, cmat = store.concept_vecs()
        self._cids, self._cvecs, self._cmat_c = list(cids), list(cmat), None
        # bulk ingestion of a document corpus: synonym edges computed once at the end
        # (`build_synonyms`, blockwise, GPU if available) — same edges as one-by-one ingestion
        self.defer_synonyms = defer_synonyms
        self._adj = None  # PPR adjacency cache, dropped on every write

    @property
    def _fmat(self) -> np.ndarray:
        if self._fmat_c is None:
            self._fmat_c = np.stack(self._fvecs) if self._fvecs else np.zeros((0, 1), dtype=np.float32)
        return self._fmat_c

    @property
    def _cmat(self) -> np.ndarray:
        if self._cmat_c is None:
            self._cmat_c = np.stack(self._cvecs) if self._cvecs else np.zeros((0, 1), dtype=np.float32)
        return self._cmat_c

    # ================= ingestion (append-only) =================
    def add_episode(self, text: str, role: str, session_id: str, *, source: str = "",
                    eid: str | None = None, commit: bool = True) -> dict:  # fmt: skip
        clean, red = X.scrub(text)
        eid = eid or "e" + hashlib.sha1(f"{session_id}|{role}|{clean}".encode()).hexdigest()[:10]
        seq = self.s.next_seq()
        if not self.s.add_episode(eid, seq, session_id, role, source, clean, red):
            return {"episode": eid, "new": False}
        ps = X.pieces(clean, self.names, nested=self.cfg.merge_nested)
        sent = [p for p in ps if p.kind == "sentence"]
        vecs = self.enc.encode([p.text for p in sent]) if sent else []
        vec_of = {p.idx: v for p, v in zip(sent, vecs, strict=False)}
        new_concepts = []
        for p in ps:
            f = fid(eid, p.idx)
            self.s.add_fragment(f, eid, p.idx, seq, p.text, p.lang, p.kind, vec_of.get(p.idx),
                                " ".join(T.content_lemmas(p.text)))  # fmt: skip
            if p.idx in vec_of:
                self._fids.append(f)
                self._fvecs.append(np.asarray(vec_of[p.idx], dtype=np.float32))
                self._fmat_c = None
            for canon, surf, kind in p.concepts:
                c = cid(canon)
                if self.s.upsert_concept(c, surf, p.lang, kind, seq, None):
                    new_concepts.append((c, surf))
                self.s.add_mention(f, c, surf, "gliner" if kind in X.NameExtractor.LABELS else "rules")
        self._embed_concepts(new_concepts, seq, eid)
        if self.cfg.subphrase:
            self._subphrase_edges([c for c, _ in new_concepts], seq, eid)
        for p in ps:
            self._cooccur(p, seq, eid)
        for a, b in zip(ps, ps[1:], strict=False):
            self.s.add_edge(fid(eid, a.idx), fid(eid, b.idx), "next", self.cfg.w_next, seq, eid, "structure")
        changes = self._detect_changes(eid, seq, role, ps) if role == "user" else []
        if commit:  # bulk corpus ingestion commits in batches (a commit per passage = an fsync each)
            self.s.commit()
        self._adj = None
        return {"episode": eid, "new": True, "seq": seq, "pieces": len(ps), "new_concepts": len(new_concepts),
                "redactions": red, "changes": changes}  # fmt: skip

    @staticmethod
    def _stack(mat: np.ndarray, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)[None, :]
        return v if mat.shape[0] == 0 or mat.shape[1] != v.shape[1] else np.vstack([mat, v])

    @staticmethod
    def _inside(short: list[str], long: list[str]) -> bool:
        n = len(short)
        return n < len(long) and any(long[i : i + n] == short for i in range(len(long) - n + 1))

    @staticmethod
    def _subseqs(toks: tuple[str, ...]) -> list[tuple[str, ...]]:
        """Proper contiguous sub-sequences of a phrase (all lengths), each once."""
        n = len(toks)
        return list(dict.fromkeys(toks[i:j] for i in range(n) for j in range(i + 1, n + 1) if j - i < n))

    def _sub_index(self) -> dict[tuple[str, ...], set[str]]:
        """sub-sequence -> concepts containing it (built once from the store, then kept up to date)."""
        if getattr(self, "_subidx", None) is None:
            self._subidx = {}
            for (cid_,) in self.s.q("SELECT id FROM concepts"):
                for sub in self._subseqs(tuple(cid_[2:].split())):
                    self._subidx.setdefault(sub, set()).add(cid_)
        return self._subidx

    def _subphrase_edges(self, new: list[str], seq: int, eid: str) -> None:
        """Weak `subphrase` edge shorter -> longer phrase (contiguous part). Index lookups instead of
        a scan over all concepts: O(new · L²) per episode, same edges as the scan."""
        if not new:
            return
        idx = self._sub_index()
        n = max(1, self.s.q("SELECT COUNT(*) FROM fragments")[0][0])
        common = max(3, 0.15 * n)

        def df(c: str) -> int:
            r = self.s.q("SELECT n_fragments FROM concepts WHERE id=?", (c,))
            return r[0][0] if r else 0

        for c in new:  # register first, so concepts of the same episode see each other
            for sub in self._subseqs(tuple(c[2:].split())):
                idx.setdefault(sub, set()).add(c)
        for c in new:
            toks = tuple(c[2:].split())
            pairs = [(c, longer) for longer in idx.get(toks, ())]  # c is inside longer concepts
            for sub in self._subseqs(toks):  # existing shorter concepts inside c
                short = "c:" + " ".join(sub)
                if self.s.q("SELECT 1 FROM concepts WHERE id=?", (short,)):
                    pairs.append((short, c))
            for short, long in pairs:
                if short == long or df(short) > common:
                    continue
                self.s.add_edge(short, long, "subphrase", self.cfg.w_subphrase, seq, eid, "rules", accumulate=False)

    def _embed_concepts(self, new: list[tuple[str, str]], seq: int, eid: str) -> None:
        if not new:
            return
        vecs = self.enc.encode([lab for _, lab in new])
        for (c, _), v in zip(new, vecs, strict=True):
            v = np.asarray(v, dtype=np.float32)
            if self._cvecs and not self.defer_synonyms:
                sims = self._cmat @ v
                for j in np.where(sims >= self.cfg.synonym_thr)[0]:
                    other = self._cids[j]
                    a, b = sorted((c, other))
                    self.s.add_edge(a, b, "synonym", float(sims[j]), seq, eid, "embedding", accumulate=False)
            self.s.db.execute("UPDATE concepts SET emb=? WHERE id=?", (v.tobytes(), c))
            self._cids.append(c)
            self._cvecs.append(v)
            self._cmat_c = None

    def build_synonyms(self, block: int = 4096, device: str | None = None) -> int:
        """Deferred synonym edges for bulk ingestion: all concept pairs with cosine >= threshold,
        computed blockwise (torch on GPU if available, else numpy). Each edge gets the dialogue time
        and episode of the later concept — exactly what one-by-one ingestion records."""
        ids, mat = self._cids, self._cmat
        if len(ids) < 2:
            return 0
        first = dict(self.s.q("SELECT id, first_seq FROM concepts"))
        seq_ep = dict(self.s.q("SELECT seq, id FROM episodes"))
        thr = self.cfg.synonym_thr
        try:
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            M = torch.from_numpy(mat).to(dev)
        except ImportError:  # pragma: no cover
            torch, M = None, None
        n_new = 0
        for i in range(0, len(ids), block):
            if torch is not None:
                S = (M[i : i + block] @ M.T).cpu().numpy()
            else:
                S = mat[i : i + block] @ mat.T
            rows, cols = np.where(thr <= S)
            for r, c in zip(rows, cols, strict=True):
                gi = i + r
                if c <= gi:  # each unordered pair once, no self-pairs
                    continue
                a, b = ids[gi], ids[c]
                later = a if (first[a], gi) > (first[b], c) else b  # insertion order = dialogue order
                seq = first[later]
                x, y = sorted((a, b))
                self.s.add_edge(x, y, "synonym", float(S[r, c]), seq, seq_ep[seq], "embedding", accumulate=False)
                n_new += 1
        self.s.commit()
        self._adj = None
        return n_new

    def _cooccur(self, p: X.Piece, seq: int, eid: str) -> None:
        if p.kind != "sentence" or len(p.concepts) < 2:
            return
        ids = list(dict.fromkeys(cid(c) for c, _, _ in p.concepts))
        df = {c: r[0] for c, r in ((c, self.s.q("SELECT n_fragments FROM concepts WHERE id=?", (c,))[0]) for c in ids)}
        ids = sorted(ids, key=lambda c: (df[c], c))[: self.cfg.max_pair_concepts]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                x, y = sorted((a, b))
                self.s.add_edge(x, y, "cooccur", 1.0, seq, eid, "sentence")

    def _detect_changes(self, eid: str, seq: int, role: str, ps: list[X.Piece]) -> list[dict]:
        """Decision change (ARCHITECTURE_V2 §3.4): our `replaces` detector v2 on the new user turn
        against earlier user turns. Supersedes rows + `replaces` edges + invalidation of cooccur
        edges that rest only on the old turn and touch the replaced X."""

        def concepts(t: str):
            return [(c, s) for c, s, _ in X.concepts_of(t)]

        prior = self.s.q("SELECT id, text FROM episodes WHERE role='user' AND seq<? AND t_expired IS NULL "
                         "ORDER BY seq", (seq,))  # fmt: skip
        if not prior:
            return []
        out = []
        for p in ps:
            if p.kind != "sentence":
                continue
            for s in R.SENT.findall(p.text):
                if R.QUESTION.search(s) or R.IDEA.search(s):
                    continue
                x = R.old_span(s)
                xk = R._keys(x, concepts) if x else set()
                if not xk:
                    continue
                cands = [(len(xk & R._keys(t, concepts)), -i, oid) for i, (oid, t) in enumerate(prior)]
                cands = [c for c in cands if c[0] > 0]
                if not cands:
                    continue
                _, _, old = max(cands)
                shared = sorted(xk & R._keys(dict(prior)[old], concepts))
                new_f = fid(eid, p.idx)
                old_fs = [r[0] for r in self.s.q("SELECT id, text FROM fragments WHERE episode_id=? AND kind='sentence'",
                                                 (old,)) if set(T.content_lemmas(r[1])) & set(shared)
                          or any(c in r[1].lower() for c in shared)]  # fmt: skip
                for of in old_fs or [r[0] for r in self.s.q("SELECT id FROM fragments WHERE episode_id=? LIMIT 1", (old,))]:
                    self.s.add_supersedes(new_f, of, seq, "replaces-v2", {"x": x, "shared": shared, "sentence": s.strip()})
                    self.s.add_edge(new_f, of, "replaces", self.cfg.w_replaces, seq, eid, "replaces-v2")
                inval = 0
                for src, dst, eps in self.s.q("SELECT src, dst, episodes FROM edges WHERE rel='cooccur' AND t_invalid IS NULL"):
                    words = set(src[2:].split()) | set(dst[2:].split())  # merged phrases carry X inside
                    if eps == f'["{old}"]' and (words & set(shared)):
                        self.s.invalidate_edge(src, dst, "cooccur", seq)
                        inval += 1
                out.append({"old_episode": old, "x": x, "shared": shared, "sentence": s.strip(),
                            "superseded_fragments": old_fs, "invalidated_edges": inval})  # fmt: skip
                break
        return out

    # ================= retrieval =================
    def _adjacency(self, upto_seq: int | None = None):
        if self._adj is not None and upto_seq is None:
            return self._adj
        cond = "" if upto_seq is None else f" AND t_valid<={int(upto_seq)}"
        inv = "t_invalid IS NULL" if upto_seq is None else f"(t_invalid IS NULL OR t_invalid>{int(upto_seq)})"
        rows = self.s.q(f"SELECT src, dst, weight, rel FROM edges WHERE {inv}{cond}")
        fcond = "" if upto_seq is None else f" WHERE f.seq<={int(upto_seq)}"
        rows += [(f, c, 1.0, "contains") for f, c in
                 self.s.q(f"SELECT m.fragment_id, m.concept_id FROM mentions m JOIN fragments f ON f.id=m.fragment_id{fcond}")]  # fmt: skip
        nodes = sorted({n for r in rows for n in r[:2]})
        ix = {n: i for i, n in enumerate(nodes)}
        w = {"cooccur": lambda x: math.log1p(x), "synonym": lambda x: x}
        data = defaultdict(float)
        for a, b, wt, rel in rows:
            v = w.get(rel, lambda x: x)(wt)
            data[(ix[a], ix[b])] += v
            data[(ix[b], ix[a])] += v
        n = len(nodes)
        if not n:
            adj = (nodes, ix, sparse.csr_matrix((0, 0)))
        else:
            ii, jj = zip(*data.keys(), strict=True)
            m = sparse.csr_matrix((list(data.values()), (ii, jj)), shape=(n, n))
            deg = np.asarray(m.sum(axis=1)).ravel()
            deg[deg == 0] = 1.0
            adj = (nodes, ix, sparse.diags(1.0 / deg) @ m)  # row-stochastic
        if upto_seq is None:
            self._adj = adj
        return adj

    def ppr(self, seeds: dict[str, float], upto_seq: int | None = None) -> dict[str, float]:
        nodes, ix, P = self._adjacency(upto_seq)
        s = np.zeros(len(nodes))
        for k, v in seeds.items():
            if k in ix:
                s[ix[k]] += v
        if not s.sum():
            return {}
        s /= s.sum()
        p = s.copy()
        PT = P.T.tocsr()
        a = self.cfg.damping
        for _ in range(self.cfg.ppr_iters):
            p = (1 - a) * s + a * (PT @ p)
        return {nodes[i]: float(p[i]) for i in np.nonzero(p)[0]}

    def seeds(self, query: str, upto_seq: int | None = None) -> Trace:
        tr = Trace(query)
        # fragments: dense + BM25 -> RRF (ARCHITECTURE_V2 §3.6)
        fids, mat = (self._fids, self._fmat) if upto_seq is None else self.s.fragment_vecs(upto_seq)
        rrf: dict[str, float] = defaultdict(float)
        if len(fids):
            qv = self.enc.one(query) if hasattr(self.enc, "one") else self.enc.encode([query])[0]
            dense = mat @ qv
            for r, i in enumerate(np.argsort(-dense, kind="stable")[: self.cfg.seed_k]):
                rrf[fids[i]] += 1 / (60 + r)
        allowed = None if upto_seq is None else set(fids)
        for r, (f, _) in enumerate(self.s.bm25(T.content_lemmas(query), self.cfg.seed_k)):
            if allowed is None or f in allowed:
                rrf[f] += 1 / (60 + r)
        tr.fragment_seeds = dict(sorted(rrf.items(), key=lambda x: -x[1])[: self.cfg.seed_k])
        # concepts named in the query, weighted by rarity
        n = max(1, self.s.q("SELECT COUNT(*) FROM fragments")[0][0])
        for canon, _, _ in X.concepts_of(query):
            r = self.s.q("SELECT n_fragments, first_seq FROM concepts WHERE id=?", (cid(canon),))
            if r and r[0][0] and (upto_seq is None or r[0][1] <= upto_seq):
                tr.concept_seeds[cid(canon)] = self.specificity(r[0][0], n)
        return tr

    def specificity(self, n_frag: int, n_total: int) -> float:
        if self.cfg.specificity == "inverse":
            return 1.0 / max(n_frag, 1)
        return math.log((n_total + 1) / (n_frag + 1)) + 1

    def retrieve(self, query: str, upto_seq: int | None = None) -> Trace:
        tr = self.seeds(query, upto_seq)
        seeds = dict(tr.concept_seeds)
        fw = self.cfg.passage_weight if seeds else 1.0
        tot = sum(tr.fragment_seeds.values()) or 1.0
        csum = sum(seeds.values()) or 1.0
        for f, v in tr.fragment_seeds.items():
            seeds[f] = seeds.get(f, 0.0) + fw * csum * v / tot
        tr.ppr = self.ppr(seeds, upto_seq)
        return tr

    def ranked_fragments(self, tr: Trace, limit: int) -> list[str]:
        fr = [(k, v) for k, v in tr.ppr.items() if k.startswith("f:")]
        return [k for k, _ in sorted(fr, key=lambda x: (-x[1], x[0]))[:limit]]

    def informative(self, text: str) -> bool:
        """Memory/periphery filter: no bare headers («Структура в монорепе:»), no bare questions
        without facts («Бэкапы как сделать?»), no scraps shorter than `min_content_words`."""
        t = text.strip()
        n = len(T.content_lemmas(t))
        if n < self.cfg.min_content_words:
            return False
        if t.endswith(":") and len(t.split()) <= 8:
            return False
        return not (t.endswith("?") and n < 6)

    def superseded(self) -> dict[str, str]:
        return {o: n for n, o in self.s.q("SELECT new_fragment, old_fragment FROM supersedes")}

    def select(self, query: str, budget: int, count: Callable[[str], int], *,
               scorer: Callable[[str, list[str]], list[float]] | None = None, candidates: int = 30,
               upto_seq: int | None = None, trace: Trace | None = None,
               extra: list[list[str]] | None = None, min_score: float | None = None,
               score_query: str | None = None, split_low: bool = False) -> tuple[str, dict]:  # fmt: skip
        """Memory block: best sentences (cross-encoder if given, else PPR / fused score) with a neighbour
        sentence as context when it fits (ARCHITECTURE_V2 §3.8). `extra` — more fragment rankings
        (e.g. sleeve mini-graphs, §3.9 step 6), fused with the query ranking by RRF. Assistant turns
        are down-weighted (× `w_assistant`, as in V6): the user's decisions matter more. `min_score` —
        with a cross-encoder, stop at this relevance instead of filling the budget to the end."""
        tr = trace or self.retrieve(query, upto_seq)
        cand = self.ranked_fragments(tr, candidates)
        base = {f: tr.ppr.get(f, 0.0) for f in cand}
        if extra:
            fused: dict[str, float] = defaultdict(float)
            for ranking in [cand] + extra:
                for r, f in enumerate(ranking):
                    fused[f] += 1 / (60 + r)
            cand = [f for f, _ in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:candidates]]
            base = {f: fused[f] for f in cand}
        if not cand:
            return "", {"trace": tr, "picked": [], "candidates": [], "lines": [], "scores": {}}
        rows = {r[0]: r for r in self.s.q(
            f"SELECT f.id, f.text, f.episode_id, f.idx, e.role FROM fragments f JOIN episodes e ON e.id=f.episode_id "
            f"WHERE f.kind='sentence' AND f.id IN ({','.join('?' * len(cand))})", tuple(cand))}  # fmt: skip
        # only informative sentences: code fragments, bare headers and bare questions are not memories
        cand = [c for c in cand if c in rows and self.informative(rows[c][1])]
        # the cross-encoder judges relevance to `score_query` (question + thinking step when given):
        # a vague question alone ranked noise like «до завтра» highest in the post-quantum example
        raw = scorer(score_query or query, [rows[c][1] for c in cand]) if scorer else [base[c] for c in cand]
        raw_of = dict(zip(cand, raw, strict=True))
        adj = {c: raw_of[c] * (self.cfg.w_assistant if rows[c][4] == "assistant" else 1.0) for c in cand}
        order = sorted(cand, key=lambda c: (-adj[c], c))
        sup = self.superseded()
        lines, picked, used, meta = [HEADER], [], set(), []
        low_lines, low_picked, low_meta = [], [], []  # hybrid: below the threshold, still within budget
        for c in order:
            if c in used:
                continue
            low = min_score is not None and scorer is not None and raw_of[c] < min_score
            if low and not split_low:
                break
            _, text, ep, idx, role = rows[c]
            nb = self.s.q("SELECT id, text FROM fragments WHERE episode_id=? AND idx IN (?,?) AND kind='sentence' "
                          "ORDER BY idx", (ep, idx - 1, idx + 1))  # fmt: skip
            mark = " (later replaced)" if c in sup else ""
            unit = [(c, text)]
            if nb and nb[0][0] not in used and self.informative(nb[0][1]):
                unit = sorted([(c, text), nb[0]], key=lambda x: int(x[0].rsplit(":", 1)[1]))
            for u in (unit, [(c, text)]) if len(unit) > 1 else ([(c, text)],):
                ln = f"- [{role} · {ep}]{mark} " + " ".join(t for _, t in u)
                # one budget for both parts: the hybrid splits what "fill to the end" would show
                if count("\n".join(lines + low_lines + [ln])) <= budget:
                    m = {"fragments": [i for i, _ in u], "role": role, "score": round(float(raw_of[c]), 6),
                         "adjusted": round(float(adj[c]), 6), "line": ln}  # fmt: skip
                    (low_lines if low else lines).append(ln)
                    (low_picked if low else picked).append([i for i, _ in u])
                    (low_meta if low else meta).append(m)
                    used |= {i for i, _ in u}
                    break
        return ("\n".join(lines) if len(lines) > 1 else ""), {"trace": tr, "picked": picked, "candidates": cand,
                                                              "lines": meta, "scores": raw_of, "low_lines": low_meta,
                                                              "low_picked": low_picked}  # fmt: skip
