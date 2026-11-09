"""Общий слой отбора памяти (одинаковый для V6, V9-control, V9): предложения + cross-encoder.

1. Кандидаты — пул реплик по запросу (вопрос или текст шага): `kind="v6"` — `v6_candidates`
   с k = `pool`, обрезан до `pool` в порядке V6; `kind="hybrid"` — RRF(эмбеддинги, BM25) без
   графа, top-`pool` (V2+ «сильный RAG», главный соперник).
2. Единица памяти — предложение (`coverage.split_sentences`), с id реплики, автором и местом.
3. Переранжирование: cross-encoder (`Reranker`, по умолчанию BAAI/bge-reranker-v2-m3 на CPU),
   пара (запрос, предложение).
4. В память — лучшие предложения в бюджет; к каждому — соседнее предложение той же реплики как
   контекст (то, что выше по баллу), если влезает. Строки — «- [роль · id] текст».
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ctxgraph.e1b import memory as M
from ctxgraph.e1b.coverage import split_sentences

RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass
class Sent:
    tid: str
    i: int
    role: str
    text: str


class Reranker:
    """CrossEncoder с кэшем баллов по (запрос, предложение)."""

    def __init__(self, model: str = RERANKER, revision: str | None = None, device: str = "cpu",
                 trust_remote_code: bool = False):  # gte-multilingual-reranker ships its own model code
        from sentence_transformers import CrossEncoder

        self.m = CrossEncoder(model, revision=revision, device=device, max_length=512,
                              trust_remote_code=trust_remote_code)
        self.name = f"{model}@{revision or 'main'}"
        self.cache: dict[tuple[str, str], float] = {}
        self.disk = None

    def score(self, query: str, texts: list[str]) -> list[float]:
        new = [t for t in dict.fromkeys(texts) if (query, t) not in self.cache]
        if new:
            scores = self.m.predict([(query, t) for t in new], batch_size=16)
            for t, sc in zip(new, scores, strict=True):
                self.cache[(query, t)] = float(sc)
            if self.disk:
                with self.disk.open("a", encoding="utf-8") as fh:
                    for t in new:
                        rec = [query, t, self.cache[(query, t)]]
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [self.cache[(query, t)] for t in texts]

    def persist(self, path) -> None:
        """Кэш баллов на диске (jsonl): перезапуск не пересчитывает готовое."""
        from pathlib import Path

        self.disk = Path(path)
        self.disk.parent.mkdir(parents=True, exist_ok=True)
        if self.disk.exists():
            for ln in self.disk.read_text("utf-8").splitlines():
                if ln.strip():
                    q, t, sc = json.loads(ln)
                    self.cache[(q, t)] = sc


def hybrid_pool(arc: M.Archive, query: str, pool: int = 30) -> list[str]:
    """Гибридный поиск без графа: RRF(эмбеддинги, BM25) по всему запросу, top-`pool`."""
    rrf = np.zeros(len(arc.turns))
    for scores in (arc.dense(query), arc.bm25(query)):
        for rank, i in enumerate(np.argsort(-scores, kind="stable")):
            rrf[i] += 1 / (60 + rank)
    return [arc.turns[i].id for i in np.argsort(-rrf, kind="stable")[:pool]]


def turn_sentences(arc: M.Archive, ranked: list[str]) -> list[Sent]:
    out = []
    for tid in ranked:
        t = arc.by_id[tid]
        out += [Sent(tid, i, t.role, s) for i, s in enumerate(split_sentences(t.text))]
    return out


def pool_sentences(arc: M.Archive, query: str, pool: int = 30, kind: str = "v6") -> list[Sent]:
    """kind="v6" — пул V6 (гибрид + шаг по графу + «заменяет»); "hybrid" — без графа (V2+)."""
    if kind == "hybrid":
        ranked = hybrid_pool(arc, query, pool)
    else:
        ranked = M.v6_candidates(arc, query, k=pool)[0][:pool]
    return turn_sentences(arc, ranked)


def pick_lines(
    sents: list[Sent],
    query: str,
    count: Callable[[str], int],
    budget: int,
    scorer: Callable[[str, list[str]], list[float]],
    *,
    head: list[str] | None = None,
    exclude: set | frozenset = frozenset(),
    neighbors: bool = True,
) -> tuple[list[str], list[list[tuple[str, int]]]]:
    """Лучшие по cross-encoder предложения (с соседним, если влезает), пока весь блок
    `head + строки` укладывается в `budget`. `exclude` — уже показанные (id реплики, номер)."""
    sents = [s for s in sents if (s.tid, s.i) not in exclude]
    head = list(head or [])
    if not sents:
        return [], []
    scores = scorer(query, [s.text for s in sents])
    by_pos = {(s.tid, s.i): (s, sc) for s, sc in zip(sents, scores, strict=True)}
    order = sorted(range(len(sents)), key=lambda j: (-scores[j], j))
    lines, used, picked = [], set(), []
    for j in order:
        s = sents[j]
        if (s.tid, s.i) in used:
            continue
        role = M.ROLE_RU[s.role]
        unit = [s]
        if neighbors:
            nb = [by_pos[p] for p in ((s.tid, s.i - 1), (s.tid, s.i + 1)) if p in by_pos and p not in used]
            if nb:
                unit = sorted([s, max(nb, key=lambda x: x[1])[0]], key=lambda x: x.i)
        for u in (unit, [s]) if len(unit) > 1 else ([s],):
            ln = f"- [{role} · {s.tid}] " + " ".join(x.text for x in u)
            if count("\n".join(head + lines + [ln])) <= budget:
                lines.append(ln)
                used |= {(x.tid, x.i) for x in u}
                picked.append([(x.tid, x.i) for x in u])
                break
    return lines, picked


def select(
    arc: M.Archive,
    query: str,
    count: Callable[[str], int],
    budget: int,
    scorer: Callable[[str, list[str]], list[float]],
    *,
    pool: int = 30,
    neighbors: bool = True,
    kind: str = "v6",
) -> tuple[str, dict]:
    if not arc.turns:
        return "", {"picked": []}
    sents = pool_sentences(arc, query, pool, kind)
    lines, picked = pick_lines(sents, query, count, budget, scorer, head=[M.HEADER], neighbors=neighbors)
    text = "\n".join([M.HEADER] + lines) if lines else ""
    return text, {"picked": picked, "n_sentences": len(sents)}


def v2_plus(arc: M.Archive, q: str, count, budget: int, scorer, pool: int = 30) -> tuple[str, dict]:
    """V2+ «сильный RAG» — главный соперник: гибрид без графа → предложения → cross-encoder."""
    return select(arc, q, count, budget, scorer, pool=pool, kind="hybrid")
