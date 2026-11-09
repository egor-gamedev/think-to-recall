"""Память вариантов E1b. Источник — архив чата (реплики, ушедшие в резюме).

V1 — без памяти; V2 — RAG по эмбеддингам; V3 — воспоминания графа, auto_depth = 1;
V4 — меню графа + раскрытие по выбору модели; V5 — случайные узлы той же формы;
V6 — гибрид: BM25 + эмбеддинги → 1 шаг по графу (общая сущность, «заменяет») → тексты лучших
фрагментов + меню ярлыков остальных; слова модели помечены и ниже по доверию.
Все блоки — в пределах одного бюджета токенов и с одной пометкой «воспоминания, не требования».
"""

from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ctxgraph import text as T
from ctxgraph.e1b.chat import Turn
from ctxgraph.ingest import extract_entities
from ctxgraph.sources.experience import is_concept

HEADER = (
    "Это воспоминания и ассоциации из прошлого разговора, не требования; "
    "используй, только если относится к вопросу."
)
ROLE_RU = {"user": "пользователь", "assistant": "ассистент"}
#: Явные слова смены решения (без «теперь» — оно встречается почти везде).
CHANGE_MARKERS = (
    "отказыва",
    "отказал",
    "передумал",
    "ушли с",
    "уходим с",
    "переходим на",
    "перейти на",
    "перешли на",
    "больше не",
    "убираем",
    "убрали",
    "вместо",
    "заменил",
    "switched",
    "replaced",
    "instead",
)
#: Версия детектора «заменяет»: v1 — как в E1b (воспроизводимость), v2 — E1c (ctxgraph/e1b/replaces.py).
REPLACES_VERSION = os.environ.get("CTXGRAPH_REPLACES", "v2")
#: Сущности, которые встречаются в большей доле реплик, — не якоря и не основание для «заменяет».
COMMON_SHARE = 0.15
_CODE = re.compile(r"```.*?```", re.S)


def excerpt(text: str, count: Callable[[str], int], max_tokens: int) -> str:
    """Начало реплики без блоков кода, обрезанное по словам до max_tokens."""
    plain = re.sub(r"\s+", " ", _CODE.sub(" [код] ", text)).strip()
    if count(plain) <= max_tokens:
        return plain
    words, out = plain.split(), []
    for w in words:
        if count(" ".join(out + [w])) > max_tokens:
            break
        out.append(w)
    return " ".join(out) + " …"


_CONCEPTS: dict[str, list[tuple[str, str]]] = {}


def _concepts(text: str) -> list[tuple[str, str]]:
    """Сущности-понятия реплики (canonical, surface); кэш — архив пересобирается к каждому вопросу."""
    if text not in _CONCEPTS:
        _CONCEPTS[text] = [
            (e.canonical, e.surface)
            for e in extract_entities(text)
            if is_concept(e.canonical, e.kind, e.surface)
        ]
    return _CONCEPTS[text]


@dataclass
class Archive:
    """Индекс архива: сущности реплик, совместная встречаемость, эмбеддинги, BM25, рёбра «заменяет»."""

    turns: list[Turn]
    encoder: object
    ents: dict[str, set[str]] = field(default_factory=dict)  # turn_id → сущности
    by_ent: dict[str, list[str]] = field(default_factory=dict)  # сущность → turn_ids по порядку
    surface: dict[str, str] = field(default_factory=dict)
    replaces: dict[str, list[str]] = field(default_factory=dict)  # поздняя реплика → ранние

    def __post_init__(self) -> None:
        self.by_id = {t.id: t for t in self.turns}
        self.order = {t.id: i for i, t in enumerate(self.turns)}
        by_ent: dict[str, list[str]] = defaultdict(list)
        for t in self.turns:
            es = set()
            for canon, surf in _concepts(t.text):
                es.add(canon)
                self.surface.setdefault(canon, surf)
            self.ents[t.id] = es
            for e in es:
                by_ent[e].append(t.id)
        self.by_ent = dict(by_ent)
        self.n = max(len(self.turns), 1)
        # эмбеддинги и BM25
        texts = [excerpt(t.text, T.approx_tokens, 200) for t in self.turns]
        self.emb = self.encoder.encode(texts) if texts else np.zeros((0, 1))
        self.docs = [T.content_lemmas(x) for x in texts]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        self.dfl = Counter(w for d in self.docs for w in set(d))
        if REPLACES_VERSION == "v2":  # E1c: явная смена, вырезано старое, неуверенные не помечаются
            from ctxgraph.e1b.replaces import detect

            for e in detect(self.turns, _concepts):
                self.replaces[e["new"]] = [e["old"]]
            return
        # v1 (E1b): реплика пользователя со словом смены решения → одна более ранняя реплика
        # пользователя: сумма idf общих не частых сущностей + 5 × косинус эмбеддингов
        for i, t in enumerate(self.turns):
            if t.role != "user" or not any(m in t.text.lower() for m in CHANGE_MARKERS):
                continue
            score: dict[int, float] = defaultdict(float)
            for e in sorted(self.ents[t.id]):  # порядок суммирования не зависит от хэша
                if self.common(e):
                    continue
                for tid in self.by_ent[e]:
                    j = self.order[tid]
                    if j < i and self.turns[j].role == "user":
                        score[j] += math.log(self.n / self.df(e))
            if score:
                best = max(
                    score, key=lambda j: (score[j] + 5 * float(self.emb[j] @ self.emb[i]), j)
                )
                self.replaces[t.id] = [self.turns[best].id]

    def common(self, ent: str) -> bool:
        return self.df(ent) > max(3, COMMON_SHARE * self.n)

    def df(self, ent: str) -> int:
        return len(self.by_ent.get(ent, []))

    def cooc(self, ent: str, min_count: int = 2) -> list[tuple[str, int]]:
        """Сущности, встречавшиеся с ent в ≥ min_count репликах, с PMI > 0."""
        cnt: Counter[str] = Counter()
        for tid in self.by_ent.get(ent, []):
            for o in self.ents[tid]:
                if o != ent:
                    cnt[o] += 1
        out = []
        for o, c in cnt.items():
            pmi = math.log(c * self.n / (self.df(ent) * self.df(o)))
            if c >= min_count and pmi > 0:
                out.append((o, c))
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def dense(self, q: str) -> np.ndarray:
        return self.emb @ self.encoder.one(q) if len(self.turns) else np.zeros(0)

    def bm25(self, q: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        qs = T.content_lemmas(q)
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            for w in qs:
                if w in tf:
                    idf = math.log(1 + (self.n - self.dfl[w] + 0.5) / (self.dfl[w] + 0.5))
                    out[i] += (
                        idf
                        * tf[w]
                        * (k1 + 1)
                        / (tf[w] + k1 * (1 - b + b * len(d) / max(self.avgdl, 1)))
                    )
        return out


def question_anchors(arc: Archive, q: str, limit: int = 4) -> list[str]:
    """Сущности вопроса, которые есть в архиве; сначала более редкие (специфичные)."""
    qe = [e.canonical for e in extract_entities(q) if is_concept(e.canonical, e.kind, e.surface)]
    seen = [e for e in dict.fromkeys(qe) if e in arc.by_ent and not arc.common(e)]
    return sorted(seen, key=lambda e: (arc.df(e), e))[:limit]


def _fit(lines: list[str], count: Callable[[str], int], budget: int) -> str:
    out = [HEADER]
    for ln in lines:
        if count("\n".join(out + [ln])) > budget:
            break
        out.append(ln)
    return "\n".join(out) if len(out) > 1 else ""


def _role(t: Turn, trust: bool = False) -> str:
    r = ROLE_RU[t.role]
    return f"{r}, сказано моделью — ниже доверие" if trust and t.origin == "model" else r


# ---------------- варианты ----------------
def v2_rag(arc: Archive, q: str, count, budget: int, k: int = 6) -> str:
    if not arc.turns:
        return ""
    idx = np.argsort(-arc.dense(q), kind="stable")[:k]
    lines = [f"- [{_role(arc.turns[i])}] {excerpt(arc.turns[i].text, count, 110)}" for i in idx]
    return _fit(lines, count, budget)


def graph_items(arc: Archive, q: str) -> list[dict]:
    """Якоря вопроса → последняя реплика с якорем + ассоциации (общий материал для V3/V4)."""
    items = []
    for a in question_anchors(arc, q):
        last = arc.by_id[arc.by_ent[a][-1]]
        assoc = [o for o, _ in arc.cooc(a)[:4]]
        items.append(
            {
                "anchor": a,
                "label": arc.surface.get(a, a),
                "last": last,
                "assoc": [(o, arc.surface.get(o, o)) for o in assoc],
            }
        )
    return items


def v3_graph(arc: Archive, q: str, count, budget: int) -> str:
    lines = []
    for it in graph_items(arc, q):
        assoc = ", ".join(lab for _, lab in it["assoc"]) or "—"
        lines.append(
            f"• {it['label']}: связано с {assoc}. Последнее упоминание "
            f"[{_role(it['last'])}]: «{excerpt(it['last'].text, count, 70)}»"
        )
        if it["assoc"]:  # auto_depth = 1: раскрыть самую частую ассоциацию
            o, lab = it["assoc"][0]
            t = arc.by_id[arc.by_ent[o][-1]]
            lines.append(f"  ↳ {lab}: «{excerpt(t.text, count, 50)}»")
    return _fit(lines, count, budget)


def v4_menu(arc: Archive, q: str, count, budget: int) -> tuple[str, dict[str, str]]:
    """Меню ярлыков + функция раскрытия: id → текст воспоминаний по сущности."""
    lines, expand = [], {}
    for it in graph_items(arc, q):
        ids = [it["anchor"]] + [o for o, _ in it["assoc"]]
        labs = [it["label"]] + [lab for _, lab in it["assoc"]]
        lines.append(
            f"• {labs[0]} [m:{ids[0]}]: связано с "
            + ", ".join(f"{lab} [m:{i}]" for i, lab in zip(ids[1:], labs[1:], strict=True))
        )
        for i in ids:
            tids = arc.by_ent[i][-3:]
            expand[f"m:{i}"] = "\n".join(
                f"- [{_role(arc.by_id[t])}] {excerpt(arc.by_id[t].text, count, 80)}" for t in tids
            )
    text = _fit(lines, count, budget)
    if text:
        text += "\nРаскрыть пункт: expand(id), id в квадратных скобках."
    return text, expand


def v5_random(arc: Archive, q: str, count, budget: int, pool: list[Turn], seed: int) -> str:
    """Форма V3, но якоря, ассоциации и выдержки — случайные из чужих проектов."""
    rng = random.Random(seed)
    shape = graph_items(arc, q)
    if not shape or not pool:
        return ""
    words = sorted({w for t in pool for w in T.content_lemmas(t.text) if len(w) > 4})
    lines = []
    for it in shape:
        lab = rng.choice(words)
        assoc = ", ".join(rng.choice(words) for _ in it["assoc"]) or "—"
        t = rng.choice(pool)
        lines.append(
            f"• {lab}: связано с {assoc}. Последнее упоминание [{ROLE_RU[t.role]}]: «{excerpt(t.text, count, 70)}»"
        )
        if it["assoc"]:
            t2 = rng.choice(pool)
            lines.append(f"  ↳ {rng.choice(words)}: «{excerpt(t2.text, count, 50)}»")
    return _fit(lines, count, budget)


def v6_candidates(arc: Archive, q: str, k: int = 8) -> tuple[list[str], dict[str, str]]:
    """Гибридный поиск: RRF(BM25, эмбеддинги) → топ-k → 1 шаг по графу. Возвращает ранжированные
    id реплик и словарь «устаревшая → заменившая». Общий пул для V6 (авто) и V7 (выбор моделью)."""
    dense, sparse = arc.dense(q), arc.bm25(q)
    rrf = np.zeros(len(arc.turns))
    for scores in (dense, sparse):
        for rank, i in enumerate(np.argsort(-scores, kind="stable")):
            rrf[i] += 1 / (60 + rank)
    top = [arc.turns[i].id for i in np.argsort(-rrf, kind="stable")[:k]]
    # 1 шаг по графу: реплики с общей сущностью у трёх лучших + цепочка «заменяет»
    cand = dict.fromkeys(top)
    for tid in top[:3]:
        for e in sorted(arc.ents[tid], key=lambda e: (arc.df(e), e))[:2]:
            for o in arc.by_ent[e][-2:]:
                cand.setdefault(o)
    newer = {old: new for new, olds in arc.replaces.items() for old in olds}
    for tid in list(cand):
        if tid in newer:
            cand.setdefault(newer[tid])
    order = {t.id: i for i, t in enumerate(arc.turns)}
    score = {tid: rrf[order[tid]] for tid in cand}
    # слова пользователя выше слов модели
    ranked = sorted(
        cand,
        key=lambda t: (-(score[t] * (1.0 if arc.by_id[t].origin != "model" else 0.6)), order[t]),
    )
    return ranked, newer


def _full_line(arc: Archive, tid: str, newer: dict, count) -> str:
    t = arc.by_id[tid]
    mark = " (позже заменено — см. более позднюю реплику)" if tid in newer else ""
    return f"- [{_role(t, trust=True)}]{mark} {excerpt(t.text, count, 110)}"


def v6_hybrid(arc: Archive, q: str, count, budget: int, k: int = 8, full: int = 4) -> str:
    if not arc.turns:
        return ""
    ranked, newer = v6_candidates(arc, q, k)
    lines = [_full_line(arc, tid, newer, count) for tid in ranked[:full]]
    rest = [arc.by_id[t] for t in ranked[full:]]
    if rest:
        lines.append("Ещё по теме: " + "; ".join(excerpt(t.text, count, 12) for t in rest))
    return _fit(lines, count, budget)


# ---------------- V7: выбор моделью без инструментов (двухшаговый протокол) ----------------
SELECT_INSTR = (
    "Ниже меню воспоминаний из прошлого разговора. Прежде чем отвечать, реши, какие пункты "
    "раскрыть, чтобы ответить с учётом проекта (не больше {max_pick}). Ответь строго одной строкой "
    "и ничего больше: «РАСКРЫТЬ: <номера через запятую>» или «НИЧЕГО»."
)
_PICK = re.compile(r"РАСКРЫТЬ\s*[:\-—]?\s*(\d[\d\s,;и]*)", re.I)
_NONE = re.compile(r"^\W*ничего", re.I | re.M)
_MARKUP = re.compile(r"[*_`«»\"]")


def v7_menu(arc: Archive, q: str, count, k: int = 8) -> tuple[str, dict[int, str]]:
    """Нумерованное меню ярлыков (тот же пул, что у V6) и отображение номер → id реплики."""
    if not arc.turns:
        return "", {}
    ranked, newer = v6_candidates(arc, q, k)
    menu, lines = {}, []
    for n, tid in enumerate(ranked, 1):
        t = arc.by_id[tid]
        mark = " [позже заменено]" if tid in newer else ""
        lines.append(f"{n}. [{_role(t, trust=True)}]{mark} {excerpt(t.text, count, 14)}")
        menu[n] = tid
    return HEADER + "\n" + "\n".join(lines), menu


def parse_selection(text: str, menu: dict[int, str], max_pick: int = 4) -> dict:
    """«РАСКРЫТЬ: 3, 7» → [3, 7]; «НИЧЕГО» → []. Нарушение формата → [] и format_ok=False.
    Номера вне меню отбрасываются; больше max_pick — берутся первые."""
    raw = (text or "").strip()
    clean = _MARKUP.sub("", raw)  # «**РАСКРЫТЬ:** 3» и т. п.
    m = _PICK.search(clean)
    if m:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        valid = [n for n in dict.fromkeys(nums) if n in menu]
        return {"picked": valid[:max_pick], "format_ok": True, "invalid": [n for n in nums if n not in menu],
                "over_limit": len(valid) > max_pick, "raw": raw[:300]}  # fmt: skip
    if _NONE.search(clean):
        return {
            "picked": [],
            "format_ok": True,
            "invalid": [],
            "over_limit": False,
            "raw": raw[:300],
        }
    return {"picked": [], "format_ok": False, "invalid": [], "over_limit": False, "raw": raw[:300]}


def v7_block(arc: Archive, picked: list[int], menu: dict[int, str], count, budget: int) -> str:
    """Раскрытые моделью пункты — тем же видом, что полные фрагменты V6."""
    if not picked:
        return ""
    newer = {old: new for new, olds in arc.replaces.items() for old in olds}
    return _fit([_full_line(arc, menu[n], newer, count) for n in picked], count, budget)
