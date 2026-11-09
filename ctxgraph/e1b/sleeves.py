"""V8 «рукава» (для эксперимента после E1c; в E1c не участвует).

Вопрос делится на 2–4 темы-рукава; на каждый рукав — свой поиск (эмбеддинги + BM25 → RRF) и свой
шаг по графу, как в V6. Узел, до которого дошли из ≥ 2 рукавов, — пересечение: получает бонус и
подаётся с пометкой «связывает темы: X и Y». Подача — как V6 (авто, `v8_auto`) или как V7
(нумерованное меню, `v8_menu` + `v8_block`).

Деление на рукава (`split_sleeves`):
1. Текст вопроса режется на куски: предложения, затем «… и …» / «; » / «, а также», если обе
   части несут сущности.
2. У каждого куска — якоря: сущности-понятия из архива, не частые (как `question_anchors`).
3. Соседние куски склеиваются, если у одного нет якорей (он уточняет соседа) или их эмбеддинги
   близки (cos ≥ `merge_cos`). Больше `max_sleeves` — склеиваются самые похожие соседи.
4. Последняя реплика контекста используется, только если у рукава нет якорей в архиве
   («а для него?»): тогда её якоря дописываются к запросу этого рукава.

Честный объём: блок памяти режется тем же `_fit` с тем же бюджетом, а полных фрагментов —
столько же, сколько у V6 (`full = 4`); каждому рукаву — хотя бы один, пересечения — первыми.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ctxgraph import text as T
from ctxgraph.e1b.memory import (
    HEADER,
    SELECT_INSTR,
    Archive,
    _concepts,
    _fit,
    _role,
    excerpt,
)

_SENT = re.compile(r"[^.!?;\n]+[.!?;]?")
_CONJ = re.compile(r"\s+(?:и|а также|,\s*а также|плюс)\s+", re.I)
#: Бонус пересечения в единицах RRF: ≈ балл первого места одного ранжирования (1/60).
INTERSECTION_BONUS = 1 / 60


@dataclass
class Sleeve:
    id: str
    text: str
    anchors: list[str] = field(default_factory=list)
    query: str = ""  # текст поиска (кусок вопроса, при нужде + якоря контекста)
    surfaces: dict = field(default_factory=dict)  # понятие → как написано в вопросе

    def label(self, arc: Archive) -> str:
        if self.anchors:
            a = self.anchors[0]
            return self.surfaces.get(a) or arc.surface.get(a, a)
        words = T.content_lemmas(self.text)
        return " ".join(words[:2]) or self.text[:20]


def _anchors(arc: Archive, text: str) -> list[str]:
    """Понятия куска вопроса: сначала те, что есть в архиве (по редкости), затем остальные.
    Понятия, которых нет в архиве дословно («jwt» при «jwt-сессия» в архиве), тоже держат тему."""
    ents = list(dict.fromkeys(c for c, _ in _concepts(text)))
    known = sorted(
        (e for e in ents if e in arc.by_ent and not arc.common(e)), key=lambda e: (arc.df(e), e)
    )
    unknown = [e for e in ents if e not in arc.by_ent]
    return known + unknown


def _first_surfaces(text: str) -> dict[str, str]:
    """Понятие → первое его написание в тексте (как в вопросе: «JWT», «переводы»)."""
    out: dict[str, str] = {}
    for c, sf in _concepts(text):
        out.setdefault(c, sf)
    return out


def _pieces(text: str) -> list[str]:
    out = []
    for s in (x.strip() for x in _SENT.findall(text)):
        if not s:
            continue
        parts = [p.strip(" ,") for p in _CONJ.split(s) if p.strip(" ,")]
        # делим по союзу, только если у каждой части есть свои понятия
        if len(parts) > 1 and all(_concepts(p) for p in parts):
            out += parts
        else:
            out.append(s)
    return out


def split_sleeves(
    arc: Archive,
    question: str,
    context: str | None = None,
    *,
    max_sleeves: int = 4,
    merge_cos: float = 0.8,
) -> list[Sleeve]:
    pieces = _pieces(question) or [question]
    groups = [[p] for p in pieces]
    anchors = [_anchors(arc, p) for p in pieces]

    def emb(g):
        return arc.encoder.one(" ".join(g))

    # склейка: кусок без якорей — к соседу; близкие соседи — вместе
    i = 0
    while i < len(groups) - 1:
        a, b = groups[i], groups[i + 1]
        if not anchors[i] or not anchors[i + 1] or float(emb(a) @ emb(b)) >= merge_cos:
            groups[i] = a + b
            anchors[i] = list(dict.fromkeys(anchors[i] + anchors[i + 1]))
            del groups[i + 1], anchors[i + 1]
        else:
            i += 1
    while len(groups) > max_sleeves:  # лишнее — склеить самую похожую пару соседей
        sims = [float(emb(groups[j]) @ emb(groups[j + 1])) for j in range(len(groups) - 1)]
        j = int(np.argmax(sims))
        groups[j] += groups.pop(j + 1)
        anchors[j] = list(dict.fromkeys(anchors[j] + anchors.pop(j + 1)))
    ctx_anchors = _anchors(arc, context) if context else []
    out = []
    for n, (g, a) in enumerate(zip(groups, anchors, strict=True), 1):
        text = " ".join(g)
        query = text if a or not ctx_anchors else text + " " + " ".join(ctx_anchors[:3])
        out.append(Sleeve(f"s{n}", text, a, query, _first_surfaces(text)))
    return out


def _rrf(arc: Archive, query: str) -> np.ndarray:
    rrf = np.zeros(len(arc.turns))
    for scores in (arc.dense(query), arc.bm25(query)):
        for rank, i in enumerate(np.argsort(-scores, kind="stable")):
            rrf[i] += 1 / (60 + rank)
    return rrf


def sleeve_candidates(
    arc: Archive, question: str, context: str | None = None, *, k: int = 8
) -> dict:
    """Поиск по рукавам + граф по рукавам + пересечения. Возвращает ранжирование и лог."""
    sleeves = split_sleeves(arc, question, context)
    order = {t.id: i for i, t in enumerate(arc.turns)}
    newer = {old: new for new, olds in arc.replaces.items() for old in olds}
    k_s = max(2, -(-k // len(sleeves)))  # top-k на рукав: общий пул ≈ как у V6
    reached: dict[str, set[str]] = {}
    best: dict[str, float] = {}
    per_sleeve: dict[str, list[str]] = {}
    for s in sleeves:
        rrf = _rrf(arc, s.query)
        top = [arc.turns[i].id for i in np.argsort(-rrf, kind="stable")[:k_s]]
        cand = dict.fromkeys(top)
        for tid in top[:3]:  # тот же шаг по графу, что в V6, но от своего рукава
            for e in sorted(arc.ents[tid], key=lambda e: (arc.df(e), e))[:2]:
                for o in arc.by_ent[e][-2:]:
                    cand.setdefault(o)
        for tid in list(cand):
            if tid in newer:
                cand.setdefault(newer[tid])
        per_sleeve[s.id] = list(cand)
        for tid in cand:
            reached.setdefault(tid, set()).add(s.id)
            best[tid] = max(best.get(tid, 0.0), float(rrf[order[tid]]))
    score = {}
    for tid, ss in reached.items():
        w = 1.0 if arc.by_id[tid].origin != "model" else 0.6  # как в V6
        score[tid] = w * (best[tid] + INTERSECTION_BONUS * (len(ss) - 1))
    ranked = sorted(score, key=lambda t: (-score[t], order[t]))
    inter = [t for t in ranked if len(reached[t]) >= 2]
    return {
        "sleeves": sleeves,
        "ranked": ranked,
        "reached": reached,
        "intersections": inter,
        "per_sleeve": per_sleeve,
        "newer": newer,
        "k_per_sleeve": k_s,
    }


def _marks(arc: Archive, tid: str, c: dict, short: bool = False) -> str:
    by_id = {s.id: s for s in c["sleeves"]}
    out = ""
    if len(c["reached"][tid]) >= 2:
        names = " и ".join(f"«{by_id[s].label(arc)}»" for s in sorted(c["reached"][tid]))
        out += f" [связывает темы: {names}]"
    if tid in c["newer"]:
        note = arc.change_note.get(c["newer"][tid])
        out += f" [позже заменено: «{note}»]" if note else " [позже заменено]"
    return out


def _pick_full(c: dict, full: int) -> list[str]:
    """Пересечения — первыми; затем каждому рукаву хотя бы по одному; остальное — по баллу."""
    picked = list(c["intersections"][:full])
    for s in c["sleeves"]:
        if len(picked) >= full:
            break
        own = [t for t in c["ranked"] if t in c["per_sleeve"][s.id] and t not in picked]
        if own and not any(t in c["per_sleeve"][s.id] for t in picked):
            picked.append(own[0])
    for t in c["ranked"]:
        if len(picked) >= full:
            break
        if t not in picked:
            picked.append(t)
    return picked


def log_of(c: dict, picked: list[str]) -> dict:
    return {
        "n_sleeves": len(c["sleeves"]),
        "sleeves": [{"id": s.id, "text": s.text, "anchors": s.anchors} for s in c["sleeves"]],
        "n_intersections": len(c["intersections"]),
        "intersections": c["intersections"],
        "k_per_sleeve": c["k_per_sleeve"],
        "picked": picked,
        "picked_sleeves": {t: sorted(c["reached"][t]) for t in picked},
    }


def v8_auto(
    arc: Archive,
    q: str,
    count: Callable[[str], int],
    budget: int,
    *,
    context: str | None = None,
    k: int = 8,
    full: int = 4,
) -> tuple[str, dict]:
    """Подача как V6: тексты `full` лучших (пересечения первыми) + ярлыки остальных."""
    if not arc.turns:
        return "", {"n_sleeves": 0}
    c = sleeve_candidates(arc, q, context, k=k)
    picked = _pick_full(c, full)
    lines = [
        f"- [{_role(arc.by_id[t], trust=True)}]{_marks(arc, t, c)} {excerpt(arc.by_id[t].text, count, 110)}"
        for t in picked
    ]
    rest = [arc.by_id[t] for t in c["ranked"] if t not in picked]
    if rest:
        lines.append("Ещё по теме: " + "; ".join(excerpt(t.text, count, 12) for t in rest))
    return _fit(lines, count, budget), log_of(c, picked)


def v8_menu(
    arc: Archive, q: str, count: Callable[[str], int], *, context: str | None = None, k: int = 8
) -> tuple[str, dict[int, str], dict]:
    """Подача как V7: нумерованное меню того же пула с пометками тем и пересечений."""
    if not arc.turns:
        return "", {}, {"n_sleeves": 0}
    c = sleeve_candidates(arc, q, context, k=k)
    menu, lines = {}, []
    for n, tid in enumerate(c["ranked"], 1):
        t = arc.by_id[tid]
        lines.append(
            f"{n}. [{_role(t, trust=True)}]{_marks(arc, tid, c, short=True)} {excerpt(t.text, count, 14)}"
        )
        menu[n] = tid
    log = log_of(c, []) | {"marks": {tid: _marks(arc, tid, c) for tid in c["ranked"]}}
    return HEADER + "\n" + "\n".join(lines), menu, log


def v8_block(
    arc: Archive, picked: list[int], menu: dict[int, str], log: dict, count, budget: int
) -> str:
    """Раскрыть выбранное моделью (V8m) — в том же виде и с теми же пометками, что V8a."""
    if not picked:
        return ""
    marks = log.get("marks", {})
    lines = [
        f"- [{_role(arc.by_id[menu[n]], trust=True)}]{marks.get(menu[n], '')} "
        f"{excerpt(arc.by_id[menu[n]].text, count, 110)}"
        for n in picked
    ]
    return _fit(lines, count, budget)


def fact_coverage(
    picked_ids: list[str], turn_facts: dict[str, list[str]], anchors: list[str]
) -> dict:
    """Лог для анализа: какие опорные факты вопроса попали в раскрытое (по разметке реплик загрузки)."""
    got = {f for t in picked_ids for f in turn_facts.get(t, [])} & set(anchors)
    return {
        "anchors": anchors,
        "covered": sorted(got),
        "share": len(got) / len(anchors) if anchors else None,
    }


__all__ = [
    "SELECT_INSTR",
    "Sleeve",
    "fact_coverage",
    "sleeve_candidates",
    "split_sleeves",
    "v8_auto",
    "v8_block",
    "v8_menu",
]
