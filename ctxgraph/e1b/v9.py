"""V9 «рукава мышления»: рукава из любого текста, поиск по рукавам, пересечения, якорь цели.

V9-q — один раунд по тексту вопроса, один блок памяти (без генерации, полнота считается без API).
Тот же `candidates` используется на шагах мышления V9 (этап B): текст = шаг, якорь = вопрос,
пересечения — с учётом рукавов прошлых шагов (`prev_reached`).

1. **Рукава** (`text_sleeves`): сущности-понятия текста, которые есть в архиве и не частые
   (как `question_anchors`), редкие первыми; если их меньше `max_sleeves` — добираются ключевые
   слова (леммы текста, редкие в архиве, не входящие в взятые сущности). До 4.
   Запрос рукава: `query_mode="entity"` — само слово; `"clause"` — слово + кусок текста с ним.
2. **На рукав:** гибридный поиск (эмбеддинги + BM25 → RRF) top-k_s, k_s = ⌈k / рукавов⌉, не
   меньше 2; шаг по графу как в V6 (две редкие сущности трёх лучших → их последние две реплики),
   для рукава-сущности — ещё две последние реплики с ней; цепочка «заменяет» (заменившая реплика).
3. **Пересечения:** реплика, найденная от ≥ 2 разных рукавов (по ключу рукава, с учётом прошлых
   шагов), получает бонус `bonus × (рукавов − 1)` к баллу RRF и идёт первой.
4. **Якорь цели:** кандидат отбрасывается, если косинус его эмбеддинга с исходным вопросом
   ниже `anchor_cos` (None — без фильтра).
5. **Подача V9-q:** как V6/V8 — `full` полных фрагментов (пересечения первыми, каждому рукаву хотя
   бы один) + ярлыки остальных, тот же `_fit` и бюджет.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ctxgraph import text as T
from ctxgraph.e1b import memory as M
from ctxgraph.e1b import sleeves as S
from ctxgraph.e1b.memory import Archive, _concepts, _fit, _role, excerpt

#: Бонус пересечения в единицах RRF (≈ балл первого места одного ранжирования), как у V8.
BONUS = 1 / 60
MAX_SLEEVES = 4
_SENT = re.compile(r"[^.!?;\n]+[.!?;]?")


@dataclass
class Sleeve:
    id: str  # ключ рукава: понятие или лемма — общий для всех шагов
    kind: str  # "entity" | "phrase"
    surface: str
    query: str
    lemmas: set[str] = field(default_factory=set)

    def label(self, arc: Archive) -> str:
        return self.surface


def _clause(text: str, surface: str) -> str:
    low = surface.lower()
    for s in (x.strip() for x in _SENT.findall(text)):
        if low in s.lower():
            return s
    return text


def text_sleeves(
    arc: Archive, text: str, *, max_sleeves: int = MAX_SLEEVES, query_mode: str = "entity"
) -> list[Sleeve]:
    surf: dict[str, str] = {}
    for c, sf in _concepts(text):
        surf.setdefault(c, sf)
    ents = sorted(
        (e for e in surf if e in arc.by_ent and not arc.common(e)), key=lambda e: (arc.df(e), e)
    )[:max_sleeves]
    out = []
    taken: set[str] = set()
    for e in ents:
        lem = set(T.content_lemmas(surf[e]))
        taken |= lem
        out.append(Sleeve(e, "entity", surf[e], surf[e], lem))
    if len(out) < max_sleeves:
        limit = max(3, 0.15 * arc.n)
        words = [
            w
            for w in dict.fromkeys(T.content_lemmas(text))
            if w not in taken and w not in surf and 0 < arc.dfl.get(w, 0) <= limit
        ]
        for w in sorted(words, key=lambda w: (arc.dfl[w], w))[: max_sleeves - len(out)]:
            out.append(Sleeve(w, "phrase", w, w, {w}))
    if query_mode == "clause":
        for s in out:
            cl = _clause(text, s.surface)
            s.query = s.surface if cl == s.surface else f"{s.surface}. {cl}"
    return out


def candidates(
    arc: Archive,
    text: str,
    question: str | None = None,
    *,
    sleeves: list[Sleeve] | None = None,
    prev_reached: dict[str, set[str]] | None = None,
    k: int = 8,
    bonus: float = BONUS,
    anchor_cos: float | None = None,
    query_mode: str = "entity",
    exclude: set[str] | frozenset = frozenset(),
    k_per_sleeve: int | None = None,
    bonus_norm: str | None = None,
) -> dict:
    """Пул кандидатов по рукавам текста. Формат совместим с `sleeves._pick_full`."""
    question = text if question is None else question
    if sleeves is None:
        sleeves = text_sleeves(arc, text, query_mode=query_mode)
    if not sleeves:  # нет ни понятий, ни редких слов — один рукав по всему тексту
        sleeves = [Sleeve("_text", "text", text[:30], text)]
    order = arc.order
    newer = {old: new for new, olds in arc.replaces.items() for old in olds}
    k_s = k_per_sleeve or max(2, -(-k // len(sleeves)))
    reached: dict[str, set[str]] = {}
    best: dict[str, float] = {}
    per_sleeve: dict[str, list[str]] = {}
    for s in sleeves:
        rrf = S._rrf(arc, s.query)
        top = [arc.turns[i].id for i in np.argsort(-rrf, kind="stable")[:k_s]]
        cand = dict.fromkeys(top)
        for tid in top[:3]:
            for e in sorted(arc.ents[tid], key=lambda e: (arc.df(e), e))[:2]:
                for o in arc.by_ent[e][-2:]:
                    cand.setdefault(o)
        if s.kind == "entity":
            for o in arc.by_ent.get(s.id, [])[-2:]:
                cand.setdefault(o)
        for tid in list(cand):
            if tid in newer:
                cand.setdefault(newer[tid])
        per_sleeve[s.id] = list(cand)
        for tid in cand:
            reached.setdefault(tid, set()).add(s.id)
            best[tid] = max(best.get(tid, 0.0), float(rrf[order[tid]]))
    dropped = []
    if anchor_cos is not None:
        qv = arc.encoder.one(question)
        for tid in list(reached):
            if float(arc.emb[order[tid]] @ qv) < anchor_cos:
                dropped.append(tid)
                del reached[tid]
    for tid in exclude:
        reached.pop(tid, None)
    for sid in per_sleeve:
        per_sleeve[sid] = [t for t in per_sleeve[sid] if t in reached]
    themes = {t: ss | (prev_reached or {}).get(t, set()) for t, ss in reached.items()}
    score = {}
    for tid, ss in themes.items():
        w = 1.0 if arc.by_id[tid].origin != "model" else 0.6  # как в V6
        b = bonus * (len(ss) - 1)
        if bonus_norm == "ents":  # «узел», где упомянуто всё, не выигрывает за счёт длины
            b /= max(1, len(arc.ents[tid]))
        score[tid] = w * (best[tid] + b)
    ranked = sorted(score, key=lambda t: (-score[t], order[t]))
    return {
        "sleeves": sleeves,
        "ranked": ranked,
        "reached": themes,
        "intersections": [t for t in ranked if len(themes[t]) >= 2],
        "per_sleeve": per_sleeve,
        "newer": newer,
        "k_per_sleeve": k_s,
        "dropped_by_anchor": dropped,
        "score": score,
        "best": best,
    }


def marks(arc: Archive, tid: str, c: dict) -> str:
    """Пометки строки: пересечение тем (названия рукавов) и «позже заменено», как у V6."""
    out = ""
    names = {s.id: s.surface for s in c["sleeves"]}
    ss = sorted(c["reached"].get(tid, ()))
    if len(ss) >= 2:
        out += " [связывает темы: " + " и ".join(f"«{names.get(x, x)}»" for x in ss) + "]"
    if tid in c["newer"]:
        out += " (позже заменено — см. более позднюю реплику)"
    return out


def v9q(
    arc: Archive,
    q: str,
    count: Callable[[str], int],
    budget: int,
    *,
    k: int = 8,
    full: int = 4,
    bonus: float = BONUS,
    anchor_cos: float | None = None,
    query_mode: str = "entity",
) -> tuple[str, dict]:
    """V9-q: один раунд по вопросу; подача как у V6 (full полных + ярлыки), тот же бюджет."""
    if not arc.turns:
        return "", {"n_sleeves": 0}
    c = candidates(arc, q, k=k, bonus=bonus, anchor_cos=anchor_cos, query_mode=query_mode)
    picked = S._pick_full(c, full)
    lines = [
        f"- [{_role(arc.by_id[t], trust=True)}]{marks(arc, t, c)} "
        f"{excerpt(arc.by_id[t].text, count, 110)}"
        for t in picked
    ]
    rest = [arc.by_id[t] for t in c["ranked"] if t not in picked]
    if rest:
        lines.append("Ещё по теме: " + "; ".join(excerpt(t.text, count, 12) for t in rest))
    log = {
        "sleeves": [{"id": s.id, "kind": s.kind, "query": s.query} for s in c["sleeves"]],
        "n_intersections": len(c["intersections"]),
        "k_per_sleeve": c["k_per_sleeve"],
        "dropped_by_anchor": len(c["dropped_by_anchor"]),
        "picked": picked,
        "pool": c["ranked"],
    }
    return _fit(lines, count, budget), log


def v9q_main(
    arc: Archive,
    q: str,
    count: Callable[[str], int],
    budget: int,
    *,
    k: int = 8,
    k_periph: int = 8,
    full: int = 4,
    reserve: int = 1,
    bonus: float = BONUS,
    bonus_norm: str | None = "ents",
    anchor_cos: float | None = None,
    query_mode: str = "clause",
) -> tuple[str, dict]:
    """V9-q, «главная линия + периферия» (поправка Егора после этапа A).

    Главная линия — пул V6 как есть (поиск по всему вопросу). Периферия — top-`k_periph` на
    каждый рукав (+ граф, пересечения, якорь). Пул — объединение без дублей. В память — `full`
    полных фрагментов, как у V6: `full − reserve` лучших из пула V6 и `reserve` мест — лучшим
    кандидатам периферии, которых нет в пуле V6 (если таких нет — места отдаются V6).
    Ярлыки — остальной пул V6, как у V6. Тот же `_fit` и бюджет.
    """
    if not arc.turns:
        return "", {"n_sleeves": 0}
    main, newer = M.v6_candidates(arc, q, k)
    c = candidates(arc, q, k=k, bonus=bonus, anchor_cos=anchor_cos, query_mode=query_mode,
                   k_per_sleeve=k_periph, bonus_norm=bonus_norm)  # fmt: skip
    periph = [t for t in c["ranked"] if t not in main]
    picked_p = periph[:reserve]
    picked = main[: full - len(picked_p)] + picked_p
    lines = [
        f"- [{_role(arc.by_id[t], trust=True)}]{marks(arc, t, c) if t in picked_p else ''}"
        f"{'' if t in picked_p or t not in newer else ' (позже заменено — см. более позднюю реплику)'} "
        f"{excerpt(arc.by_id[t].text, count, 110)}"
        for t in picked
    ]
    rest = [arc.by_id[t] for t in main if t not in picked]
    if rest:
        lines.append("Ещё по теме: " + "; ".join(excerpt(t.text, count, 12) for t in rest))
    log = {
        "sleeves": [{"id": s.id, "kind": s.kind, "query": s.query} for s in c["sleeves"]],
        "n_intersections": len(c["intersections"]),
        "intersections": c["intersections"],
        "dropped_by_anchor": len(c["dropped_by_anchor"]),
        "picked": picked,
        "picked_periphery": picked_p,
        "pool_main": main,
        "periphery": periph,
        "pool": list(dict.fromkeys(main + c["ranked"])),
    }
    return _fit(lines, count, budget), log


PERIPHERY_HEADER = (
    "Рядом есть (воспоминания и слабые ассоциации, не требования; используй, если относится "
    "к вопросу):"
)


def v9q_rest(
    arc: Archive,
    q: str,
    count: Callable[[str], int],
    budget: int,
    *,
    k: int = 8,
    k_periph: int = 8,
    item_tokens: int = 40,
    bonus: float = BONUS,
    bonus_norm: str | None = None,
    anchor_cos: float | None = None,
    query_mode: str = "clause",
) -> tuple[str, dict]:
    """V9-q3, «периферия без вытеснения»: блок V6 как есть, периферия — в остаток бюджета.

    Кандидаты периферии — рукава по `k_periph` вне пула V6, в порядке `candidates`; каждый —
    выдержка до `item_tokens` токенов; добавляются, пока весь блок укладывается в `budget`.
    """
    v6 = M.v6_hybrid(arc, q, count, budget, k)
    if not arc.turns:
        return v6, {"periphery_shown": []}
    main, _ = M.v6_candidates(arc, q, k)
    c = candidates(arc, q, k=k, bonus=bonus, anchor_cos=anchor_cos, query_mode=query_mode,
                   k_per_sleeve=k_periph, bonus_norm=bonus_norm)  # fmt: skip
    periph = [t for t in c["ranked"] if t not in main]
    lines = [v6] if v6 else [M.HEADER]
    shown, head = [], False
    for t in periph:
        ln = f"- [{_role(arc.by_id[t], trust=True)}] {excerpt(arc.by_id[t].text, count, item_tokens)}"
        extra = ([PERIPHERY_HEADER] if not head else []) + [ln]
        if count("\n".join(lines + extra)) > budget:
            continue
        lines += extra
        head = True
        shown.append(t)
    text = "\n".join(lines) if (v6 or shown) else ""
    return text, {"periphery_shown": shown, "periphery": periph, "pool_main": main,
                  "v6_tokens": count(v6) if v6 else 0}  # fmt: skip


# ---------------- рукава-фразы: связная группа слов по теме (без генеративной модели) ----------
_TOK = re.compile(r"[\w][\w\-.+#/]*\w|\w|[^\w\s]")
_LEFT = {"ADJF", "PRTF", "NUMR", "ADJS"}  # зависимые слева: «мобильное приложение»
_STOP_RIGHT = {"VERB", "INFN", "CONJ", "PRCL", "INTJ", "PRED", "ADVB", "GRND", "NPRO"}


def _tag(w: str):
    if not T.is_cyrillic(w.lower()):
        return None  # латиница/код: имя, часть группы
    return T._morph().parse(w.lower())[0].tag


def _is_word(w: str) -> bool:
    return bool(re.match(r"\w", w))


def _nw(words: list[str], a: int, b: int) -> int:
    return sum(_is_word(words[x]) for x in range(a, b + 1))


def phrase_sleeves(
    arc: Archive,
    text: str,
    *,
    max_sleeves: int = MAX_SLEEVES,
    max_len: int = 6,
    with_pp: bool = True,
) -> list[Sleeve]:
    """Рукав = именная группа вокруг редкой сущности текста (2–`max_len` слов).

    Правило (морфология pymorphy3, без парсера и генерации):
    1. Головы — сущности-понятия текста, которые есть в архиве и не частые (редкие первыми),
       при нехватке — редкие в архиве леммы (как `text_sleeves`).
    2. Группа головы: влево — прилагательные/причастия/числительные и латиница; вправо —
       существительные в родительном падеже с их прилагательными («переводы интерфейса»), при
       `with_pp` — один предложный оборот («для мобилки», «в монорепе»). Глагол, союз, знак
       препинания — граница. Не длиннее `max_len` слов.
    3. Группы, которые пересекаются, — один рукав. Не больше `max_sleeves`.
    """
    toks = [(m.group(), m.start()) for m in _TOK.finditer(text)]
    words = [w for w, _ in toks]
    tags = [_tag(w) if _is_word(w) else None for w in words]
    heads = text_sleeves(arc, text, max_sleeves=max_sleeves)
    spans: list[tuple[int, int, Sleeve]] = []
    for s in heads:
        hl = set(T.content_lemmas(s.surface)) or {s.id}
        idx = [i for i, w in enumerate(words) if _is_word(w) and (set(T.content_lemmas(w)) & hl)]
        if not idx:
            continue
        a = b = idx[0]
        while b + 1 < len(words) and b + 1 in idx:  # многословная сущность
            b += 1
        while a > 0 and _is_word(words[a - 1]) and _nw(words, a, b) < max_len:
            t = tags[a - 1]
            if t is None or (t.POS in _LEFT):
                a -= 1
            else:
                break
        j = b + 1
        while j < len(words) and _is_word(words[j]) and _nw(words, a, b) < max_len:
            t = tags[j]
            if t is None or (t.POS == "NOUN" and "gent" in t) or t.POS in _LEFT and j + 1 < len(words) and tags[j + 1] is not None \
                    and tags[j + 1].POS == "NOUN" and "gent" in tags[j + 1]:
                b = j
            elif with_pp and t.POS == "PREP" and j + 1 < len(words) and _is_word(words[j + 1]):
                k = j + 1
                while k < len(words) and _is_word(words[k]) and tags[k] is not None                         and tags[k].POS in _LEFT:  # fmt: skip
                    k += 1
                if k < len(words) and _is_word(words[k]) and (tags[k] is None or tags[k].POS == "NOUN") \
                        and sum(_is_word(words[x]) for x in range(a, k + 1)) <= max_len:  # fmt: skip
                    b = k
                    j = k
                break
            else:
                break
            j += 1
        spans.append((a, b, s))
    spans.sort(key=lambda x: x[0])
    merged: list[list] = []
    for a, b, s in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2].append(s)
        else:
            merged.append([a, b, [s]])
    out = []
    for a, b, ss in merged[:max_sleeves]:
        phrase = text[toks[a][1] : toks[b][1] + len(words[b])]
        key = ss[0].id
        out.append(Sleeve(key, ss[0].kind, phrase, phrase, set().union(*(x.lemmas for x in ss))))
    # порядок — как у голов: редкие первыми
    rank = {s.id: i for i, s in enumerate(heads)}
    return sorted(out, key=lambda s: rank.get(s.id, 99))
