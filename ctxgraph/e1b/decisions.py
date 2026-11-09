"""«Последнее решение по сущности»: самая поздняя реплика пользователя с решением по ней.

Подаётся в память с пометкой «актуально», чтобы перебивать устаревшее из резюме.

Реплика-решение — реплика пользователя (загрузка или follow-up), где сущность встречается в
предложении-утверждении (не вопросе) с маркером решения («решили», «будем», «берём», «оставляем»,
«перешли», «вместо» …), либо реплика, которая по детектору «заменяет» отменяет более раннюю или отменена более поздней.
Учёт «заменяет»: реплика, которую заменила более поздняя, не может быть последним решением;
вместо неё берётся конец цепочки замен, даже если в нём сущность не названа.
"""

from __future__ import annotations

import re

from ctxgraph.e1b import memory as M
from ctxgraph.e1b.replaces import QUESTION, SENT, SOURCES

DECISION = re.compile(
    r"(?<!\w)(?:"
    r"решил\w*|решено|договорил\w*|остановил\w*|выбрал\w*|выбираем|"
    r"будем|будет|буду|делаем|сделаем|делаю|берём|берем|возьм\w*|ставим|поставил\w*|"
    r"оставля\w*|оставил\w*|используем|пользуемся|"
    r"перешл\w*|переход\w*|перехож\w*|перейд\w*|переехал\w*|"
    r"вместо|отказал\w*|отказываемся|убираем|бросаем|замени\w*|передума\w*|"
    r"теперь|итог\w*|окончательно|финально|"
    r"нужно|надо|должн\w*|планир\w*|хочу|хотим"
    r")(?!\w)",
    re.I,
)


def _replaced_by(arc: M.Archive) -> dict[str, str]:
    """Ранняя реплика → заменившая её (при нескольких — самая поздняя)."""
    out: dict[str, str] = {}
    for new in sorted(arc.replaces, key=arc.order.__getitem__):
        for old in arc.replaces[new]:
            out[old] = new
    return out


def chain_end(arc: M.Archive, tid: str, replaced_by: dict[str, str] | None = None) -> str:
    rb = _replaced_by(arc) if replaced_by is None else replaced_by
    seen = {tid}
    while tid in rb and rb[tid] not in seen:
        tid = rb[tid]
        seen.add(tid)
    return tid


def is_decision(
    arc: M.Archive,
    tid: str,
    entity: str,
    replaced_by: dict[str, str] | None = None,
    same_sentence: bool = True,
) -> bool:
    t = arc.by_id[tid]
    if t.role != "user" or getattr(t, "origin", "load") not in SOURCES:
        return False
    rb = _replaced_by(arc) if replaced_by is None else replaced_by
    if tid in arc.replaces or tid in rb:  # заменяет или заменена — значит, была решением
        return True
    for s in SENT.findall(t.text):
        if QUESTION.search(s) or not DECISION.search(s):
            continue
        if not same_sentence or entity in {c for c, _ in M._concepts(s)}:
            return True
    return False


def latest_decision(
    arc: M.Archive, entity: str, *, decision_only: bool = True, same_sentence: bool = True
) -> str | None:
    """id самой поздней реплики-решения по сущности (с учётом «заменяет») или None.

    decision_only=False — просто самая поздняя реплика пользователя с сущностью (для сравнения);
    same_sentence=False — маркер решения в любом предложении реплики, не обязательно с сущностью.
    """
    rb = _replaced_by(arc)
    cands = []
    for tid in arc.by_ent.get(entity, []):
        t = arc.by_id[tid]
        if t.role != "user" or getattr(t, "origin", "load") not in SOURCES:
            continue
        if decision_only and not is_decision(arc, tid, entity, rb, same_sentence):
            continue
        cands.append(chain_end(arc, tid, rb))
    if not cands:
        return None
    return max(cands, key=arc.order.__getitem__)


def latest_decisions(arc: M.Archive, question: str, limit: int = 4, **kw) -> dict[str, str]:
    """Сущности вопроса (как у V6) → последнее решение по каждой."""
    out = {}
    for e in M.question_anchors(arc, question, limit):
        tid = latest_decision(arc, e, **kw)
        if tid:
            out[e] = tid
    return out


def render_current(arc: M.Archive, decisions: dict[str, str], count, max_tokens: int = 110) -> str:
    """Блок «актуально»: по одной строке на реплику, повторы склеены."""
    lines, seen = [], set()
    for e, tid in decisions.items():
        if tid in seen:
            continue
        seen.add(tid)
        lines.append(
            f"- [актуально: {arc.surface.get(e, e)}] {M.excerpt(arc.by_id[tid].text, count, max_tokens)}"
        )
    return "\n".join(lines)
