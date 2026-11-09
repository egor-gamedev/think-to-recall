"""Детектор «заменяет» v2 (E1c): высокая точность, неуверенные смены не помечаются.

Реплика пользователя (из загрузки или follow-up, не вопрос) → предложение с явной сменой решения →
из него вырезается СТАРОЕ (X): «перешли с X на …», «вместо X», «от X отказались», «X убираем»,
«X заменили», «передумал по X», «теперь не X, а …». Ребро ставится к самой ранней реплике
пользователя, где встречается X (по сущностям и леммам), при равенстве — с большим совпадением.
Нет X, нет такой реплики, предложение-вопрос или отказ от идеи, которую не принимали, — ребра нет.

Сделано на диалогах E1b и E1c: по эталону скелетов точность 8/10, полнота 8/29 (смены без явной
формулировки не ловятся — сознательно). Приёмка — ручная разметка Егора (≥ 80 %).
"""

from __future__ import annotations

import re

from ctxgraph import text as T

STRONG_OLD = [
    re.compile(p, re.I)
    for p in (
        r"(?<!\w)(?:перешл\w*|переход\w*|перехож\w*|перейд\w*|ушли|уходим|переехал\w*)\s+с\s+(.{2,50}?)\s+на\s",
        r"(?<!\w)вместо\s+(.{2,50}?)(?=\s*(?:[,.;:—]|$|\s+(?:берём|будет|ставим|используем|делаем)))",
        r"(?<!\w)от\s+(.{2,50}?)\s+отказ\w*",
        r"отказ\w*\s+от\s+(.{2,50}?)(?=\s*(?:[,.;:—!]|$))",
        r"(?:^|[,.;:—]\s*)([^,.;:—]{2,50}?)\s+(?:бросаем|убираем|выкидываем|закрыли|отменяем|больше не используем|больше не нужн\w*)",
        r"замени\w*\s+(.{2,50}?)\s+на\s",
        r"(?:^|[,.;:—]\s*)([^,.;:—]{2,50}?)\s+замени\w*",
        r"передума\w*\s+(?:по|насчёт|насчет|про|с)\s+(.{2,40}?)(?=\s*(?:[,.;:—!]|$))",
        r"(?<!\w)теперь\s+не\s+(.{2,40}?),\s+а\s",
    )
]
#: Отказ от идеи, которую не принимали («думал про X, но отказались»), — не смена решения.
IDEA = re.compile(
    r"(думал|обсуждали|предлага\w*|советова\w*|спрашивал|хотел)[^.!?]{0,80}(отказ\w*)"
    r"|отказ\w*\s+делать|отказал\w*\s+от\s+иде",
    re.I,
)
SENT = re.compile(r"[^.!?\n]+[.!?]?")
QUESTION = re.compile(r"\?\s*$")
SOURCES = ("load", "followup")  # вопросы прогона и ответы модели решений не меняют


def _keys(text: str, concepts) -> set[str]:
    ks = {c for c, _ in concepts(text)} | set(T.content_lemmas(text))
    return {k for k in ks if len(k) > 2}


_LEAD = re.compile(r"^(?:а|и|но|да|ещё|еще|теперь|так что|значит)\s+", re.I)


def old_span(sentence: str) -> str | None:
    """Старое (X) из предложения со сменой: без вводной части до двоеточия и союзов в начале."""
    for p in STRONG_OLD:
        m = p.search(sentence)
        if m:
            x = m.group(1).split(":")[-1].strip(' «»"—,')
            while _LEAD.match(x):
                x = _LEAD.sub("", x, count=1)
            return x or None
    return None


def change_note(text: str, sentence: str, max_words: int = 40) -> str:
    """Чем заменили: предложение со сменой и следующее за ним, до max_words слов."""
    sents = [x.strip() for x in SENT.findall(text) if x.strip()]
    k = next((i for i, x in enumerate(sents) if x == sentence.strip()), None)
    picked = sents[k : k + 2] if k is not None else [sentence.strip()]
    words = " ".join(picked).split()
    return " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")


def detect(turns, concepts) -> list[dict]:
    """turns: объекты с .id, .role, .text, .origin. concepts(text) → [(canonical, surface)]."""
    tk = [_keys(t.text, concepts) for t in turns]
    edges = []
    for i, t in enumerate(turns):
        if t.role != "user" or getattr(t, "origin", "load") not in SOURCES:
            continue
        for s in SENT.findall(t.text):
            if QUESTION.search(s) or IDEA.search(s):
                continue
            x = old_span(s)
            xk = _keys(x, concepts) if x else set()
            if not xk:
                continue
            cands = [
                j
                for j in range(i)
                if turns[j].role == "user"
                and getattr(turns[j], "origin", "load") in SOURCES
                and xk & tk[j]
            ]
            if not cands:
                continue
            j = max(
                cands, key=lambda j: (len(xk & tk[j]), -j)
            )  # старое решение — первое упоминание X
            edges.append(
                {
                    "new": t.id,
                    "old": turns[j].id,
                    "x": x,
                    "shared": sorted(xk & tk[j]),
                    "sentence": s.strip(),
                    "note": change_note(t.text, s),
                }
            )
            break
    return edges
