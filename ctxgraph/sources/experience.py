"""Собственный опыт: сущности прошлых сессий пользователя и их совместная встречаемость (PMI).

Узел `exp:<каноническое имя>`. Сосед — сущность, встречавшаяся вместе с ним в одном фрагменте
не меньше чем в `min_count` **разных** прошлых сессиях (текущая сессия исключена, иначе меню
повторяет сам контекст). PMI считается по сессиям и должен быть > `min_pmi`.
Хабы гасятся: сущности из больше чем `hub_share` сессий не предлагаются.
"""

from __future__ import annotations

import math

from ctxgraph import text as T
from ctxgraph.sources.base import Link, Source
from ctxgraph.store import Node, Store

NOT_CONCEPT = {"VERB", "INFN", "ADVB", "GRND", "PRTF", "PRTS", "NUMR", "NPRO", "PRED"}
KIND_RU = {
    "keyphrase": "",
    "lemma": "",
    "identifier": "идентификатор",
    "path": "файл",
    "url": "ссылка",
    "date": "дата",
    "quantity": "величина",
    "code": "код",
}


#: Виды сущностей, которые могут быть узлами опыта. URL, даты и величины — нет.
CONCEPT_KINDS = {"keyphrase", "lemma", "identifier", "code", "path"}


def is_concept(canon: str, kind: str, surface: str | None = None) -> bool:
    """Годится ли сущность в узел опыта: термин или понятие, а не обрывок, служебное слово,
    слишком общее слово или кусок кода."""
    if kind not in CONCEPT_KINDS or len(canon) < 3:
        return False
    toks = canon.split()
    if canon in T.GENERIC or canon in T.STOP_COLUMNS or any(t in T.STOP_ALL for t in toks):
        return False
    english = kind in ("keyphrase", "lemma", "identifier") and not T.is_cyrillic(toks[-1])
    if english and not T.en_noun_ok(toks[-1]):
        return False  # английский глагол или наречие (run, add, deployed) — не понятие
    if kind in ("keyphrase", "lemma") and not T.is_noun_phrase(surface or canon):
        return False
    if T.is_cyrillic(canon) and kind == "lemma":
        tags = [T.pos(w) for w in toks]
        if "NOUN" not in tags or any(t in NOT_CONCEPT for t in tags):
            return False  # глаголы и одиночные прилагательные — не понятия
    return True


class ExperienceSource(Source):
    name = "experience"
    prefix = "exp:"

    def __init__(
        self, store: Store, *, hub_share: float = 0.3, min_pmi: float = 0.0, min_count: int = 2
    ) -> None:
        self.store = store
        self.hub_share = hub_share
        self.min_pmi = min_pmi
        self.min_count = min_count  # одно совместное появление — совпадение, не связь
        self.current_session: str | None = None

    def link(self, term: str, lang: str) -> list[Node]:
        if term in T.GENERIC or self.store.doc_freq(term, self.current_session) == 0:
            return []
        return [Node(f"exp:{term}", term, self.name, lang)]

    def neighbors(self, node: Node) -> list[Link]:
        a = node.id[4:]
        sess = self.current_session
        n = self.store.n_sessions(sess)
        df_a = self.store.session_freq(a, sess)
        if not n or not df_a:
            return []
        out: list[Link] = []
        for b, kind, c, surface in self.store.cooccurring_sessions(a, exclude_session=sess):
            if c < self.min_count or not is_concept(b, kind, surface):
                continue
            df_b = self.store.session_freq(b, sess)
            if n >= 10 and df_b / n > self.hub_share:
                continue
            pmi = math.log(c * n / (df_a * df_b)) if df_b else 0.0
            if pmi <= self.min_pmi:
                continue  # пара встречается не чаще, чем случайно
            w = 1 - math.exp(-max(pmi, 0.05) * c / 2)  # 0..1: выше PMI и чаще вместе — сильнее
            tag = KIND_RU.get(kind, "")
            rel = f"вместе в {c} сессиях{', ' + tag if tag else ''}"
            out.append(
                Link(Node(f"exp:{b}", surface or b, self.name), rel, "experience", w, self.name)
            )
        out.sort(key=lambda link: -link.weight)
        return out[:20]
