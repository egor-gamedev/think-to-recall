"""Полнота извлечения: какие опорные факты вопроса попали в память полным фрагментом.

Факт считается попавшим, если хотя бы одна реплика загрузки, где он помечен, есть в памяти
полным фрагментом (строка «- [роль] …» блока памяти) или передана явно по id (многошаговая
память V9 знает id показанных реплик). Для многошаговой памяти — объединение по шагам.

Сопоставление текстом: первые 50 символов нормализованной реплики ищутся в нормализованных
строках полных фрагментов (как в разведке E1c). Код в ```…``` при нормализации выбрасывается.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import yaml

FULL_KEY = 50
LABEL_KEY = 25


def norm(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    return re.sub(r"\W+", " ", s.lower()).strip()


def full_part(memory: str) -> str:
    """Нормализованный текст только полных фрагментов блока памяти."""
    return norm("\n".join(ln for ln in memory.splitlines() if ln.startswith("- [")))


def in_memory(turn_text: str, memory: str) -> str:
    """'full' — реплика есть полным фрагментом; 'label' — только ярлыком; '' — нет."""
    key = norm(turn_text)[:60]
    if not key:
        return ""
    if key[:FULL_KEY] in full_part(memory):
        return "full"
    if key[:LABEL_KEY] in norm(memory):
        return "label"
    return ""


def anchor_coverage(
    anchors: Iterable[str],
    fact_turns: dict[str, list[str]],
    turn_text: dict[str, str],
    memories: str | Iterable[str] = (),
    shown_ids: Iterable[str] = (),
) -> dict:
    """Покрытие опорных фактов памятью (объединение по всем шагам).

    anchors — опорные факты вопроса; fact_turns — факт → id реплик загрузки, где он помечен;
    turn_text — id реплики → текст; memories — блок памяти или блоки по шагам;
    shown_ids — id реплик, показанных полным фрагментом (если известны).
    """
    anchors = list(dict.fromkeys(anchors))
    if isinstance(memories, str):
        memories = [memories]
    fulls = [full_part(m) for m in memories if m]
    shown = set(shown_ids)

    def turn_in(tid: str) -> bool:
        if tid in shown:
            return True
        key = norm(turn_text.get(tid, ""))[:FULL_KEY]
        return bool(key) and any(key in f for f in fulls)

    got = [f for f in anchors if any(turn_in(t) for t in fact_turns.get(f, []))]
    n = len(anchors)
    return {
        "in": got,
        "missing": [f for f in anchors if f not in got],
        "n": len(got),
        "need": n,
        "share": len(got) / n if n else 0.0,
        "all": n > 0 and len(got) == n,
    }


def load_turns(root: Path, pid: str) -> tuple[dict, dict, dict]:
    """spec проекта; факт → id реплик загрузки, где он помечен; id реплики → текст.

    id реплик — как в архиве прогона (runner.load_project): «S1.0», «S1.1», …"""
    from ctxgraph.e1b.runner import project_dir

    base = project_dir(Path(root), pid)
    spec = yaml.safe_load((base / "spec.yaml").read_text("utf-8"))
    fact_turns, text = defaultdict(list), {}
    for seg in spec["segments"]:
        d = yaml.safe_load((base / "load" / f"{seg['id']}.yaml").read_text("utf-8"))
        for i, t in enumerate(d["turns"]):
            tid = f"{seg['id']}.{i}"
            text[tid] = t["text"]
            for f in t.get("facts") or []:
                fact_turns[f].append(tid)
    return spec, dict(fact_turns), text


# ---------------- уровень факта: в памяти есть предложение, которое несёт факт ----------------
_CODE = re.compile(r"```.*?```", re.S)
_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Предложения реплики: код выброшен, строки списков — отдельные предложения."""
    plain = _CODE.sub("\n", text)
    out = []
    for s in _SPLIT.split(plain):
        s = _BULLET.sub("", s).strip()
        if len(s.split()) >= min_words:
            out.append(s)
    return out


def fact_sentences(
    fact_text: str, turns: list[str], *, rel: float = 0.75, floor: float = 0.25
) -> list[str]:
    """Предложения помеченных реплик, которые несут факт: доля лемм текста факта в предложении
    ≥ `floor` и не меньше `rel` от лучшего предложения."""
    from ctxgraph import text as T

    fl = set(T.content_lemmas(fact_text))
    if not fl:
        return []
    scored = []
    for t in turns:
        for s in split_sentences(t):
            sc = len(fl & set(T.content_lemmas(s))) / len(fl)
            scored.append((sc, s))
    if not scored:
        return []
    best = max(sc for sc, _ in scored)
    return [s for sc, s in scored if sc >= floor and sc >= rel * best]


def fact_coverage(
    anchors: Iterable[str],
    facts: dict[str, dict],
    fact_turns: dict[str, list[str]],
    turn_text: dict[str, str],
    memories: str | Iterable[str] = (),
) -> dict:
    """Опорный факт засчитан, если хотя бы одно предложение, несущее его (`fact_sentences`),
    есть в памяти целиком (после нормализации). Начало реплики без самого факта не считается."""
    anchors = list(dict.fromkeys(anchors))
    if isinstance(memories, str):
        memories = [memories]
    mem = [norm(m) for m in memories if m]
    got, evidence = [], {}
    for f in anchors:
        sents = fact_sentences(facts[f]["text"], [turn_text[t] for t in fact_turns.get(f, [])])
        hit = next((s for s in sents if (k := norm(s)) and any(k in m for m in mem)), None)
        evidence[f] = {"sentences": sents, "hit": hit}
        if hit:
            got.append(f)
    n = len(anchors)
    return {
        "in": got,
        "missing": [f for f in anchors if f not in got],
        "n": len(got),
        "need": n,
        "share": len(got) / n if n else 0.0,
        "all": n > 0 and len(got) == n,
        "evidence": evidence,
    }
