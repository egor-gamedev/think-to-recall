"""Базовая линия для V9 по готовым ответам E1c (без вызовов API).

    python scripts/e1c_v9_baseline.py   # → reports/e1c/main_final/v9_baseline.md

1. Полнота извлечения (ctxgraph.e1b.coverage) для V1/V2/V6/V7 по четырём типам на 200 новых вопросах:
   какие опорные факты вопроса были в блоке памяти полным фрагментом.
2. «Последнее решение по сущности» (ctxgraph.e1b.decisions) на 60 вопросах «менялось»:
   архив — реплики загрузки; сущности — из текста вопроса (как у V6). Попадание — среди
   возвращённых реплик есть помеченная актуальным эталонным фактом (конец цепочки `replaces`
   среди опорных). Ложное «последнее решение» — возвращённая реплика без актуального опорного факта
   (с устаревшим опорным, с другим фактом проекта или без факта).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_e1c as E  # noqa: E402

from ctxgraph.e1b import coverage as C  # noqa: E402
from ctxgraph.e1b import decisions as D  # noqa: E402
from ctxgraph.e1b import memory as M  # noqa: E402
from ctxgraph.e1b.chat import Turn  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.embed import HashingEncoder  # noqa: E402
from ctxgraph.text import approx_tokens  # noqa: E402

OUT = ROOT / "reports" / "e1c" / "main_final" / "v9_baseline.md"
TYPES = [("recall", "вспомнить"), ("link", "связать"), ("change", "менялось"), ("clarify", "уточнить")]
VARIANTS = (1, 2, 6, 7)


def coverage_table(A) -> list[str]:
    cell = {(v, ty): Counter() for v in VARIANTS for ty, _ in TYPES + [("all", "")]}
    for pid in E.NEW:
        spec, ft, text = C.load_turns(ROOT, pid)
        for q in spec["questions"]:
            for v in VARIANTS:
                r = C.anchor_coverage(q["anchors"], ft, text, A[f"{pid}|{q['id']}|{v}"]["memory"])
                for ty in (q["type"], "all"):
                    c = cell[(v, ty)]
                    c["q"] += 1
                    c["in"] += r["n"]
                    c["need"] += r["need"]
                    c["share"] += r["share"]
                    c["all"] += r["all"]
                    c["zero"] += r["n"] == 0
    L = ["## 1. Полнота извлечения: опорные факты в блоке памяти полным фрагментом\n",
         "Клетка: доля фактов (в памяти / нужно) · вопросов, где в памяти все опорные факты · где ни одного.\n",
         "| | " + " | ".join(f"{ru} ({cell[(1, ty)]['q']})" for ty, ru in TYPES) + " | все (200) |",
         "|---|" + "---|" * (len(TYPES) + 1)]  # fmt: skip
    for v in VARIANTS:
        row = []
        for ty, _ in TYPES + [("all", "")]:
            c = cell[(v, ty)]
            row.append(
                f"{100 * c['in'] / c['need']:.0f} % ({c['in']}/{c['need']}) · все {c['all']} · ни одного {c['zero']}"
            )
        L.append(f"| V{v} | " + " | ".join(row) + " |")
    L.append(
        "\nСредняя доля по вопросам (а не по фактам): "
        + "; ".join(
            f"V{v} " + ", ".join(f"{ru} {100 * cell[(v, ty)]['share'] / cell[(v, ty)]['q']:.0f} %" for ty, ru in TYPES)
            for v in VARIANTS
        )
        + ".\n"
    )  # fmt: skip
    return L


def successors(facts: dict) -> set[str]:
    """Факты, которые чем-то заменены (устаревшие)."""
    return {f["replaces"] for f in facts.values() if "replaces" in f}


def decisions_explore(A) -> list[str]:
    modes = {
        "решение (маркер в предложении с сущностью) + «заменяет»": {},
        "решение (маркер где угодно в реплике) + «заменяет»": {"same_sentence": False},
        "просто последняя реплика пользователя + «заменяет»": {"decision_only": False},
    }
    st = {m: Counter() for m in modes}
    ceiling = Counter()

    for pid in E.NEW:
        spec, turns = load_project(ROOT, pid)
        _, ft, text = C.load_turns(ROOT, pid)
        arc = M.Archive(
            [Turn(t["id"], t["role"], t["text"], "load", approx_tokens(t["text"])) for t in turns],
            HashingEncoder(),
        )
        stale = successors(spec["facts"])
        tags = {}
        for f, ts in ft.items():
            for t in ts:
                tags.setdefault(t, set()).add(f)
        for q in spec["questions"]:
            if q["type"] != "change":
                continue
            current = [f for f in q["anchors"] if f not in stale]
            cur_turns = {t for f in current for t in ft.get(f, [])}
            ents = M.question_anchors(arc, q["text"])
            reach = {t for e in ents for t in arc.by_ent[e] if arc.by_id[t].role == "user"}
            ceiling["потолок: актуальная реплика содержит сущность вопроса"] += bool(reach & cur_turns)
            mem6 = A[f"{pid}|{q['id']}|6"]["memory"]
            in6 = C.anchor_coverage(current, ft, text, mem6)["n"] > 0
            ceiling["для сравнения: актуальный факт был в памяти V6 полным фрагментом"] += in6
            for m, kw in modes.items():
                got = D.latest_decisions(arc, q["text"], **kw)
                ret = list(dict.fromkeys(got.values()))
                s = st[m]
                s["вопросов"] += 1
                s["сущностей в вопросе найдено"] += len(ents)
                s["вопросов без сущностей"] += not ents
                s["вопросов без ответа функции"] += not ret
                hit = bool(set(ret) & cur_turns)
                s["  из попаданий — актуального факта не было в памяти V6"] += hit and not in6
                s["попадание: актуальный факт найден"] += hit
                s["возвращено реплик"] += len(ret)
                for t in ret:
                    fs = tags.get(t, set())
                    if t in cur_turns:
                        s["  верных (актуальный опорный факт)"] += 1
                    elif fs & set(q["anchors"]):
                        s["  ложных: устаревший опорный факт"] += 1
                    elif fs - stale:
                        s["  ложных: другой действующий факт проекта"] += 1
                    elif fs:
                        s["  ложных: другой устаревший факт"] += 1
                    else:
                        s["  ложных: реплика без факта"] += 1
    keys = ["вопросов", "вопросов без сущностей", "вопросов без ответа функции", "попадание: актуальный факт найден", "  из попаданий — актуального факта не было в памяти V6",
            "возвращено реплик", "  верных (актуальный опорный факт)", "  ложных: устаревший опорный факт",
            "  ложных: другой действующий факт проекта", "  ложных: другой устаревший факт", "  ложных: реплика без факта"]  # fmt: skip
    ms = list(modes)
    L = ["## 2. «Последнее решение по сущности» на 60 вопросах «менялось» (разведка)\n",
         "Архив — реплики загрузки; сущности — из текста вопроса (до 4, как у V6); на сущность — одна реплика.\n",
         "| | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]  # fmt: skip
    L += [f"| {k} | " + " | ".join(str(st[m][k]) for m in ms) + " |" for k in keys]
    L += [f"\n{k}: {n} из 60." for k, n in ceiling.items()]
    return L


def main() -> None:
    A = E.answers(E.NEW)
    L = ["# E1c → V9: базовая линия (разведка по готовым ответам, без вызовов API)\n"]
    L += coverage_table(A)
    L += decisions_explore(A)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
