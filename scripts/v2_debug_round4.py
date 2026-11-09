"""Debug round 4 (<5 min): threshold vs fill-to-the-end vs hybrid on the post-quantum example.

    python scripts/v2_debug_round4.py   # -> reports/v2_debug_round4.md

For the main block and the periphery: how many of the 7 relevant facts are there and how many lines
carry none of them ("lines without a needed fact" — not necessarily junk). Threshold = config
`min_score` (provisional, not tuned on this example). Debug only — no quality claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_debug_loss as LO  # noqa: E402
import v2_debug_pq as P  # noqa: E402

from ctxgraph.e1b.coverage import load_turns  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODES = [("cutoff", "порог"), ("fill", "заполнение до конца"), ("hybrid", "гибрид")]


def main() -> None:
    from ctxgraph.e1b.backends import E5Encoder
    from ctxgraph.e1b.selection import Reranker

    enc, count = E5Encoder(), P.counter()
    spec, ft, text = load_turns(ROOT, P.PID)
    g = Graph(Store(), enc, config=Config())
    for t in load_project(ROOT, P.PID)[1]:
        g.add_episode(t["text"], t["role"], "s1", eid=t["id"])
    rr = Reranker(revision=REV)
    rr.persist(ROOT / ".cache" / "v2_rerank.jsonl")
    ev = {k: set(LO.frags_for(g, s)) for k, s in LO.evidence(spec, ft, text).items()}
    need_frags = set().union(*ev.values())

    def tally(units: list[set[str]]) -> tuple[int, int, int]:
        got = {k for k, fs in ev.items() if any(fs & u for u in units)}
        return len(got), sum(not (u & need_frags) for u in units), len(units)

    L = ["# Отладка v2, раунд 4: порог / заполнение / гибрид (≤ 5 мин, без выводов о качестве)\n",
         f"Вопрос: «{P.Q}». Шаг: «{P.STEP}». Порог — параметр `Config.min_score` = {g.cfg.min_score} "
         "(временный, по этому примеру не подбирался; выбор — на чужом бенчмарке). Переранжирование: bge-reranker-v2-m3, "
         "запрос = вопрос + шаг. Бюджет основного блока 600 токенов, периферии рукавов 200.\n",
         "«Строк без нужного факта» — строки, где нет ни одного из 7 нужных фактов (это не обязательно мусор).\n",
         "| режим | основной блок: фактов из 7 | основной блок: строк без нужного факта / всего | периферия: фактов из 7 | "
         "периферия: строк без нужного факта / всего | всего фактов (без повторов) |", "|---|---|---|---|---|---|"]  # fmt: skip
    blocks = {}
    for mode, ru in MODES:
        tr = SL.sleeves_trace(g, P.STEP, P.Q, count, 200, anchor_cos=0.76, scorer=rr.score, mode=mode)
        main_units = [set(x["fragments"]) for x in tr["final_lines"]]
        per_units = [{it["fragment"]} for it in tr["periphery_items"]]
        fm, jm, nm = tally(main_units)
        fp, jp, npr = tally(per_units)
        both = {k for k, fs in ev.items() if any(fs & u for u in main_units + per_units)}
        L.append(f"| {ru} | {fm} | {jm} / {nm} | {fp} | {jp} / {npr} | {len(both)} |")
        blocks[mode] = tr
    h = blocks["hybrid"]
    L += ["\n## Гибрид: что получает модель\n", "### Основной блок (выше порога)\n", "```text", h["final"] or "(пусто)", "```",
          "\n### Периферия (рукава + ниже порога с пометкой «less certain, optional»)\n", "```text", h["periphery"] or "(пусто)", "```"]  # fmt: skip
    (ROOT / "reports" / "v2_debug_round4.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
