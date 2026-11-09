"""E1c: выгрузка 40 случайных выборов V7 на ручную проверку Егором (вопрос, меню, что выбрано).

    python scripts/e1c_selection_sample.py <папка прогона>   # → <папка>/selection_review.md и .csv

Берутся ответы V7 на новых проектах E1c (основной слой), включая «НИЧЕГО» и нарушения формата;
seed 0. Для каждого пункта Егор отмечает «относится к вопросу» 1/0 и «упущено важное» да/нет.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

NEW = ("greenhouse-iot", "clinic-bot", "econ-thesis", "indie-game")


def main(run: Path, n: int = 40, seed: int = 0) -> None:
    recs = []
    for pid in NEW:
        f = run / f"{pid}.jsonl"
        if f.exists():
            recs += [
                r
                for r in map(json.loads, f.read_text("utf-8").splitlines())
                if r.get("variant") == 7
            ]
    pick = sorted(
        random.Random(seed).sample(recs, min(n, len(recs))),
        key=lambda r: (r["project"], r["question"]),
    )
    md = [f"# Выборы V7 на проверку ({len(pick)} из {len(recs)}, seed {seed})\n",
          "Для каждого выбранного пункта: относится к вопросу — 1/0. И в целом: упущен ли важный пункт меню — да/нет.\n"]  # fmt: skip
    rows = []
    for i, r in enumerate(pick, 1):
        s = r.get("selection") or {}
        md += [f"\n## S{i:02d} · {r['project']} {r['question']} ({r['type']})\n",
               f"**Вопрос:** {r.get('question_text', '')}\n" if r.get("question_text") else "",
               f"**Меню, как его видела модель:**\n\n```\n{s.get('menu_text', '(пустое меню)')}\n```\n",
               f"**Ответ модели на шаге выбора:** `{s.get('raw', '')}` → выбрано {s.get('picked', [])}"
               + ("" if s.get("format_ok", True) else " (НАРУШЕН ФОРМАТ → «ничего»)") + "\n",
               f"**Раскрыто в памяти:**\n\n{r.get('memory') or '(ничего)'}\n"]  # fmt: skip
        rows.append({"id": f"S{i:02d}", "проект": r["project"], "вопрос": r["question"], "тип": r["type"],
                     "выбрано": ",".join(map(str, s.get("picked", []))), "формат_ок": s.get("format_ok"),
                     "относится (по пунктам 1/0)": "", "упущено важное (да/нет)": "", "комментарий": ""})  # fmt: skip
    (run / "selection_review.md").write_text("\n".join(x for x in md if x), encoding="utf-8")
    with (run / "selection_review.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(run / "selection_review.md", len(pick))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
