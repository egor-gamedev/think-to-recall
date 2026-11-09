"""Выборка рёбер «заменяет» (детектор v2) на ручную разметку Егора: все рёбра на диалогах E1b и E1c.

    python scripts/e1c_replaces_sample.py   # → reports/e1c/replaces_labeling.csv и .md

Разметка: колонка `верно (1/0)` — 1, если новая реплика действительно отменяет или меняет решение,
сказанное в старой реплике. Приёмка — точность ≥ 80 %.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ctxgraph.e1b.chat import Turn  # noqa: E402
from ctxgraph.e1b.memory import _concepts  # noqa: E402
from ctxgraph.e1b.replaces import detect  # noqa: E402

PROJECTS = [("benchmark_e1b", p) for p in ("crypto-wallet", "django-shop", "rnaseq-thesis")] + [
    ("benchmark_e1c", p) for p in ("greenhouse-iot", "clinic-bot", "econ-thesis", "indie-game")
]


def turns_of(root: str, pid: str) -> list[Turn]:
    pdir = ROOT / root / "projects" / pid
    spec = yaml.safe_load((pdir / "spec.yaml").read_text(encoding="utf-8"))
    out = []
    for seg in spec["segments"]:
        d = yaml.safe_load((pdir / "load" / f"{seg['id']}.yaml").read_text(encoding="utf-8"))
        out += [
            Turn(f"{seg['id']}.{i}", t["role"], t["text"].strip(), "load", 0)
            for i, t in enumerate(d["turns"])
        ]
    return out


def main() -> None:
    rows = []
    for root, pid in PROJECTS:
        ts = turns_of(root, pid)
        by = {t.id: t for t in ts}
        for e in detect(ts, _concepts):
            rows.append({"id": f"R{len(rows) + 1:02d}", "проект": pid, "новая": e["new"], "старая": e["old"],
                         "что заменено (X)": e["x"], "предложение со сменой": e["sentence"],
                    "на что заменили (в памяти)": e["note"],
                         "текст старой реплики": by[e["old"]].text, "верно (1/0)": "", "комментарий": ""})  # fmt: skip
    out = ROOT / "reports" / "e1c"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replaces_labeling.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [
        "# Рёбра «заменяет» — разметка, раунд 2\n",
        "Вопрос один: **новая реплика отменяет или меняет решение, которое сказано в старой реплике?** "
        "1 — да, 0 — нет. Поле «что заменено» больше не оцениваем: теперь в памяти рядом с устаревшей "
        "репликой показывается, на что её заменили (строка «В памяти будет»).\n",
    ]
    for r in rows:
        note = r["на что заменили (в памяти)"]
        md += [
            f"\n## {r['id']} · {r['проект']}\n",
            f"**Старая реплика ({r['старая']}):** {r['текст старой реплики'][:700]}\n",
            f"**Новая реплика ({r['новая']}):** {note}\n",
            f"*В памяти будет:* старая реплика с пометкой «позже заменено: «{note}»»\n",
            "**Отменяет/меняет решение из старой? 1 / 0**\n",
        ]
    (out / "replaces_labeling.md").write_text("\n".join(md), encoding="utf-8")
    print(f"рёбер: {len(rows)} → {out / 'replaces_labeling.md'}")


if __name__ == "__main__":
    main()
