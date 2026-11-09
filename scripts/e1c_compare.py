"""E1c: сводная таблица проектов (сопоставимость по трудности) и выборка вопросов для Егора.

python scripts/e1c_compare.py            # таблица → stdout и reports/e1c/projects_table.md
python scripts/e1c_compare.py sample     # по 10 случайных вопросов из проекта → reports/e1c/review_sample.md
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e1b_validate import chain  # noqa: E402

PROJ = ROOT / "benchmark_e1c" / "projects"
OUT = ROOT / "reports" / "e1c"
TYPES = ("recall", "link", "change", "clarify")


def qwen_count():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))
    return lambda s: len(tok.encode(s).ids)


def load(pdir: Path):
    spec = yaml.safe_load((pdir / "spec.yaml").read_text(encoding="utf-8"))
    turns = []
    for seg in spec["segments"]:
        d = yaml.safe_load((pdir / "load" / f"{seg['id']}.yaml").read_text(encoding="utf-8"))
        turns += [t["text"] for t in d["turns"]]
    return spec, turns


def row(spec: dict, turns: list[str], count) -> dict:
    facts, qs = spec["facts"], spec["questions"]
    first = {f: i for i, s in enumerate(spec["segments"]) for f in s["facts"]}
    ends = [f for f in facts if len(chain(facts, f)) >= 3
            and not any(g.get("replaces") == f for g in facts.values())]  # fmt: skip
    types = Counter(q["type"] for q in qs)
    return {
        "проект": spec["id"],
        "реплик": len(turns),
        "токенов (Qwen)": sum(count(t) for t in turns),
        "фактов": len(facts),
        "неявных": sum(bool(f.get("implicit")) for f in facts.values()),
        "смен (replaces)": sum("replaces" in f for f in facts.values()),
        "цепочек ≥3": len(ends),
        "отказов": sum(f.endswith("r") for f in facts),
        "вопросов": len(qs),
        **{t: types[t] for t in TYPES},
        "связать ≥3 сегм.": sum(q["type"] == "link" and len({first.get(a) for a in q["anchors"]}) >= 3 for q in qs),
        "опор на вопрос": round(sum(len(q["anchors"]) for q in qs) / max(len(qs), 1), 1),
        "follow-up/удерж.": f"{sum(bool(q.get('followup')) for q in qs)}/{sum(bool(q.get('retention_of')) for q in qs)}",
    }  # fmt: skip


def table() -> None:
    count = qwen_count()
    rows = [row(*load(p), count) for p in sorted(PROJ.iterdir()) if (p / "spec.yaml").exists()]
    if not rows:
        print("нет проектов")
        return
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "projects_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def sample(n: int = 10, seed: int = 0) -> None:
    rng = random.Random(seed)
    md = ["# E1c — выборка вопросов для просмотра (по 10 случайных из проекта, seed 0)\n"]
    for p in sorted(PROJ.iterdir()):
        if not (p / "spec.yaml").exists():
            continue
        spec, _ = load(p)
        md.append(f"\n## {spec['id']} — {spec['title']}\n")
        for q in sorted(
            rng.sample(spec["questions"], min(n, len(spec["questions"]))), key=lambda q: q["id"]
        ):
            md.append(f"**{q['id']} ({q['type']})** {q['text']}\n")
            for a in q["anchors"]:
                f = spec["facts"][a]
                tag = (
                    " *(неявный)*"
                    if f.get("implicit")
                    else (f" *(заменяет {f['replaces']})*" if "replaces" in f else "")
                )
                md.append(f"- {a}{tag}: {f['text']}")
            md.append(f"\n*Ожидается:* {q['expected']}\n")
            md.append(f"*Неуместно:* {'; '.join(q.get('inappropriate', []))}\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review_sample.md").write_text("\n".join(md), encoding="utf-8")
    print(OUT / "review_sample.md")


if __name__ == "__main__":
    sample() if sys.argv[1:] == ["sample"] else table()
