"""E1b: проверка загрузочного диалога проекта против скелета.

- каждый факт сегмента помечен в реплике этого сегмента, и в ней ≥ 50% содержательных лемм факта;
- факт не помечен раньше своего сегмента; помечен не больше двух раз за весь диалог;
- объём по сегментам (приблизительно, `approx_tokens`);
- E1c: неявный факт (`implicit: true`, с `evidence`) помечен, но сказан НЕ дословно (< 50 % лемм);
  цепочки `replaces` идут по сегментам вперёд; у вопросов опорные факты существуют;
  квоты типов на 50 вопросов (10/15/15/10), «связать» с фактами из ≥ 3 сегментов, «менялось» про конец цепочки.

Запуск: python scripts/e1b_validate.py crypto-wallet
        python scripts/e1b_validate.py greenhouse-iot --root benchmark_e1c
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import text as T  # noqa: E402

PROJECTS = ROOT / "benchmark_e1b" / "projects"  # --root меняет


def load_dialogue(project: str) -> tuple[dict, list[dict]]:
    pdir = PROJECTS / project
    spec = yaml.safe_load((pdir / "spec.yaml").read_text(encoding="utf-8"))
    segs = []
    for seg in spec["segments"]:
        d = yaml.safe_load((pdir / "load" / f"{seg['id']}.yaml").read_text(encoding="utf-8"))
        assert d["segment"] == seg["id"], (d["segment"], seg["id"])
        segs.append(d)
    return spec, segs


def validate(project: str) -> list[str]:
    spec, segs = load_dialogue(project)
    problems: list[str] = []
    first_seg = {fid: i for i, s in enumerate(spec["segments"]) for fid in s["facts"]}
    counts: dict[str, int] = {}
    total = 0
    for i, (sseg, dseg) in enumerate(zip(spec["segments"], segs, strict=True)):
        turns = dseg["turns"]
        tok = sum(T.approx_tokens(t["text"]) for t in turns)
        total += tok
        print(f"{sseg['id']}: реплик {len(turns)}, ~{tok} ток.")
        for t in turns:
            assert t["role"] in ("user", "assistant"), t["role"]
            for fid in t.get("facts") or []:
                counts[fid] = counts.get(fid, 0) + 1
                if fid not in spec["facts"]:
                    problems.append(f"{sseg['id']}: неизвестный факт {fid}")
                elif first_seg[fid] > i:
                    problems.append(f"{sseg['id']}: факт {fid} помечен раньше своего сегмента")
        for fid in sseg["facts"]:
            need = set(T.content_lemmas(spec["facts"][fid]["text"]))
            tagged = [t for t in turns if fid in (t.get("facts") or [])]
            best = max(
                (len(need & set(T.content_lemmas(t["text"]))) / max(len(need), 1) for t in tagged),
                default=0,
            )
            if not tagged:
                problems.append(f"{sseg['id']}: факт {fid} не помечен")
            elif spec["facts"][fid].get("implicit"):
                if not spec["facts"][fid].get("evidence"):
                    problems.append(f"{fid}: неявный факт без evidence")
                if best >= 0.5:
                    problems.append(
                        f"{sseg['id']}: неявный факт {fid} сказан почти дословно ({best:.0%})"
                    )
            elif best < 0.5:
                problems.append(
                    f"{sseg['id']}: факт {fid} помечен, но совпадение лемм {best:.0%} < 50%"
                )
    for fid, n in counts.items():
        if n > 2:
            problems.append(f"факт {fid} помечен {n} раз (> 2)")
    print(f"всего ~{total} ток.; фактов {len(spec['facts'])}, помечено {len(counts)}")
    problems += check_e1c(spec, first_seg) if spec.get("e1c") else []
    return problems


def chain(facts: dict, fid: str) -> list[str]:
    """Цепочка замен, оканчивающаяся на fid: [A, B, fid]."""
    out = [fid]
    while "replaces" in facts[out[0]]:
        out.insert(0, facts[out[0]]["replaces"])
    return out


def check_e1c(spec: dict, first_seg: dict) -> list[str]:
    facts, qs, probs = spec["facts"], spec["questions"], []
    for fid, f in facts.items():
        if "replaces" in f:
            if f["replaces"] not in facts:
                probs.append(f"{fid}: заменяет неизвестный {f['replaces']}")
            elif first_seg.get(f["replaces"], 99) >= first_seg.get(fid, -1):
                probs.append(
                    f"{fid}: заменяет {f['replaces']} из того же или более позднего сегмента"
                )
    ends = {
        fid
        for fid in facts
        if len(chain(facts, fid)) >= 3 and not any(g.get("replaces") == fid for g in facts.values())
    }
    if len(ends) < 2:
        probs.append(f"цепочек из ≥ 3 звеньев {len(ends)} < 2")
    implicit = [f for f in facts if facts[f].get("implicit")]
    if not 4 <= len(implicit) <= 6:
        probs.append(f"неявных фактов {len(implicit)}, нужно 4–6")
    from collections import Counter

    types = Counter(q["type"] for q in qs)
    for t, n in {"recall": 10, "link": 15, "change": 15, "clarify": 10}.items():
        if types[t] != n:
            probs.append(f"вопросов {t}: {types[t]} ≠ {n}")
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        probs.append("повторяющиеся id вопросов")
    for q in qs:
        bad = [a for a in q["anchors"] if a not in facts]
        if bad:
            probs.append(f"{q['id']}: неизвестные опорные факты {bad}")
    seg3 = [
        q for q in qs if q["type"] == "link" and len({first_seg.get(a) for a in q["anchors"]}) >= 3
    ]
    if len(seg3) < 6:
        probs.append(f"«связать» с фактами из ≥ 3 сегментов: {len(seg3)} < 6")
    chain_q = [q for q in qs if q["type"] == "change" and set(q["anchors"]) & ends]
    if len(chain_q) < 8:
        probs.append(f"«менялось» про конец цепочки из 3: {len(chain_q)} < 8")
    imp_q = [q for q in qs if q["type"] == "recall" and set(q["anchors"]) & set(implicit)]
    if len(imp_q) < 4:
        probs.append(f"«вспомнить» про неявные факты: {len(imp_q)} < 4")
    fol = [q for q in qs if q.get("followup")]
    ret = [q for q in qs if q.get("retention_of")]
    if len(fol) < 2 or len(ret) < 2:
        probs.append(f"follow-up {len(fol)}, удержание {len(ret)} — нужно ≥ 2 и ≥ 2")
    return probs


if __name__ == "__main__":
    if "--root" in sys.argv:
        PROJECTS = ROOT / sys.argv[sys.argv.index("--root") + 1] / "projects"
    probs = validate(sys.argv[1])
    print("\n".join(probs) if probs else "OK: проблем нет")
    sys.exit(1 if probs else 0)
