"""Stage 2 main report (`reports/prereg_stage2.md`, frozen 2026-11-08, amendment 1): base A, variants SC / SR / SG.

    python scripts/stage2_report.py <out>          # <out>/{imlogic,implexconv}_main/{S0,SC,SR,SG}_A.jsonl
                                                   # -> <out>/stage2_report.md + stage2_report.json

H1 SC - A (Holm over the two sets); H2 SR - SC, H3 SR - A (Holm over 4); SG descriptive (IMLogic only, SG on
ImplexConv postponed). Test as in bench1b: one-sided paired sign-flip permutation, 10 000 permutations, seed 0;
95 % bootstrap; wins / losses per question (R@5 difference sign). Criterion: prereg §4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import bench1_run as B  # noqa: E402
import bench1b_run as BB  # noqa: E402

SETS = {"imlogic": "IMLogic (memory_l)", "implexconv": "ImplexConv"}


def load(path: Path) -> dict[str, dict]:
    return {json.loads(x)["id"]: json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()} if path.exists() else {}


def test(rows: dict, x: str, y: str, ids: list[str]) -> dict:
    d = [B.recall(rows[x][i]["rank"], rows[x][i]["gold"], 5) - B.recall(rows[y][i]["rank"], rows[y][i]["gold"], 5) for i in ids]
    m, lo, hi = B.bootstrap(d)
    return {"diff": 100 * m, "ci": [100 * lo, 100 * hi], "p": BB.perm_p(d), "p_reverse": BB.perm_p([-v for v in d]),
            "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "n": len(d)}  # fmt: skip


def main() -> None:
    out = Path(sys.argv[1])
    res: dict = {}
    for ds in SETS:
        rows = {v: load(out / f"{ds}_main" / f"{v}_A.jsonl") for v in ("S0", "SC", "SR", "SG")}
        rows = {v: r for v, r in rows.items() if r}
        if "S0" not in rows:
            continue
        ids = sorted(set.intersection(*(set(r) for r in rows.values())))
        res[ds] = {"n": len(ids), "empty_plans": sum(rows["S0"][i].get("steps", 1) == 0 for i in ids),
                   "recall": {f"{v}@{k}": 100 * sum(B.recall(r[i]["rank"], r[i]["gold"], k) for i in ids) / len(ids)
                              for v, r in rows.items() for k in (1, 3, 5, 10)},
                   "tests": {name: test(rows, x, y, ids) for name, x, y in
                             (("H1 SC-A", "SC", "S0"), ("H2 SR-SC", "SR", "SC"), ("H3 SR-A", "SR", "S0"),
                              ("SG-A", "SG", "S0"), ("SG-SR", "SG", "SR")) if x in rows and y in rows}}  # fmt: skip
        if ds == "implexconv":  # descriptive split by kind
            res[ds]["by_type"] = {}
            for t in ("opposed", "supportive"):
                sub = [i for i in ids if rows["S0"][i]["type"] == t]
                res[ds]["by_type"][t] = {"n": len(sub), **{v: 100 * sum(B.recall(rows[v][i]["rank"], rows[v][i]["gold"], 5)
                                                                     for i in sub) / max(1, len(sub)) for v in rows}}  # fmt: skip
    h1 = BB.holm({ds: r["tests"]["H1 SC-A"]["p"] for ds, r in res.items()})
    sec = BB.holm({f"{ds}|{t}": r["tests"][t]["p"] for ds, r in res.items() for t in ("H2 SR-SC", "H3 SR-A")})
    for ds, r in res.items():
        r["tests"]["H1 SC-A"]["p_holm"] = h1[ds]
        for t in ("H2 SR-SC", "H3 SR-A"):
            r["tests"][t]["p_holm"] = sec[f"{ds}|{t}"]
    sig = {ds: r["tests"]["H1 SC-A"]["p_holm"] < 0.05 for ds, r in res.items()}
    rev = {ds: r["tests"]["H1 SC-A"]["p_reverse"] < 0.05 for ds, r in res.items()}
    if any(rev.values()):
        verdict = "вредит: обратный тест (A > SC) значим на " + ", ".join(SETS[d] for d, v in rev.items() if v)
    elif any(sig.values()) and len(res) == 2:
        verdict = "поиск по шагам плана полезен (H1 значим на " + ", ".join(SETS[d] for d, v in sig.items() if v) + "; обратный нигде не значим)"
    elif any(sig.values()):
        verdict = "H1 значим на единственном наборе в анализе — вывод по одному набору"
    else:
        verdict = "прироста нет (H1 нигде не значим, обратный тоже) — без вывода «работает»"
    h2 = [ds for ds, r in res.items() if r["tests"]["H2 SR-SC"]["p_holm"] < 0.05]
    res["verdict"] = verdict
    res["h2_verdict"] = ("рукава полезны сверх поиска по всему тексту шагов на " + ", ".join(SETS[d] for d in h2)) if h2 else \
        "деление на рукава не дало прироста над поиском по всему тексту шагов"  # fmt: skip
    (out / "stage2_report.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf-8")

    f = lambda t: (f"{t['diff']:+.1f} [{t['ci'][0]:+.1f}; {t['ci'][1]:+.1f}]; {t['wins']} / {t['losses']}; "  # noqa: E731
                   f"p = {t['p']:.4f}" + (f", по Холму {t['p_holm']:.4f}" if "p_holm" in t else "") + f"; обратный p = {t['p_reverse']:.3f}")
    md = ["# Этап 2, основная часть: поиск по шагам плана поверх A", "",
          "Пре-регистрация: `reports/prereg_stage2.md` (заморожена 2026-11-08, поправка 1). Вопросы — по поправке 1.", "",
          f"**Вывод (H1, §4):** {verdict}.", "", f"**Деление на рукава (H2):** {res['h2_verdict']}.", ""]
    for ds, r in ((d, res[d]) for d in SETS if d in res):
        md += [f"## {SETS[ds]} — {r['n']} вопросов (пустых планов: {r['empty_plans']})", "",
               "| | R@1 | R@3 | R@5 | R@10 |", "|---|---|---|---|---|"]
        for v in ("S0", "SC", "SR", "SG"):
            if f"{v}@5" in r["recall"]:
                md.append(f"| {'A' if v == 'S0' else v} | " + " | ".join(f"{r['recall'][f'{v}@{k}']:.1f}" for k in (1, 3, 5, 10)) + " |")
        md += ["", "| сравнение (R@5) | разность [95 %]; победы / поражения; p |", "|---|---|"]
        md += [f"| {t} | {f(x)} |" for t, x in r["tests"].items()]
        if "by_type" in r:
            md += ["", "Описательно, R@5 по типу: " + "; ".join(
                f"{t} (n = {b['n']}): " + ", ".join(f"{'A' if v == 'S0' else v} {b[v]:.1f}" for v in ("S0", "SC", "SR") if v in b)
                for t, b in r["by_type"].items())]
        if ds == "implexconv":
            md += ["", "SG ImplexConv отложен из-за времени построения графа на CPU (поправка 1, п. 5)."]
        md.append("")
    (out / "stage2_report.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
