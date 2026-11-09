"""Анализ E1b по пре-регистрации v1.0 (reports/prereg_e1b.md, §5–§7).

    python scripts/analyze_e1b.py review   # список ответов на ручную проверку «выдумки» (вслепую)
    python scripts/analyze_e1b.py          # таблицы и тесты → reports/e1b/main/analysis.{json,md}

Единица — вопрос (проект, id); у каждого варианта на вопрос один ответ, сравнения парные.
Оценки — основной судья (Nemotron-3 Ultra, поправка v1.1; внутри кода он — `glm`); «выдумка» — по ручной проверке ответов, отмеченных
хоть одним судьёй (пока проверки нет — отчёт даёт границы: только GLM / любой судья).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(os.environ.get("E1B_MAIN", ROOT / "reports" / "e1b" / "main"))
PROJECTS = ("crypto-wallet", "django-shop", "rnaseq-thesis")
VARIANTS = (1, 2, 3, 4, 5, 6)
TYPES = ("recall", "link", "change", "clarify")
TYPE_RU = {"recall": "вспомнить", "link": "связать", "change": "менялось", "clarify": "уточнить"}
BINARY = ("inference", "inappropriate", "stale", "fabrication", "intrusion")
SEED, BOOT = 0, 10_000


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]


def verdicts(name: str) -> dict[str, dict]:
    """Последний успешный вердикт на ключ «проект|вопрос|вариант»."""
    p = MAIN / f"judge_{name}.jsonl"
    return {r["key"]: r for r in read_jsonl(p) if "error" not in r} if p.exists() else {}


def load():
    answers = {}
    for pid in PROJECTS:
        for r in read_jsonl(MAIN / f"{pid}.jsonl"):
            answers[f"{pid}|{r['question']}|{r['variant']}"] = r
    # v1.1: основной судья — Nemotron; v1.2: второй (20 %) — gpt-oss-20b; GLM — третий, описательно
    return answers, verdicts("nemotron"), verdicts("gptoss")


def blind_id(key: str) -> str:
    return hashlib.sha256(("e1b-review|" + key).encode()).hexdigest()[:10]


def review(answers, glm, nem) -> None:
    """CSV для ручной проверки: ответы, где выдумку отметил хоть один судья; вариант скрыт."""
    flagged = sorted(
        (
            k
            for k in answers
            if glm.get(k, {}).get("fabrication") or nem.get(k, {}).get("fabrication")
        ),
        key=blind_id,
    )
    out = MAIN / "fabrication_review.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "project", "question", "judge1_fabricated", "judge2_fabricated",
                    "answer", "fabrication(0/1)", "reviewer", "note"])  # fmt: skip
        for k in flagged:
            pid, qid, _ = k.split("|")
            w.writerow([blind_id(k), pid, qid,
                        "; ".join(glm.get(k, {}).get("fabricated", [])),
                        "; ".join(nem.get(k, {}).get("fabricated", [])),
                        answers[k]["answer"], "", "", ""])  # fmt: skip
    print(f"{out}: {len(flagged)} ответов на проверку")


def fabrication_values(answers, glm, nem) -> tuple[dict[str, int], str]:
    """Итоговая «выдумка»: ручная проверка, если заполнена для всех отмеченных; иначе — только GLM."""
    flagged = {
        k for k in answers if glm.get(k, {}).get("fabrication") or nem.get(k, {}).get("fabrication")
    }
    p = MAIN / "fabrication_review.csv"
    manual = {}
    if p.exists():
        by_id = {blind_id(k): k for k in answers}
        for row in csv.DictReader(p.open(encoding="utf-8")):
            if row["fabrication(0/1)"].strip() in ("0", "1"):
                manual[by_id[row["id"]]] = int(row["fabrication(0/1)"])
    if flagged and flagged <= set(manual):
        # v1.3: перепроверка подтверждённых выдумок с диалогом загрузки перекрывает первую проверку
        rp = ROOT / "reports" / "e1b" / "review2" / "recheck.csv"
        if rp.exists() and os.environ.get("E1B_FAB", "v1.3") == "v1.3":
            re_ = {
                r["id"]: r["fabrication(0/1)"].strip()
                for r in csv.DictReader(rp.open(encoding="utf-8"))
            }
            if all(v in ("0", "1") for v in re_.values()):
                for bid_, v in re_.items():
                    manual[by_id[bid_]] = int(v)
                return {k: manual.get(k, 0) for k in answers}, "ручная проверка с диалогом (v1.3)"
        return {k: manual.get(k, 0) for k in answers}, "ручная проверка по фактам (v1.2)"
    return {
        k: int(bool(glm.get(k, {}).get("fabrication"))) for k in answers
    }, "ПРЕДВАРИТЕЛЬНО: только основной судья"


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Точный двусторонний McNemar: a, b — 0/1 по вопросам (вариант A, вариант B)."""
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2**n)
    return {"A_only": n10, "B_only": n01, "p": p, "diff": float(a.mean() - b.mean())}


def boot_ci(d: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def holm(ps: list[float]) -> list[float]:
    order, m = np.argsort(ps), len(ps)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def kappa(x: list[bool], y: list[bool]) -> float | None:
    n = len(x)
    if not n:
        return None
    po = sum(a == b for a, b in zip(x, y, strict=True)) / n
    p1, p2 = sum(x) / n, sum(y) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return None if pe == 1 else (po - pe) / (1 - pe)


def analyze() -> dict:
    answers, glm, nem = load()
    fab, fab_src = fabrication_values(answers, glm, nem)
    items = sorted({(k.split("|")[0], k.split("|")[1]) for k in answers})
    itype = {(r["project"], r["question"]): r["type"] for r in answers.values()}
    missing = [k for k in answers if k not in glm]

    def score(v: int, metric: str) -> np.ndarray:
        """Вектор по всем вопросам; NaN — нет оценки."""
        out = []
        for pid, qid in items:
            k = f"{pid}|{qid}|{v}"
            if metric == "fabrication":
                out.append(fab[k] if k in glm else np.nan)
            elif k not in glm:
                out.append(np.nan)
            elif metric == "support":
                out.append(glm[k]["support"])
            else:
                out.append(float(bool(glm[k][metric])))
        return np.array(out, dtype=float)

    def mask(t: str | None) -> np.ndarray:
        return np.array([t is None or itype[i] == t for i in items])

    def pair(a: int, b: int, metric: str, t: str | None = None):
        x, y, m = score(a, metric), score(b, metric), mask(t)
        ok = m & ~np.isnan(x) & ~np.isnan(y)
        return x[ok], y[ok]

    rng = np.random.default_rng(SEED)
    res: dict = {"fabrication_source": fab_src, "n_answers": len(answers), "n_judged": len(answers) - len(missing),
                 "missing_glm": missing}  # fmt: skip

    # описательные таблицы: вариант × тип
    table = {}
    for v in VARIANTS:
        for t in (*TYPES, None):
            m = mask(t)
            row = {}
            for metric in (*BINARY, "support"):
                s = score(v, metric)[m]
                row[metric] = None if np.isnan(s).all() else float(np.nanmean(s))
            row["n"] = int(m.sum())
            table[f"V{v}|{t or 'all'}"] = row
    res["table"] = table

    # H1: V6 vs V2 по «выводу» на «связать» и «менялось», Холм по 2
    h1 = {t: mcnemar(*pair(6, 2, "inference", t)) for t in ("link", "change")}
    for t, adj in zip(h1, holm([h1[t]["p"] for t in h1]), strict=True):
        h1[t]["p_holm"] = adj
        h1[t]["supported"] = adj < 0.05 and h1[t]["A_only"] > h1[t]["B_only"]
    for t in ("link", "change"):  # вторичное: опора, неуместный совет
        x, y = pair(6, 2, "support", t)
        h1[t]["support_diff"] = float((x - y).mean())
        h1[t]["support_ci"] = boot_ci(x - y, rng)
        h1[t]["inappropriate"] = mcnemar(*pair(6, 2, "inappropriate", t))
    res["H1"] = h1

    # H2: {3,4,6} лучше каждого из {1,2,5} по «выводу» (9 тестов, Холм) и не хуже по «выдумке»
    tests = {f"V{g}>V{r}": mcnemar(*pair(g, r, "inference")) for g in (3, 4, 6) for r in (1, 2, 5)}
    for name, adj in zip(tests, holm([t["p"] for t in tests.values()]), strict=True):
        tests[name]["p_holm"] = adj
        tests[name]["better"] = adj < 0.05 and tests[name]["A_only"] > tests[name]["B_only"]
    noninf = {}
    for g in (3, 4, 6):
        for r in (1, 2, 5):
            x, y = pair(g, r, "fabrication")
            lo, hi = boot_ci(x - y, rng)
            noninf[f"V{g}vsV{r}"] = {
                "diff": float((x - y).mean()),
                "ci": (lo, hi),
                "noninferior": hi <= 0.05,
            }
    success = {
        f"V{g}": all(tests[f"V{g}>V{r}"]["better"] for r in (1, 2, 5))
        and all(noninf[f"V{g}vsV{r}"]["noninferior"] for r in (1, 2, 5))
        for g in (3, 4, 6)
    }
    res["H2"] = {"tests": tests, "fabrication_noninferiority": noninf, "success": success}

    # H3: «вспомнить», V6 − V2, разность и интервал (без теста)
    x, y = pair(6, 2, "inference", "recall")
    res["H3"] = {"diff": float((x - y).mean()), "ci": boot_ci(x - y, rng), "n": len(x)}

    # описательно: обрывы, раскрытия, память, сжатия, удержание
    desc = {}
    for v in VARIANTS:
        rs = [answers[f"{p}|{q}|{v}"] for p, q in items]
        ret = [
            glm[k]["retention"]
            for p, q in items
            if (k := f"{p}|{q}|{v}") in glm and glm[k]["retention"] is not None
        ]
        desc[f"V{v}"] = {
            "truncated": sum(r.get("finish_reason") == "length" for r in rs) / len(rs),
            "expand_calls": sum(len(r["expanded"]) for r in rs),
            "memory_tokens_mean": float(np.mean([r["memory_tokens"] for r in rs])),
            "memory_empty": sum(r["memory_tokens"] == 0 for r in rs) / len(rs),
            "compressions_max": max(r["compressions_so_far"] for r in rs),
            "retention": f"{sum(ret)}/{len(ret)}",
        }
    res["descriptive"] = desc

    # согласие судей
    common = [k for k in answers if k in glm and k in nem]
    agree = {
        m: kappa([bool(glm[k][m]) for k in common], [bool(nem[k][m]) for k in common])
        for m in BINARY
    }
    xs = [v for k in common for a, v in glm[k]["anchors"].items() if a in nem[k]["anchors"]]
    ys = [nem[k]["anchors"][a] for k in common for a in glm[k]["anchors"] if a in nem[k]["anchors"]]
    agree["anchors"] = kappa(xs, ys)
    res["kappa"] = {"n": len(common), **agree}
    return res


def fmt(x, pct=True) -> str:
    return "—" if x is None else (f"{100 * x:.0f}%" if pct else f"{x:.2f}")


def markdown(res: dict) -> str:
    L = ["# E1b — результаты (анализ по пре-регистрации v1.3)\n",
         f"Оценено основным судьёй: {res['n_judged']} из {res['n_answers']}. «Выдумка»: {res['fabrication_source']}.\n"]  # fmt: skip
    for metric, ru in (("inference", "вывод"), ("support", "опора"), ("inappropriate", "неуместный совет"),
                       ("stale", "устаревшее"), ("fabrication", "выдумка"), ("intrusion", "постороннее")):  # fmt: skip
        L.append(
            f"\n## {ru}\n\n| вариант | " + " | ".join(TYPE_RU[t] for t in TYPES) + " | всего |"
        )
        L.append("|---" * (len(TYPES) + 2) + "|")
        for v in VARIANTS:
            cells = [
                fmt(res["table"][f"V{v}|{t}"][metric], metric != "support") for t in (*TYPES, "all")
            ]
            L.append(f"| V{v} | " + " | ".join(cells) + " |")
    L.append(
        "\n## H1: V6 против V2 по «выводу»\n\n| тип | только V6 | только V2 | разность | p | p (Холм) | подтверждена |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for t, h in res["H1"].items():
        L.append(
            f"| {TYPE_RU[t]} | {h['A_only']} | {h['B_only']} | {100 * h['diff']:+.0f} п.п. | {h['p']:.3f} | {h['p_holm']:.3f} | {'да' if h['supported'] else 'нет'} |"
        )
    L.append(
        "\n## H2: V3/V4/V6 против V1/V2/V5 по «выводу» (9 тестов, Холм)\n\n| сравнение | разность | p (Холм) | лучше |"
    )
    L.append("|---|---|---|---|")
    for n, t in res["H2"]["tests"].items():
        L.append(
            f"| {n} | {100 * t['diff']:+.0f} п.п. | {t['p_holm']:.3f} | {'да' if t['better'] else 'нет'} |"
        )
    L.append("\nНе хуже по «выдумке» (верхняя граница 95 % ≤ +5 п.п.):")
    for n, t in res["H2"]["fabrication_noninferiority"].items():
        L.append(
            f"- {n}: {100 * t['diff']:+.0f} п.п. [{100 * t['ci'][0]:+.0f}; {100 * t['ci'][1]:+.0f}] — {'да' if t['noninferior'] else 'нет'}"
        )
    L.append(f"\nУспех по спецификации: {res['H2']['success']}")
    h3 = res["H3"]
    L.append(
        f"\n## H3: «вспомнить», V6 − V2\n\n{100 * h3['diff']:+.0f} п.п., 95 % [{100 * h3['ci'][0]:+.0f}; {100 * h3['ci'][1]:+.0f}], n = {h3['n']}"
    )
    L.append(
        "\n## Описательно\n\n| вариант | обрывы | expand | память, ток. | пустая память | удержание |\n|---|---|---|---|---|---|"
    )
    for v, d in res["descriptive"].items():
        L.append(
            f"| {v} | {fmt(d['truncated'])} | {d['expand_calls']} | {d['memory_tokens_mean']:.0f} | {fmt(d['memory_empty'])} | {d['retention']} |"
        )
    k = res["kappa"]
    L.append(f"\n## Согласие судей (Nemotron и gpt-oss-20b, n = {k['n']})\n")
    L.append(", ".join(f"{m}: κ {fmt(v, False)}" for m, v in k.items() if m != "n"))
    return "\n".join(L) + "\n"


def main() -> None:
    if sys.argv[1:] == ["review"]:
        review(*load())
        return
    res = analyze()
    (MAIN / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    md = markdown(res)
    (MAIN / "analysis.md").write_text(md, "utf-8")
    print(md)


if __name__ == "__main__":
    main()
