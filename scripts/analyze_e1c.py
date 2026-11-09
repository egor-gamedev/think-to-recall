"""Анализ E1c по пре-регистрации v1.0 (reports/prereg_e1c.md).

    python scripts/analyze_e1c.py      # → reports/e1c/main_final/analysis.{json,md}

Основной анализ — 200 новых вопросов × 4 варианта, судья DeepSeek (e1c.2).
Главный тест: V7 против V6 по «выводу», точный McNemar, один тест, α = 0,05.
«Выдумка» — ручная слепая проверка (reports/e1c/review/<проект>/review.csv), пока её нет —
предварительно по судье. Вторично: старые 54 вопроса (gpt-oss), мягкая рубрика, κ, выбор V7.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_e1b import boot_ci, kappa, mcnemar  # noqa: E402

MAIN = ROOT / "reports" / "e1c" / "main_final"
NEW = ("greenhouse-iot", "clinic-bot", "econ-thesis", "indie-game")
OLD = ("crypto-wallet", "django-shop", "rnaseq-thesis")
VARIANTS = (1, 2, 6, 7)
TYPES = ("recall", "link", "change", "clarify")
TYPE_RU = {"recall": "вспомнить", "link": "связать", "change": "менялось", "clarify": "уточнить"}
METRICS = ("inference", "support", "inappropriate", "stale", "fabrication", "intrusion")
RU = {"inference": "вывод", "support": "опора", "inappropriate": "неуместный совет",
      "stale": "устаревшее", "fabrication": "выдумка", "intrusion": "постороннее"}  # fmt: skip


def rows(p: Path) -> list[dict]:
    return (
        [json.loads(ln) for ln in p.read_text("utf-8").splitlines() if ln.strip()]
        if p.exists()
        else []
    )


def verdicts(name: str) -> dict[str, dict]:
    return {r["key"]: r for r in rows(MAIN / f"judge_{name}.jsonl") if "error" not in r}


def answers(pids) -> dict[str, dict]:
    out = {}
    for pid in pids:
        for r in rows(MAIN / f"{pid}.jsonl"):
            out[f"{pid}|{r['question']}|{r['variant']}"] = r
    return out


def manual_fabrication(keys) -> tuple[dict[str, int] | None, int, int]:
    """Ручная проверка: id = sha256('e1c-review|'+key)[:10]. None — если размечено не всё."""
    by_id = {hashlib.sha256(("e1c-review|" + k).encode()).hexdigest()[:10]: k for k in keys}
    lab, total = {}, 0
    for pid in NEW:
        for name in ("review.csv", "review_add.csv"):  # add — отмеченные только вторым судьёй
            p = ROOT / "reports" / "e1c" / "review" / pid / name
            for r in csv.DictReader(p.open(encoding="utf-8")) if p.exists() else []:
                total += 1
                if r["fabrication(0/1)"].strip() in ("0", "1"):
                    lab[by_id[r["id"]]] = int(r["fabrication(0/1)"])
    return (lab if total and len(lab) == total else None), len(lab), total


def analyze() -> dict:
    A, J = answers(NEW), verdicts("deepseek")
    items = sorted({(k.split("|")[0], k.split("|")[1]) for k in A})
    itype = {(r["project"], r["question"]): r["type"] for r in A.values()}
    lab, done, total = manual_fabrication(A)
    fab_src = (
        "ручная проверка с диалогом"
        if lab is not None
        else f"ПРЕДВАРИТЕЛЬНО: судья (ручная проверка {done}/{total})"
    )

    def val(k, m):
        if k not in J:
            return np.nan
        if m == "fabrication":
            return float(lab.get(k, 0)) if lab is not None else float(bool(J[k]["fabrication"]))
        return J[k]["support"] if m == "support" else float(bool(J[k][m]))

    def vec(v, m, t=None):
        return np.array(
            [val(f"{p}|{q}|{v}", m) for p, q in items if t is None or itype[(p, q)] == t]
        )

    def pair(a, b, m, t=None):
        x, y = vec(a, m, t), vec(b, m, t)
        ok = ~np.isnan(x) & ~np.isnan(y)
        return x[ok], y[ok]

    rng = np.random.default_rng(0)
    res = {"n_items": len(items), "n_judged": sum(k in J for k in A), "fabrication_source": fab_src}
    table = {}
    for v in VARIANTS:
        for t in (*TYPES, None):
            row = {}
            for m in METRICS:
                x = vec(v, m, t)
                x = x[~np.isnan(x)]
                mu, (lo, hi) = (
                    (float(x.mean()), boot_ci(x, rng)) if len(x) else (None, (None, None))
                )
                row[m] = {"mean": mu, "ci": (lo, hi)}
            row["n"] = int(len(vec(v, "inference", t)))
            table[f"V{v}|{t or 'all'}"] = row
    res["table"] = table
    # главный тест
    p = mcnemar(*pair(7, 6, "inference"))
    p["supported"] = p["p"] < 0.05 and p["A_only"] > p["B_only"]
    x, y = pair(7, 6, "fabrication")
    lo, hi = boot_ci(x - y, rng)
    p["fabrication_noninferior"] = {"diff": float((x - y).mean()), "ci": (lo, hi), "ok": hi <= 0.05}
    res["primary_V7_vs_V6"] = p
    # вторично
    res["secondary"] = {
        f"V{a}vsV{b}": mcnemar(*pair(a, b, "inference"))
        for a, b in ((7, 2), (7, 1), (6, 2), (6, 1), (2, 1))
    }
    res["by_type_V7_vs_V6"] = {t: mcnemar(*pair(7, 6, "inference", t)) for t in TYPES}
    # выбор V7
    sel = [A[f"{p}|{q}|7"].get("selection") or {} for p, q in items]
    by_t = {}
    for t in TYPES:
        s = [A[f"{p}|{q}|7"].get("selection") or {} for p, q in items if itype[(p, q)] == t]
        by_t[t] = {"picked_any": sum(bool(x.get("picked")) for x in s) / len(s)}
    res["selection"] = {
        "picked_any": sum(bool(s.get("picked")) for s in sel) / len(sel),
        "nothing": sum(s.get("format_ok") and not s.get("picked") for s in sel) / len(sel),
        "format_violations": sum(not s.get("format_ok", True) for s in sel),
        "invalid_numbers": sum(bool(s.get("invalid")) for s in sel),
        "mean_picked": float(np.mean([len(s.get("picked", [])) for s in sel])),
        "by_type": by_t,
    }
    # описательно
    res["descriptive"] = {
        f"V{v}": {
            "truncated": sum(A[f"{p}|{q}|{v}"].get("finish_reason") == "length" for p, q in items)
            / len(items),
            "memory_tokens": float(np.mean([A[f"{p}|{q}|{v}"]["memory_tokens"] for p, q in items])),
            "answer_tokens": float(
                np.mean([A[f"{p}|{q}|{v}"]["completion_tokens"] for p, q in items])
            ),
        }
        for v in VARIANTS
    }
    # κ: DeepSeek против gpt-oss (20 %)
    G = verdicts("gptoss_kappa")
    common = [k for k in A if k in J and k in G]
    res["kappa"] = {"n": len(common), **{m: kappa([bool(J[k][m]) for k in common], [bool(G[k][m]) for k in common])
                                          for m in ("inference", "inappropriate", "stale", "fabrication", "intrusion")}}  # fmt: skip
    if common:
        res["kappa"]["by_type_inference"] = {
            t: kappa([bool(J[k]["inference"]) for k in common if J[k]["type"] == t],
                     [bool(G[k]["inference"]) for k in common if J[k]["type"] == t]) for t in TYPES}  # fmt: skip
    # старый слой (gpt-oss) и мягкая рубрика — если есть
    old_v, AO = verdicts("gptoss_old"), answers(OLD)
    res["old_layer"] = {
        f"V{v}": (lambda xs: {"inference": float(np.mean(xs)) if xs else None, "n": len(xs)})(
            [float(bool(old_v[k]["inference"])) for k in AO if k.endswith(f"|{v}") and k in old_v])
        for v in VARIANTS
    }  # fmt: skip
    S = verdicts("gptoss_soft")

    def in_sample(k):  # v1.1: мягкая рубрика — только на детерминированной выборке 20 % (как κ)
        return int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.2

    res["soft"] = {
        f"V{v}": (lambda xs: {"inference": float(np.mean(xs)) if xs else None, "n": len(xs)})(
            [float(bool(S[k]["inference"])) for k in A if k.endswith(f"|{v}") and k in S and in_sample(k)])
        for v in VARIANTS
    }  # fmt: skip
    return res


def pct(d):
    return (
        "—"
        if d["mean"] is None
        else f"{100 * d['mean']:.0f} [{100 * d['ci'][0]:.0f}–{100 * d['ci'][1]:.0f}]"
    )


def markdown(r: dict) -> str:
    L = ["# E1c — результаты (пре-регистрация v1.1)\n",
         f"Вопросов основного анализа: {r['n_items']}; оценено DeepSeek: {r['n_judged']} из {4 * r['n_items']}. «Выдумка»: {r['fabrication_source']}.\n",
         "## Все 200 новых вопросов, % [95 %]\n",
         "| | " + " | ".join(RU[m] for m in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]  # fmt: skip
    for v in VARIANTS:
        c = r["table"][f"V{v}|all"]
        L.append(
            f"| V{v} | "
            + " | ".join(
                pct(c[m])
                if m != "support"
                else (f"{c[m]['mean']:.2f}" if c[m]["mean"] is not None else "—")
                for m in METRICS
            )
            + " |"
        )
    L.append(
        "\n## «Вывод» по типам вопросов, %\n\n| | "
        + " | ".join(f"{TYPE_RU[t]} ({r['table'][f'V1|{t}']['n']})" for t in TYPES)
        + " |"
    )
    L.append("|---" * (len(TYPES) + 1) + "|")
    for v in VARIANTS:
        L.append(
            f"| V{v} | "
            + " | ".join(pct(r["table"][f"V{v}|{t}"]["inference"]) for t in TYPES)
            + " |"
        )
    p = r["primary_V7_vs_V6"]
    f = p["fabrication_noninferior"]
    L += ["\n## Главный тест: V7 против V6 по «выводу»\n",
          f"Только V7 — {p['A_only']}, только V6 — {p['B_only']}, разность {100 * p['diff']:+.1f} п.п., p = {p['p']:.3f} → **{'подтверждена' if p['supported'] else 'не подтверждена'}**.",
          f"Защита по выдумке (V7 − V6): {100 * f['diff']:+.1f} п.п. [{100 * f['ci'][0]:+.1f}; {100 * f['ci'][1]:+.1f}] — {'не хуже' if f['ok'] else 'граница +5 п.п. не выполнена'}.\n",
          "## Вторично (без поправок)\n", "| сравнение | только A | только B | разность | p |", "|---|---|---|---|---|"]  # fmt: skip
    for n, t in r["secondary"].items():
        L.append(
            f"| {n} | {t['A_only']} | {t['B_only']} | {100 * t['diff']:+.1f} п.п. | {t['p']:.3f} |"
        )
    for t, x in r["by_type_V7_vs_V6"].items():
        L.append(
            f"| V7vsV6 · {TYPE_RU[t]} | {x['A_only']} | {x['B_only']} | {100 * x['diff']:+.1f} п.п. | {x['p']:.3f} |"
        )
    s = r["selection"]
    L += ["\n## Выбор V7\n", f"Что-то выбрал: {100 * s['picked_any']:.0f} %; «НИЧЕГО»: {100 * s['nothing']:.0f} %; нарушений формата: {s['format_violations']}; номеров вне меню: {s['invalid_numbers']}; в среднем выбрано {s['mean_picked']:.1f}.",
          "По типам (выбрал что-то): " + ", ".join(f"{TYPE_RU[t]} {100 * x['picked_any']:.0f} %" for t, x in s["by_type"].items()) + "\n",
          "## Описательно\n", "| | обрывы | память, ток. | ответ, ток. |", "|---|---|---|---|"]  # fmt: skip
    for v, d in r["descriptive"].items():
        L.append(
            f"| {v} | {100 * d['truncated']:.0f} % | {d['memory_tokens']:.0f} | {d['answer_tokens']:.0f} |"
        )
    k = r["kappa"]
    L.append(f"\n## Согласие DeepSeek и gpt-oss (20 %, n = {k['n']})\n")
    L.append(
        ", ".join(
            f"{m}: κ {v:.2f}" if isinstance(v, float) else f"{m}: —"
            for m, v in k.items()
            if m not in ("n", "by_type_inference")
        )
    )
    if "by_type_inference" in k:
        L.append(
            "\n«Вывод» по типам: "
            + ", ".join(
                f"{TYPE_RU[t]} κ {v:.2f}" if isinstance(v, float) else f"{TYPE_RU[t]} —"
                for t, v in k["by_type_inference"].items()
            )
        )
    L.append("\n## Вторичные слои\n")
    L.append(
        "Старые 54 вопроса (gpt-oss, e1c.2), «вывод»: "
        + ", ".join(
            f"{v} {100 * d['inference']:.0f} % (n={d['n']})"
            if d["inference"] is not None
            else f"{v} —"
            for v, d in r["old_layer"].items()
        )
    )
    L.append(
        "\nМягкая рубрика (gpt-oss, e1b.3; выборка 20 %, поправка v1.1), «вывод»: "
        + ", ".join(
            f"{v} {100 * d['inference']:.0f} % (n={d['n']})"
            if d["inference"] is not None
            else f"{v} —"
            for v, d in r["soft"].items()
        )
    )
    return "\n".join(L) + "\n"


def main() -> None:
    r = analyze()
    (MAIN / "analysis.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1, default=float), "utf-8"
    )
    md = markdown(r)
    (MAIN / "analysis.md").write_text(md, "utf-8")
    print(md)


if __name__ == "__main__":
    main()
