"""Анализ основного прогона E1 строго по пре-регистрации v1.3 (§6, §7, §4а).

Выход: reports/e1/main_analysis.json, reports/e1/calibration_sample.csv (20 пар для Егора, вслепую).
Запуск: python scripts/analyze_e1.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1 import tasks as TK  # noqa: E402

OUT = ROOT / "reports" / "e1"
SEED = 20261019
B = 10_000
PRIMARY_PAIRS = [(3, 1), (3, 7), (3, 8)]
MIN_EFFECT = 0.05


def read_jsonl(p: Path) -> list[dict]:
    return (
        [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if p.exists()
        else []
    )


def load(phase: str = "main"):
    tasks = TK.load_tasks()
    answers: dict[str, dict] = {}
    for a in read_jsonl(OUT / f"{phase}_answers.jsonl"):
        prev = answers.get(a["run_id"])
        # Последняя успешная запись важнее ошибки; иначе — последняя.
        if prev is None or not a.get("error") or prev.get("error"):
            answers[a["run_id"]] = a
    judged = {}
    for j in read_jsonl(OUT / f"{phase}_judged.jsonl"):
        if not j.get("error"):
            judged[j["run_id"]] = j
    return tasks, answers, judged


def run_scores(task: dict, ans: dict, judged: dict) -> dict | None:
    """Оценки одного прогона. Ошибка прогона = ничего не замечено и не учтено (§4а).
    Ответ без вердикта судьи — исключается (None)."""
    reqs = task.get("hidden_requirements") or []
    if ans.get("error"):
        return {
            "addressed": 0.0 if reqs else None,
            "noticed": 0.0 if reqs else None,
            "initiative": 0.0,
            "off_topic": 0.0,
            "kept_standard": 0.0 if task.get("control") else None,
            "unsupported": 0,
            "failed_run": True,
        }
    j = judged.get(ans["run_id"])
    if j is None:
        return None
    v = j["verdicts"]
    return {
        "addressed": (sum(x["addressed"] for x in v) / len(v)) if v else None,
        "noticed": (sum(x["noticed"] for x in v) / len(v)) if v else None,
        "initiative": float(j["initiative"]),
        "off_topic": float(j["off_topic"]),
        "kept_standard": float(j["kept_standard"]) if task.get("control") else None,
        "unsupported": len(j["unsupported_claims"]),
        "failed_run": False,
    }


def task_matrix(
    tasks, answers, judged, metric: str, variants, *, only=None
) -> tuple[list[str], np.ndarray]:
    """Задачи × варианты: среднее по повторам. Задачи без значения хотя бы в одном варианте — вне сравнения."""
    ids = sorted(t for t in tasks if only is None or only(tasks[t]))
    rows, keep = [], []
    for tid in ids:
        row = []
        for v in variants:
            vals = []
            for a in answers.values():
                if a["task"] == tid and a["variant"] == v:
                    s = run_scores(tasks[tid], a, judged)
                    if s is not None and s[metric] is not None:
                        vals.append(s[metric])
            row.append(np.mean(vals) if vals else np.nan)
        if not any(np.isnan(row)):
            rows.append(row)
            keep.append(tid)
    return keep, np.array(rows)


def boot_diff(m: np.ndarray, i: int, k: int, rng: np.random.Generator) -> dict:
    d = m[:, i] - m[:, k]
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "diff": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p": float(min(p, 1.0)),
        "n_tasks": n,
    }


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj, running = [0.0] * m, 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[idx]))
        adj[idx] = running
    return adj


def analyze(phase: str = "main") -> dict:
    tasks, answers, judged = load(phase)
    rng = np.random.default_rng(SEED)
    has_req = lambda t: not t.get("control")  # noqa: E731
    is_ctrl = lambda t: bool(t.get("control"))  # noqa: E731
    variants = (1, 2, 3, 4, 5, 6, 7, 8)
    res: dict = {"phase": phase, "prereg": "v1.3 (f8af093)", "B": B, "seed": SEED}

    # Покрытие и сбои
    res["coverage"] = {
        str(v): {
            "runs": sum(a["variant"] == v for a in answers.values()),
            "failed_runs": sum(
                a["variant"] == v and bool(a.get("error")) for a in answers.values()
            ),
            "unjudged": sum(
                a["variant"] == v and not a.get("error") and a["run_id"] not in judged
                for a in answers.values()
            ),
        }
        for v in variants
    }

    # Основные: addressed, 3 против 1/7/8
    ids, m = task_matrix(tasks, answers, judged, "addressed", (3, 1, 7, 8), only=has_req)
    col = {3: 0, 1: 1, 7: 2, 8: 3}
    prim = [boot_diff(m, col[a], col[b], rng) for a, b in PRIMARY_PAIRS]
    for r, adj in zip(prim, holm([r["p"] for r in prim]), strict=True):
        r["p_holm"] = adj
    res["primary"] = {f"{a}-{b}": r for (a, b), r in zip(PRIMARY_PAIRS, prim, strict=True)}
    res["primary_tasks"] = ids
    res["success_H1_H4"] = all(
        r["diff"] >= MIN_EFFECT and r["p_holm"] < 0.05 and r["diff"] > 0 for r in prim
    )
    res["partial"] = [k for k, r in res["primary"].items() if r["diff"] > 0 and r["p_holm"] < 0.05]

    # Вторичные (Холм на 4)
    sec = {}
    _, mi = task_matrix(tasks, answers, judged, "initiative", (3, 1))
    sec["initiative_3-1"] = boot_diff(mi, 0, 1, rng)
    _, mn = task_matrix(tasks, answers, judged, "noticed", (3, 1), only=has_req)
    sec["noticed_3-1"] = boot_diff(mn, 0, 1, rng)
    _, mo = task_matrix(tasks, answers, judged, "off_topic", (3, 1))
    sec["off_topic_3-1"] = boot_diff(mo, 0, 1, rng)
    _, mk = task_matrix(tasks, answers, judged, "kept_standard", (3, 1), only=is_ctrl)
    sec["kept_standard_3-1"] = boot_diff(mk, 0, 1, rng)
    for r, adj in zip(sec.values(), holm([r["p"] for r in sec.values()]), strict=True):
        r["p_holm"] = adj
    res["secondary"] = sec
    res["guardrails"] = {
        "off_topic_ok": sec["off_topic_3-1"]["diff"] <= 0.10,
        "kept_standard_ok": sec["kept_standard_3-1"]["diff"] >= -0.10,
    }

    # Средние по вариантам (все метрики) и токены
    means = {}
    for metric, only in (
        ("addressed", has_req),
        ("noticed", has_req),
        ("initiative", None),
        ("off_topic", None),
        ("kept_standard", is_ctrl),
    ):
        _, mm = task_matrix(tasks, answers, judged, metric, variants, only=only)
        means[metric] = (
            {str(v): float(mm[:, i].mean()) for i, v in enumerate(variants)} if len(mm) else {}
        )
    res["means"] = means
    tok = {}
    for v in variants:
        ok = [a for a in answers.values() if a["variant"] == v and not a.get("error")]
        tok[str(v)] = {
            "in": float(np.mean([a["usage"]["prompt_tokens"] for a in ok])) if ok else None,
            "out": float(np.mean([a["usage"]["completion_tokens"] for a in ok])) if ok else None,
            "rub": float(np.mean([a["cost_rub"] for a in ok])) if ok else None,
            "expand_calls": int(sum(a["expand_calls"] for a in ok)),
            "runs_with_expand": int(sum(a["expand_calls"] > 0 for a in ok)),
            "far_expanded": int(sum(a["far_expanded"] for a in ok)),
        }
    res["tokens"] = tok
    unsupported = {}
    for v in variants:
        vals = [
            run_scores(tasks[a["task"]], a, judged)
            for a in answers.values()
            if a["variant"] == v and is_ctrl(tasks[a["task"]])
        ]
        vals = [x["unsupported"] for x in vals if x]
        unsupported[str(v)] = float(np.mean(vals)) if vals else None
    res["unsupported_claims_controls"] = unsupported

    # Глубина: варианты 3–6 по addressed, кривая и разведочный тест перевёрнутой U
    _, md = task_matrix(tasks, answers, judged, "addressed", (3, 4, 5, 6), only=has_req)
    depth = {}
    for i, v in enumerate((3, 4, 5, 6)):
        idx = rng.integers(0, len(md), size=(B, len(md)))
        bm = md[:, i][idx].mean(axis=1)
        depth[str(v)] = {
            "mean": float(md[:, i].mean()),
            "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
        }
    peak = max((4, 5), key=lambda v: depth[str(v)]["mean"])
    pi = (3, 4, 5, 6).index(peak)
    vs3, vs6 = boot_diff(md, pi, 0, rng), boot_diff(md, pi, 3, rng)
    top = max(depth, key=lambda v: depth[v]["mean"])
    res["depth"] = {
        "curve": depth,
        "peak_candidate": peak,
        "peak_vs_3": vs3,
        "peak_vs_6": vs6,
        "inverted_u": top in ("4", "5")
        and vs3["diff"] > 0
        and vs3["p"] < 0.05
        and vs6["diff"] > 0
        and vs6["p"] < 0.05,
    }
    _, m23 = task_matrix(tasks, answers, judged, "addressed", (2, 3), only=has_req)
    res["menu_vs_inject_2-3"] = boot_diff(m23, 0, 1, rng)

    res["fingerprints"] = sorted({f for a in answers.values() for f in a.get("fingerprints", [])})
    spend = read_jsonl(OUT / "spend.jsonl")
    res["spent_rub"] = {
        "pilot": round(sum(s["rub"] for s in spend if s["phase"] == "pilot"), 2),
        "main": round(sum(s["rub"] for s in spend if s["phase"] == "main"), 2),
    }
    (OUT / f"{phase}_analysis.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    calibration_sample(tasks, answers, judged)
    return res


def calibration_sample(tasks, answers, judged, n: int = 20) -> None:
    """20 пар (задача с требованиями, ответ) со стратами по вариантам, без указания варианта (§6)."""
    rng = random.Random(SEED + 2)
    pool = [
        a
        for a in answers.values()
        if not a.get("error") and a["run_id"] in judged and not tasks[a["task"]].get("control")
    ]
    by_v: dict[int, list] = {}
    for a in sorted(pool, key=lambda a: a["run_id"]):
        by_v.setdefault(a["variant"], []).append(a)
    picked, vs = [], sorted(by_v)
    while len(picked) < n and any(by_v.values()):
        for v in vs:
            if by_v[v] and len(picked) < n:
                picked.append(by_v[v].pop(rng.randrange(len(by_v[v]))))
    rng.shuffle(picked)
    rows = []
    for k, a in enumerate(picked, 1):
        t = tasks[a["task"]]
        for r in t["hidden_requirements"]:
            rows.append(
                {
                    "pair": k,
                    "task": t["id"],
                    "requirement_id": r["id"],
                    "requirement": r["description"].strip(),
                    "noticed_question": r["noticed"],
                    "addressed_question": r["addressed"],
                    "answer": a["answer"],
                    "noticed": "",
                    "addressed": "",
                    "_run_id": a["run_id"],
                }
            )
    with (OUT / "calibration_sample.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in rows[0] if c != "_run_id"])
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k != "_run_id"} for r in rows)
    (OUT / "calibration_key.json").write_text(
        json.dumps(
            [
                {"pair": r["pair"], "requirement_id": r["requirement_id"], "run_id": r["_run_id"]}
                for r in rows
            ],
            ensure_ascii=False,
            indent=0,
        ),
        encoding="utf-8",
    )


def kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


if __name__ == "__main__":
    out = analyze(sys.argv[1] if len(sys.argv) > 1 else "main")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "coverage",
                    "primary",
                    "success_H1_H4",
                    "secondary",
                    "guardrails",
                    "means",
                    "depth",
                    "spent_rub",
                )
            },
            ensure_ascii=False,
            indent=1,
        )
    )
