"""Раннер E1 (reports/prereg_e1.md). Ограничитель расхода — ctxgraph/e1/budget.py (310 ₽ на весь E1).

Шаги:
  warm   — прогреть кэш источников для меню всех задач (без LLM) и собрать пул узлов для варианта 7
  run    — ответы тестируемой модели (с возобновлением: готовые прогоны пропускаются)
  judge  — слепая оценка судьёй (ответы перемешаны, вариант не передаётся)
  summary — сводка и прогноз расхода основного прогона

Пилот: python scripts/run_e1.py all --phase pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1 import tasks as TK  # noqa: E402
from ctxgraph.e1.budget import LIMIT_RUB, BudgetExceeded, Ledger, cost_rub  # noqa: E402
from ctxgraph.e1.judge import grade  # noqa: E402
from ctxgraph.e1.llm import Client  # noqa: E402
from ctxgraph.e1.variants import VARIANTS, GraphFactory, RandomMenus, run_variant  # noqa: E402

OUT = ROOT / "reports" / "e1"
CACHE_SRC = ROOT / ".cache" / "cache_r2.sqlite"
CACHE_E1 = ROOT / ".cache" / "e1_cache.sqlite"
POOL = OUT / "graph_pool.json"
SOLVER = "deepseek/deepseek-flash"
# v1.4: GPT-5.4 Nano на timeweb отвечает 402 (2026-10-21), судья заменён на GPT-5.4 Mini.
JUDGE = "openai/gpt-5.4-mini"
SOLVER_PARAMS = {"temperature": 0, "reasoning_effort": "none", "max_tokens": 6000}
JUDGE_PARAMS = {"temperature": 0, "max_tokens": 2000}
SEED = 20261019
PILOT_TASKS = [
    "auth-offline-field-app",
    "hard-shipping-sms",
    "control-rate-limit-public-api",
    "trip-japan-itinerary",
    "control-weekend-kazan",
]
PRIMARY, SECONDARY = (1, 3, 7, 8), (4, 2, 5, 6)

log = logging.getLogger("run_e1")


def load_env() -> None:
    from scripts.check_model import load_env as _le

    _le(ROOT / ".env")


def encoder():
    from ctxgraph.embed import Model2VecEncoder

    return Model2VecEncoder()


# ---------------- warm ----------------
def ensure_cache_db() -> None:
    """База, где есть только кэш источников (без графа, опыта, событий)."""
    if CACHE_E1.exists():
        return
    from ctxgraph.store import Store

    Store(CACHE_E1).close()
    con = sqlite3.connect(CACHE_E1)
    con.execute("ATTACH DATABASE ? AS src", (str(CACHE_SRC),))
    con.execute("INSERT OR IGNORE INTO cache SELECT * FROM src.cache")
    con.commit()
    con.close()


def warm(task_ids: list[str], expand_ids: set[str]) -> None:
    """Меню всех задач на auto_depth 0–3 и раскрытия первого уровня для `expand_ids` — без LLM.

    Новые ответы источников пишутся в кэш e1_cache.sqlite; пул узлов — в graph_pool.json.
    """
    from ctxgraph import E1_CONFIG, build_engine

    ensure_cache_db()
    tasks = TK.load_tasks()
    enc = encoder()
    pool: dict[str, list[list[str]]] = (
        json.loads(POOL.read_text(encoding="utf-8")) if POOL.exists() else {}
    )
    for tid in task_ids:
        eng = build_engine(
            CACHE_E1, encoder=enc, experience=False, config=E1_CONFIG, session=f"warm-{tid}"
        )
        items: dict[str, str] = {}
        for depth in (0, 1, 2, 3):
            m = eng.menu(TK.graph_items(tasks[tid]), auto_depth=depth)
            for it in m.items():
                items[it.node_id] = it.label
        if tid in expand_ids:
            m = eng.menu(TK.graph_items(tasks[tid]))
            for it in m.items():
                arm = eng.expand(it.node_id)
                for sub in arm.all_items() if arm else []:
                    items[sub.node_id] = sub.label
        # Граф прогрева не должен копить узлы и рёбра: в кэш-базе остаётся только кэш.
        for t in ("nodes", "edges", "shortcuts", "fragments", "mentions", "events"):
            eng.store.db.execute(f"DELETE FROM {t}")
        eng.store.close()
        pool[tid] = sorted([k, v] for k, v in items.items())
        wd = next(s for s in eng.sources if s.name == "wikidata")
        print(f"warm {tid}: узлов {len(items)}, живых запросов к Викиданным {wd.live_requests}")
    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=0), encoding="utf-8")


# ---------------- run ----------------
def jobs_for(
    phase: str, task_ids: list[str], repeats: int, variants: tuple[int, ...]
) -> list[tuple]:
    """Порядок прогонов (пре-регистрация §4, §10).

    Пилот: вариант 3 → остальные (кроме 2) в случайном порядке → вариант 2.
    Основной: по повторам 1..3 — вариант 3, затем 1, 7, 8 в случайном порядке;
    потом вторичные 4, 2, 5, 6 по одному повтору (вариант 2 берёт раскрытия варианта 3, повтор 1).
    """
    rng = random.Random(SEED)
    out: list[tuple] = []
    if phase == "pilot":
        out += [(t, 3, 1) for t in task_ids if 3 in variants]
        rest = [(t, v, 1) for t in task_ids for v in variants if v not in (2, 3)]
        rng.shuffle(rest)
        out += rest + [(t, 2, 1) for t in task_ids if 2 in variants]
        return out
    for r in range(1, repeats + 1):
        out += [(t, 3, r) for t in task_ids if 3 in variants]
        rest = [(t, v, r) for t in task_ids for v in PRIMARY if v != 3 and v in variants]
        rng.shuffle(rest)
        out += rest
    for v in SECONDARY:
        if v in variants:
            out += [(t, v, 1) for t in task_ids]
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(phase: str, task_ids: list[str], repeats: int, variants: tuple[int, ...]) -> None:
    ensure_cache_db()
    tasks = TK.load_tasks()
    ledger = Ledger(OUT / "spend.jsonl")
    client = Client(SOLVER, ledger, phase=phase, params=SOLVER_PARAMS)
    graphs = GraphFactory(CACHE_E1, encoder())
    randoms = RandomMenus(json.loads(POOL.read_text(encoding="utf-8")))
    path = OUT / f"{phase}_answers.jsonl"
    done = {
        (r["task"], r["variant"], r["repeat"]): r for r in read_jsonl(path) if not r.get("error")
    }
    todo = [j for j in jobs_for(phase, task_ids, repeats, variants) if j not in done]
    print(
        f"{phase}: прогонов {len(todo)} (готово {len(done)}), потрачено {ledger.spent:.2f} ₽ из {LIMIT_RUB:.0f}"
    )
    for i, (tid, v, r) in enumerate(todo, 1):
        v3 = done.get((tid, 3, 1))
        try:
            rec = run_variant(
                tasks[tid],
                v,
                r,
                client=client,
                graphs=graphs,
                randoms=randoms,
                v3_expanded=(v3 or {}).get("expanded") if v == 2 else None,
            )
        except BudgetExceeded as e:
            print(f"ОСТАНОВ по бюджету: {e}")
            break
        rec["phase"] = phase
        rec["cost_rub"] = sum(cost_rub(SOLVER, c["usage"]) for c in rec["calls"])
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if not rec.get("error"):
            done[(tid, v, r)] = rec
        print(
            f"[{i}/{len(todo)}] {rec['run_id']}: {rec['usage']} expand={len(rec['expanded'])} "
            f"{rec['cost_rub']:.2f} ₽ всего {ledger.spent:.2f} ₽ {('ОШИБКА ' + rec['error']) if rec['error'] else ''}"
        )


# ---------------- judge ----------------
def judge(phase: str) -> None:
    tasks = TK.load_tasks()
    ledger = Ledger(OUT / "spend.jsonl")
    client = Client(JUDGE, ledger, phase=phase, params=JUDGE_PARAMS)
    answers = [a for a in read_jsonl(OUT / f"{phase}_answers.jsonl") if not a.get("error")]
    latest = {}
    for a in answers:  # при повторных записях берём последнюю
        latest[a["run_id"]] = a
    path = OUT / f"{phase}_judged.jsonl"
    judged = {j["run_id"] for j in read_jsonl(path) if not j.get("error")}
    todo = [a for a in latest.values() if a["run_id"] not in judged]
    random.Random(SEED + 1).shuffle(todo)  # слепо: порядок не связан с вариантом
    print(f"судья: ответов {len(todo)}, потрачено {ledger.spent:.2f} ₽")
    for i, a in enumerate(todo, 1):
        try:
            v = grade(client, tasks[a["task"]], a["answer"], run_id=a["run_id"])
        except BudgetExceeded as e:
            print(f"ОСТАНОВ по бюджету: {e}")
            break
        v.update(
            run_id=a["run_id"],
            task=a["task"],
            variant=a["variant"],
            repeat=a["repeat"],
            control=a["control"],
            ts=time.time(),
        )
        if "judge_usage" in v:
            v["cost_rub"] = cost_rub(JUDGE, v["judge_usage"])
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {a['run_id']} {'ОШИБКА ' + v['error'] if v.get('error') else ''}")


# ---------------- summary ----------------
def summary(phase: str) -> dict:
    ans = {a["run_id"]: a for a in read_jsonl(OUT / f"{phase}_answers.jsonl")}
    jud = {j["run_id"]: j for j in read_jsonl(OUT / f"{phase}_judged.jsonl") if not j.get("error")}
    spend = read_jsonl(OUT / "spend.jsonl")
    rows = {}
    for v in VARIANTS:
        a = [x for x in ans.values() if x["variant"] == v]
        ok = [x for x in a if not x.get("error")]
        j = [jud[x["run_id"]] for x in ok if x["run_id"] in jud]
        reqs = [vd for x in j if not x["control"] for vd in x["verdicts"]]
        rows[v] = {
            "runs": len(a),
            "errors": len(a) - len(ok),
            "in_tok": round(sum(x["usage"]["prompt_tokens"] for x in ok) / max(len(ok), 1)),
            "out_tok": round(sum(x["usage"]["completion_tokens"] for x in ok) / max(len(ok), 1)),
            "rub_per_run": round(sum(x["cost_rub"] for x in ok) / max(len(ok), 1), 3),
            "calls": round(sum(len(x["calls"]) for x in ok) / max(len(ok), 1), 2),
            "expanded": round(sum(len(x["expanded"]) for x in ok) / max(len(ok), 1), 2),
            "addressed": f"{sum(r['addressed'] for r in reqs)}/{len(reqs)}",
            "noticed": f"{sum(r['noticed'] for r in reqs)}/{len(reqs)}",
            "initiative": f"{sum(x['initiative'] for x in j)}/{len(j)}",
            "off_topic": f"{sum(x['off_topic'] for x in j)}/{len(j)}",
            "judge_missing": sum(
                bool(x.get("fields_missing")) or any(r.get("missing") for r in x["verdicts"])
                for x in j
            ),
        }
    judge_rub = [j.get("cost_rub", 0) for j in jud.values()]
    per_judge = sum(judge_rub) / max(len(judge_rub), 1)
    plan = {v: 30 * (3 if v in PRIMARY else 1) for v in VARIANTS}
    forecast = sum(rows[v]["rub_per_run"] * n for v, n in plan.items()) + per_judge * sum(
        plan.values()
    )
    spent_phase = sum(s["rub"] for s in spend if s["phase"] == phase)
    spent_all = sum(s["rub"] for s in spend)
    out = {
        "phase": phase,
        "variants": rows,
        "judge_rub_per_answer": round(per_judge, 4),
        "spent_phase_rub": round(spent_phase, 2),
        "spent_total_rub": round(spent_all, 2),
        "forecast_main_rub": round(forecast, 1),
        "limit_rub": LIMIT_RUB,
        "fingerprints": sorted({f for a in ans.values() for f in a.get("fingerprints", [])}),
    }
    (OUT / f"{phase}_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["warm", "run", "judge", "summary", "all"])
    ap.add_argument("--phase", default="pilot", choices=["pilot", "main"])
    ap.add_argument("--tasks", default="")
    ap.add_argument("--repeats", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_env()
    all_ids = sorted(TK.load_tasks())
    if args.phase == "pilot":
        task_ids = args.tasks.split(",") if args.tasks else PILOT_TASKS
        repeats, variants = args.repeats or 1, VARIANTS
    else:
        task_ids = args.tasks.split(",") if args.tasks else all_ids
        repeats, variants = args.repeats or 3, VARIANTS
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step in ("warm", "all"):
        warm(all_ids, expand_ids=set(task_ids))
    if args.step in ("run", "all"):
        run(args.phase, task_ids, repeats, variants)
    if args.step in ("judge", "all"):
        judge(args.phase)
    if args.step in ("summary", "all"):
        summary(args.phase)
    return 0


if __name__ == "__main__":
    sys.exit(main())
