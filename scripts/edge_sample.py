"""Выгрузка случайных рёбер для ручной разметки.

По `--per-type` рёбер каждого типа источника; в каждом типе половина — рёбра, показанные
в меню, половина — остальные. Граф наращивается меню и одним раскрытием на рукав по
контекстам: 5 тем из demo_topics, задачи widethink (`--tasks`) и, после метки времени,
отложенные контексты (`--heldout`). Колонка `context_set`: `heldout`, если ребро впервые
появилось на отложенных контекстах. `--exclude` — CSV прошлых раундов: эти рёбра в выборку
не попадут. `--fresh` — новая база, из `--cache-from` переносится только кэш источников.
Колонку `meaningful` заполняет разметчик: 1 — связь осмысленна, 0 — нет.

Раунд 2: python scripts/edge_sample.py --fresh --db .cache/round2.sqlite \\
    --exclude reports/labels/round1_labeled.csv --seed 29 --out reports/edge_sample_round2.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sqlite3
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import build_engine  # noqa: E402
from scripts.demo_topics import TOPICS  # noqa: E402

TYPES = ("wordnet", "ruwordnet", "wikidata", "conceptnet", "experience", "link")


def task_contexts(tasks_dir: Path) -> list[tuple[str, list[dict]]]:
    out = []
    for p in sorted(tasks_dir.glob("*.yaml")):
        t = yaml.safe_load(p.read_text(encoding="utf-8"))
        items = [{"role": "user", "content": t["task"]}] + list(t.get("context", []))
        out.append((t["id"], items))
    return out


def fresh_db(db: Path, cache_from: Path) -> None:
    """Новая база без графа и опыта; кэш ответов источников переносится, чтобы не ходить в сеть."""
    db.unlink(missing_ok=True)
    build_engine(db).store.close()  # создать схему
    if cache_from.exists():
        con = sqlite3.connect(db)
        con.execute("ATTACH DATABASE ? AS old", (str(cache_from),))
        con.execute("INSERT OR IGNORE INTO cache SELECT * FROM old.cache")
        con.commit()
        con.close()


def grow(eng, sid: str, items) -> int:
    eng.new_session(sid)
    eng.observe(items)  # опыт для следующих контекстов
    m = eng.menu(items)
    for a in m.arms:
        first = next((i for s in ("deeper", "side", "assoc") for i in a.items.get(s, [])), None)
        if first:
            eng.expand(first.node_id)
    return len(m.arms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / ".cache" / "demo.sqlite")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--cache-from", type=Path, default=ROOT / ".cache" / "demo.sqlite")
    ap.add_argument(
        "--tasks", type=Path, default=ROOT.parent / "thinkingLlm" / "benchmark" / "tasks"
    )
    ap.add_argument(
        "--heldout", type=Path, default=ROOT / "scripts" / "data" / "heldout_contexts.yaml"
    )
    ap.add_argument("--exclude", type=Path, action="append", default=[])
    ap.add_argument("--per-type", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "edge_sample.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.fresh:
        fresh_db(args.db, args.cache_from)
    eng = build_engine(args.db, seed=args.seed)
    for name, (past, current) in TOPICS.items():
        eng.new_session(f"past-{name}")
        eng.observe(past)
        grow(eng, f"now-{name}", current)
    if args.tasks.exists():
        for tid, items in task_contexts(args.tasks):
            print(f"{tid}: рукавов {grow(eng, f'sample-{tid}', items)}")
    t_heldout = time.time()
    if args.heldout.exists():
        for c in yaml.safe_load(args.heldout.read_text(encoding="utf-8")):
            for i, p in enumerate(c.get("past", [])):
                eng.new_session(f"past-{c['id']}-{i}")
                eng.observe(p)
            print(f"heldout {c['id']}: рукавов {grow(eng, 'heldout-' + c['id'], c['current'])}")

    excluded: set[tuple[str, str]] = set()
    for path in args.exclude:
        with path.open(encoding="utf-8-sig") as fh:
            excluded |= {(r["from_id"], r["to_id"]) for r in csv.DictReader(fh)}

    store = eng.store
    created = {
        (s, d): c
        for s, d, c in store.db.execute(
            "SELECT src, dst, MIN(created) FROM edges GROUP BY src, dst"
        )
    }
    rng = random.Random(args.seed)
    by_type: dict[str, list] = {t: [] for t in TYPES}
    for e in store.all_edges():
        if e.source in by_type and (e.src, e.dst) not in excluded:
            by_type[e.source].append(e)
    rows = []
    for t in TYPES:
        pool = sorted(by_type[t], key=lambda e: (e.src, e.dst, e.rel))
        rng.shuffle(pool)
        seen_p = [e for e in pool if e.shown > 0]
        rest = [e for e in pool if e.shown == 0]
        half = args.per_type // 2
        take = seen_p[:half]
        take += rest[: args.per_type - len(take)]
        take += seen_p[half : half + args.per_type - len(take)]
        for e in take:
            a, b = store.node(e.src), store.node(e.dst)
            rows.append(
                {
                    "source_type": t,
                    "from": a.label if a else e.src,
                    "relation": e.rel,
                    "to": b.label if b else e.dst,
                    "from_id": e.src,
                    "to_id": e.dst,
                    "shown_in_menu": int(e.shown > 0),
                    "context_set": "heldout" if created[(e.src, e.dst)] >= t_heldout else "seen",
                    "provenance": e.provenance,
                    "meaningful": "",
                    "comment": "",
                }
            )
    rng.shuffle(rows)  # разметчик не должен видеть тип источника блоками
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["source_type"])
        w.writeheader()
        w.writerows(rows)
    print("рёбер в графе по типам (без исключённых):", {t: len(v) for t, v in by_type.items()})
    print(f"исключено из прошлых раундов: {len(excluded)}; выгружено {len(rows)} → {args.out}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
