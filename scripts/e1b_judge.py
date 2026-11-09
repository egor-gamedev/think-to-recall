"""Оценка ответов E1b судьёй через OpenAI-совместимый API (по умолчанию — NVIDIA).

    python scripts/e1b_judge.py --answers reports/e1b/pilot2 --out reports/e1b/pilot2/judge_a.jsonl \
        --model deepseek-ai/deepseek-v3.1 --limit 60
    # второй судья на доле ответов: --sample 0.2

Порядок ответов перемешан (seed 0), вариант судье не показывается. Повторный запуск
пропускает уже оценённые ответы. Ключ — только из .env (переменная --key-env).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1.llm import Client  # noqa: E402
from ctxgraph.e1b.judge import RequestLedger, grade, load_spec  # noqa: E402


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.lstrip().startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def key_of(r: dict) -> str:
    return f"{r['project']}|{r['question']}|{r['variant']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="папка с <project>.jsonl от e1b_run.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--sample", type=float, default=1.0, help="доля ответов (детерминированно)")
    ap.add_argument("--limit", type=int, required=True, help="максимум запросов за запуск")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4, help="параллельных запросов")
    ap.add_argument("--timeout", type=float, default=400, help="секунд на запрос")
    ap.add_argument("--wall", type=float, default=900, help="предел секунд на ответ с повторами")
    ap.add_argument("--rubric", default="e1b.3", help="e1b.3 (E1b) или e1c.1 (строгий «вывод»)")
    ap.add_argument(
        "--ledger-rub", type=float, default=0, help="платный API: лимит ₽ по учёту (Ledger E1)"
    )
    ap.add_argument(
        "--extra",
        default="{}",
        help='доп. параметры запроса, JSON (напр. {"reasoning_effort":"none"})',
    )
    ap.add_argument("--projects", default="", help="только эти проекты (через запятую)")
    ap.add_argument(
        "--by-question", action="store_true", help="варианты одного вопроса подряд (для лимита ₽)"
    )
    a = ap.parse_args()
    load_env()

    recs = []
    for f in sorted(Path(a.answers).glob("*.jsonl")):
        if f.name.endswith(".calls.jsonl") or f.name.startswith("judge"):
            continue
        recs += [json.loads(ln) for ln in f.read_text("utf-8").splitlines() if ln.strip()]
    recs = [r for r in recs if "variant" in r]
    if a.sample < 1:
        recs = [
            r
            for r in recs
            if int(hashlib.sha256(key_of(r).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < a.sample
        ]
    if a.projects:
        keep = set(a.projects.split(","))
        recs = [r for r in recs if r["project"] in keep]
    if a.by_question:
        # все варианты вопроса подряд, вопросы — в случайном порядке: если упрёмся в лимит ₽,
        # оценённые вопросы будут полными парами
        qs = sorted({(r["project"], r["question"]) for r in recs})
        random.Random(0).shuffle(qs)
        rank = {k: i for i, k in enumerate(qs)}
        recs.sort(key=lambda r: (rank[(r["project"], r["question"])], r["variant"]))
    else:
        random.Random(0).shuffle(recs)
    out = Path(a.out)
    done = set()
    if out.exists():
        rows = [json.loads(ln) for ln in out.read_text("utf-8").splitlines() if ln.strip()]
        done = {r["key"] for r in rows if "error" not in r}  # ошибки дооцениваются
    todo = [r for r in recs if key_of(r) not in done]
    print(f"ответов {len(recs)}, уже оценено {len(done)}, к оценке {len(todo)}", flush=True)

    if a.ledger_rub:  # платный API (timeweb): учёт в рублях с жёстким лимитом, как в E1
        from ctxgraph.e1.budget import Ledger

        ledger = Ledger(out.with_suffix(".spend.jsonl"), limit_rub=a.ledger_rub)
        ledger.n = 0
    else:
        ledger = RequestLedger(out.with_suffix(".requests.jsonl"), limit_requests=10**9)
    client = Client(
        a.model, ledger, phase="e1b-judge", base_url=a.base_url,
        key=os.environ[a.key_env], params={"temperature": 0, "max_tokens": a.max_tokens, **json.loads(a.extra)}, timeout=a.timeout,
    )  # fmt: skip
    specs = {r["project"]: load_spec(ROOT, r["project"]) for r in todo}
    todo = todo[: a.limit]  # один запрос на ответ (+ редкий повтор при битом JSON)
    lock, stop = threading.Lock(), threading.Event()
    start = ledger.n

    def one(i_r):
        i, r = i_r
        spec = specs[r["project"]]
        q = next(x for x in spec["questions"] if x["id"] == r["question"])
        # жёсткий предел на ответ: зависшее соединение NVIDIA не должно держать поток часами
        box: dict = {}

        def call() -> None:
            try:
                box["v"] = grade(client, spec, q, r["answer"], run_id=key_of(r), rubric=a.rubric)
            except Exception as e:  # noqa: BLE001 — ответ останется неоценённым, перезапуск дооценит
                box["e"] = e

        if stop.is_set():
            return
        t = threading.Thread(target=call, daemon=True)
        t.start()
        t.join(a.wall)
        if "e" in box and type(box["e"]).__name__ == "BudgetExceeded":
            stop.set()  # лимит ₽ исчерпан: дальше не оцениваем, перезапуск после пополнения дооценит
            print(f"ОСТАНОВ: {box['e']}", flush=True)
            return
        if "v" not in box:
            print(
                f"{i + 1}/{len(todo)} {key_of(r)}: ОШИБКА {box.get('e', 'нет ответа за --wall')}",
                flush=True,
            )
            return
        v = box["v"]
        row = {"key": key_of(r), "project": r["project"], "question": r["question"],
               "type": r["type"], "variant": r["variant"]} | v  # fmt: skip
        with lock:
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{i + 1}/{len(todo)} {row['key']}: "
                  + ("ОШИБКА " + v["error"] if "error" in v else
                     f"вывод={v['inference']} неуместн={v['inappropriate']} выдумка={v['fabrication']}"),
                  flush=True)  # fmt: skip

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, enumerate(todo)))
    print(f"запросов за запуск: {ledger.n - start}", flush=True)


if __name__ == "__main__":
    main()
