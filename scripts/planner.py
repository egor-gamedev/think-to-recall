"""Stage 2 planner: one LLM call per question -> <= 3 reasoning steps, each a retrieval query ("sleeve").

    python scripts/planner.py --dataset imlogic --split dev --limit 20    # smoke test
    python scripts/planner.py --dataset imlogic --split dev               # -> reports/implicit/plans_<dataset>_<split>.jsonl

Free NVIDIA API (`ctxgraph.e1b.backends.NvidiaChat`: temperature 0, cached by request hash, so a rerun makes no calls).
An unparsable reply = no steps (the question then has no sleeves); the share is reported.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

MODEL = "mistralai/mistral-large-4-1t-a52b"
PROMPT = """A user asks their personal assistant the question below. The assistant has a long-term memory of past \
conversations with this user. Before answering, plan what to look up in that memory: up to 3 short search queries, \
each for one past fact about the user (a goal, constraint, preference, plan, condition or earlier decision) that \
could change the right answer. Think about what is NOT stated in the question but would matter.

Question: {question}

Reply with JSON only: {{"steps": ["query 1", "query 2", "query 3"]}}"""


def parse(text: str) -> list[str] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        steps = json.loads(m.group(0)).get("steps")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(steps, list):
        return None
    steps = [s.strip() for s in steps if isinstance(s, str) and s.strip()][:3]
    return steps or None


def question_order(dataset: str, qs: list[dict]) -> list[dict]:
    """Amendment 1 (prereg_stage2): ImplexConv questions in a seed-0 shuffle of their sorted ids, so any prefix
    mixes opposed and supportive; IMLogic in loader order (user by user, so whole users finish first)."""
    if dataset != "implexconv":
        return list(qs)
    import random

    ids = sorted(q["id"] for q in qs)
    random.Random(0).shuffle(ids)
    by = {q["id"]: q for q in qs}
    return [by[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imlogic", "implexconv"], required=True)
    ap.add_argument("--split", choices=["dev", "main"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--max-pause", type=float, default=None, help="exponential back-off cap, s (default: NvidiaChat's)")
    ap.add_argument("--data", default=str(ROOT / ".cache" / "bench"))
    ap.add_argument("--cache", default=None, help="answer cache .jsonl (default out/planner_cache/<ds>_<split>.jsonl)")
    ap.add_argument("--out", default=None, help="plans .jsonl (default reports/implicit/plans_<ds>_<split>.jsonl)")
    ap.add_argument("--deadline", type=float, default=None, help="unix time: no new request after it")
    a = ap.parse_args()
    import time
    from concurrent.futures import ThreadPoolExecutor

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import implicit_run as R

    from ctxgraph.e1b.backends import NvidiaChat

    cs = (R.imlogic if a.dataset == "imlogic" else R.implexconv)(a.split, Path(a.data))
    qs = question_order(a.dataset, [q for c in cs for q in c["qs"]])[: a.limit]
    cache = a.cache or str(ROOT / "out" / "planner_cache" / f"{a.dataset}_{a.split}.jsonl")
    llm = NvidiaChat(MODEL, cache, max_pause=a.max_pause)
    out = Path(a.out) if a.out else ROOT / "reports" / "implicit" / f"plans_{a.dataset}_{a.split}.jsonl"

    def plan(q: dict) -> dict:
        base = {"id": q["id"], "question": q["question"]}
        if a.deadline is not None and time.time() > a.deadline:
            llm.tries = 0  # past the deadline: cached answers are still read, no new request is sent
        try:
            r = llm.chat([{"role": "user", "content": PROMPT.format(question=q["question"])}], max_tokens=3000)
        except RuntimeError as e:  # all retries timed out (or past the deadline): not cached, retried in the next pass
            return base | {"steps": None, "finish": "skipped" if llm.tries == 0 else "failed", "error": str(e)[:200]}
        return base | {"steps": parse(r["content"]), "finish": r.get("finish_reason")}

    res: dict[str, dict] = {}
    todo = qs
    for rnd in range(a.passes):  # one network failure must not kill the run: failed questions are retried
        with ThreadPoolExecutor(a.workers) as pool:  # a call takes ~1-2 min (hidden reasoning): run them in parallel
            for r in pool.map(plan, todo):
                res[r["id"]] = r
        todo = [q for q in qs if res[q["id"]]["finish"] == "failed"]
        print(f"pass {rnd + 1}: answered {llm.api_calls}, failed questions {len(todo)}, "
              f"failed attempts {llm.failed_attempts}", flush=True)  # fmt: skip
        if not todo or (a.deadline is not None and time.time() > a.deadline):
            break
    # only answered questions are written (amendment 1: a failed / skipped request is NOT "no steps")
    done = [res[q["id"]] for q in qs if res[q["id"]]["finish"] not in ("failed", "skipped")]
    bad = sum(r["steps"] is None for r in done)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in done:
            fh.write(json.dumps(r | {"steps": r["steps"] or []}, ensure_ascii=False) + "\n")
    print(f"{a.dataset} {a.split}: {len(qs)} questions, answered {len(done)}, new API calls {llm.api_calls}, "
          f"unparsable {bad}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
