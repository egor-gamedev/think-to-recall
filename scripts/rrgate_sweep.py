"""Stage 2 dev: replay the reranker-aware gate over D0 from `rrtrace.jsonl` (`implicit_run.py --stage rrtrace`).

    python scripts/rrgate_sweep.py out/kaggle_rrgate/implicit/imlogic_dev/rrtrace.jsonl out/.../implexconv_dev/rrtrace.jsonl

Rule (fixed before looking at any trace): a graph candidate displaces a base entry only if
rr(candidate) >= rr(displaced) - delta (rr = Qwen3-Reranker-4B P(yes) against the question). Grid DELTAS; delta = 1
equals the old gate (P(yes) is in [0, 1]). Choice: among deltas where, on EVERY dev set, both D_new and SG_new have
losses <= wins against D0 (R@5), take the one with the largest summed SG_new - D0 R@5 gain; ties -> the smaller
delta (stricter). No eligible delta -> sleeves are not run on the main part (Egor, 2026-11-07).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import bench1_run as B  # noqa: E402

from ctxgraph.v2.gate import PprGateConfig, apply_picked, pick  # noqa: E402

DELTAS = [-0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
CFG = PprGateConfig(**json.loads((ROOT / "scripts" / "bench1b_config.json").read_text("utf-8"))["gate"])


def replay(r: dict, key: str, delta: float | None, rr: str = "rr_q") -> list[int]:
    t = r[key]
    if not t["order"]:
        return r["rank"]
    pr = {int(p): v for p, v in t["pr"].items()}
    sc = {int(p): v for p, v in r[rr].items()}
    admit = None if delta is None else (lambda c, d: sc.get(c, 0.0) >= sc.get(d, 0.0) - delta)
    return apply_picked(r["rank"], pick(r["rank"], t["order"], pr, CFG, admit), CFG)


def compare(rows: list[dict], ranks: list[list[int]]) -> dict:
    d = [B.recall(x, r["gold"], 5) - B.recall(r["rank"], r["gold"], 5) for x, r in zip(ranks, rows, strict=True)]
    return {"r5": 100 * sum(B.recall(x, r["gold"], 5) for x, r in zip(ranks, rows, strict=True)) / len(rows),
            "diff": 100 * sum(d) / len(d), "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d)}  # fmt: skip


def main() -> None:
    sets = {Path(p).parent.name: [json.loads(x) for x in Path(p).read_text("utf-8").splitlines() if x.strip()]
            for p in sys.argv[1:]}  # fmt: skip
    res: dict = {}
    for name, rows in sets.items():
        res[name] = {
            "n": len(rows), "empty_plans": sum(r["steps"] == 0 for r in rows),
            "D0": compare(rows, [r["rank"] for r in rows]),
            "SR": compare(rows, [r["sr"] for r in rows]),
            "D_old": compare(rows, [replay(r, "old", None) for r in rows]),
            "SG_old": compare(rows, [replay(r, "sg", None) for r in rows]),
            "by_delta": {str(dl): {"D_new": compare(rows, [replay(r, "old", dl) for r in rows]),
                                   "SG_new": compare(rows, [replay(r, "sg", dl) for r in rows]),
                                   "SG_new_steps_rr": compare(rows, [replay(r, "sg", dl, "rr_steps") if r["rr_steps"]
                                                                     else replay(r, "sg", dl) for r in rows])}
                         for dl in DELTAS},
        }  # fmt: skip
    ok = [dl for dl in DELTAS if all(v["by_delta"][str(dl)][k]["losses"] <= v["by_delta"][str(dl)][k]["wins"]
                                     for v in res.values() for k in ("D_new", "SG_new"))]  # fmt: skip
    best = max(ok, key=lambda dl: (sum(v["by_delta"][str(dl)]["SG_new"]["diff"] for v in res.values()), -dl), default=None)
    res["eligible"], res["chosen_delta"] = ok, best
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
