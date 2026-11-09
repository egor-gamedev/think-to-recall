"""Recompute the main result tables from the saved per-question outputs in `reports/` (no GPU, no API, ~1 min).

    python scripts/reproduce_tables.py

Same test as in every pre-registration: one-sided paired sign-flip permutation (10 000, seed 0), Holm, 95 % bootstrap.
Prints bench1b H1/H2 (MuSiQue, primary code d707ea0), stage 1 on IMLogic (C-A, D-D0) and stage 2 (H1 SC-A, H2 SR-SC,
H3 SR-A). Full reruns (Kaggle kernels, planner calls) are described in README.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import bench1_run as B  # noqa: E402
import bench1b_run as BB  # noqa: E402

R = ROOT / "reports"


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def line(name: str, d: list[float], p_holm: float | None = None) -> str:
    m, lo, hi = B.bootstrap(d)
    s = f"  {name:<34} {100 * m:+5.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]  wins/losses {sum(v > 0 for v in d)}/{sum(v < 0 for v in d)}"
    s += f"  p {BB.perm_p(d):.4f}" + (f"  Holm {p_holm:.4f}" if p_holm is not None else "")
    return s + f"  p_reverse {BB.perm_p([-v for v in d]):.3f}  n {len(d)}"


def main() -> None:
    print("bench1b, MuSiQue R@5 (primary: code d707ea0)")
    rows = jl(R / "bench1b" / "test" / "musique_recall.jsonl")
    d = {"H1 gate over dense (G - d0)": [r["G"]["r5"] - r["d0"]["r5"] for r in rows],
         "H2 gate over rerank (KG - K)": [r["KG"]["r5"] - r["K"]["r5"] for r in rows]}  # fmt: skip
    h = BB.holm({k: BB.perm_p(v) for k, v in d.items()})
    for k, v in d.items():
        print(line(k, v, h[k]))

    print("\nstage 1, IMLogic main (memory_l R@5): graph from the question")
    im = {v: {r["id"]: r for r in jl(R / "implicit" / "main" / "imlogic" / f"{v}.jsonl")} for v in ("A", "C", "D0", "D")}
    ids = sorted(im["A"])
    for name, x, y in (("S1 C - A", "C", "A"), ("S2 D - D0", "D", "D0")):
        print(line(name, [B.recall(im[x][i]["rank"], im[x][i]["gold"], 5) - B.recall(im[y][i]["rank"], im[y][i]["gold"], 5) for i in ids]))

    print("\nstage 2 (base A): plan steps; main analysis (ImplexConv 300 seed-0, IMLogic 3 whole users)")
    sets = {}
    for ds in ("implexconv", "imlogic"):
        d2 = R / "implicit" / "stage2_main" / f"{ds}_main"
        rr = {v: {r["id"]: r for r in jl(d2 / f"{v}_A.jsonl")} for v in ("S0", "SC", "SR")}
        ids = sorted(rr["S0"])
        f = lambda x, y, rr=rr, ids=ids: [B.recall(rr[x][i]["rank"], rr[x][i]["gold"], 5) - B.recall(rr[y][i]["rank"], rr[y][i]["gold"], 5) for i in ids]  # noqa: E731
        sets[ds] = {"H1 SC - A": f("SC", "S0"), "H2 SR - SC": f("SR", "SC"), "H3 SR - A": f("SR", "S0")}
    h1 = BB.holm({ds: BB.perm_p(v["H1 SC - A"]) for ds, v in sets.items()})
    h23 = BB.holm({f"{ds}|{t}": BB.perm_p(v[t]) for ds, v in sets.items() for t in ("H2 SR - SC", "H3 SR - A")})
    for ds, v in sets.items():
        print(f" {ds}")
        print(line("H1 SC - A", v["H1 SC - A"], h1[ds]))
        for t in ("H2 SR - SC", "H3 SR - A"):
            print(line(t, v[t], h23[f"{ds}|{t}"]))


if __name__ == "__main__":
    main()
