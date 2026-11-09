"""Pack the private Kaggle dataset `ctxgraph-stage2-data` for the stage-2 CPU kernel (amendment 1).

    python scripts/stage2_data_pack.py kaggle/stage2data/data
    kaggle datasets create -p kaggle/stage2data/data           # first time; later: datasets version ... -m "..."

Flat file names (Kaggle unpacks folders unpredictably): `cache__<name>.jsonl` (planner answer caches),
`e5__<corpus>__e5.npy` / `e5__<corpus>__e5_fids.json` (ImplexConv main e5 fragments), `graph__<user>.sqlite`
(IMLogic main graphs). The kernel restores them into the layout `implicit_run.py` expects.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "out" / "implicit_cpu"


def main() -> None:
    out = Path(sys.argv[1])
    with_e5 = "--no-e5" not in sys.argv
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in (ROOT / "out" / "planner_cache").glob("*_main.jsonl"):
        shutil.copy2(f, out / f"cache__{f.name}")
    n_e5 = 0
    for d in sorted((SRC / "implexconv_main").iterdir()) if with_e5 else ():
        if d.is_dir() and (d / "e5_fids.json").exists():
            shutil.copy2(d / "e5.npy", out / f"e5__{d.name}__e5.npy")
            shutil.copy2(d / "e5_fids.json", out / f"e5__{d.name}__e5_fids.json")
            n_e5 += 1
    n_g = 0
    for d in sorted((SRC / "imlogic_main").iterdir()):
        if d.is_dir() and (d / "graph.sqlite").exists():
            shutil.copy2(d / "graph.sqlite", out / f"graph__{d.name}.sqlite")
            n_g += 1
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": "ctxgraph-stage2-data", "id": "egor123hhh/ctxgraph-stage2-data", "licenses": [{"name": "other"}]}), "utf-8")
    size = sum(f.stat().st_size for f in out.iterdir()) / 2**20
    print(f"{out}: {n_e5} ImplexConv e5 corpora, {n_g} IMLogic graphs, {size:.0f} MB")


if __name__ == "__main__":
    main()
