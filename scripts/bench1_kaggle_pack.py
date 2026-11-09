"""Pack the private Kaggle dataset `ctxgraph-bench1`: the ctxgraph package, the bench runner, WordNet data.

    python scripts/bench1_kaggle_pack.py kaggle/bench1/data
    kaggle datasets create -p kaggle/bench1/data --dir-mode zip        # first time (private by default)
    kaggle datasets version -p kaggle/bench1/data --dir-mode zip -m "..."  # updates
    kaggle kernels push -p kaggle/bench1                                # run (after Egor's "yes")
    kaggle kernels push -p kaggle/bench1b                               # bench1b (MODE in run.py: debug / test)

Only the listed paths are copied; no keys, no .env, no project dialogues. Benchmark data is downloaded
inside the kernel from Hugging Face at pinned revisions (see scripts/bench1_run.py).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["ctxgraph", "scripts/bench1_run.py", "scripts/bench1b_run.py", "scripts/bench1b_config.json",
         "scripts/evermem_stack.py", "scripts/implicit_run.py", "scripts/imlogic_run.py", "scripts/implexconv_run.py",
         "scripts/planner.py", "scripts/stage2_report.py",
         "reports/implicit"]
#: Kaggle skips hidden folders on upload, so WordNet data travels as `nltk_data/` (kernel restores it)
RENAMED = {".cache/nltk": "nltk_data", ".cache/bench/imlogic": "imlogic_data"}


def main() -> None:
    out = Path(sys.argv[1])
    if out.exists():
        shutil.rmtree(out)
    for p in PATHS:
        s, d = ROOT / p, out / p
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "archive"))
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    for src, dst in RENAMED.items():
        shutil.copytree(ROOT / src, out / dst)
    # NLTK keeps both «x/» and «x.zip» for some packages; a dataset with both names is never versioned on Kaggle
    for z in (out / "nltk_data").rglob("*.zip"):
        if z.with_suffix("").is_dir():
            z.unlink()
    meta = {"title": "ctxgraph-bench1", "id": "egor123hhh/ctxgraph-bench1", "licenses": [{"name": "other"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta), "utf-8")
    bad = [f for f in out.rglob("*") if f.name.startswith(".env") or "secret" in f.name.lower()]
    assert not bad, bad
    print(out, sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // 1024, "KB")


if __name__ == "__main__":
    main()
