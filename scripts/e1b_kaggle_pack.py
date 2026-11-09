"""Собрать датасет Kaggle `ctxgraph-e1b`: пакет ctxgraph, раннер, диалоги E1b, данные WordNet.

    python scripts/e1b_kaggle_pack.py <папка>   # затем: kaggle datasets create|version -p <папка> --dir-mode zip
Ключи и .env не попадают: копируются только перечисленные пути.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["ctxgraph", "scripts/e1b_run.py", "benchmark_e1b/projects", ".cache/nltk"]


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
    meta = {
        "title": "ctxgraph-e1b",
        "id": "egor123hhh/ctxgraph-e1b",
        "licenses": [{"name": "other"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(meta), "utf-8")
    bad = [f for f in out.rglob("*") if f.name.startswith(".env")]
    assert not bad, bad
    print(out, sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // 1024, "КБ")


if __name__ == "__main__":
    main()
