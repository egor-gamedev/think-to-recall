"""Пары (запрос, предложение) для реранкера этапа A4 — для расчёта на GPU Kaggle.

    python scripts/e1d_rerank_pairs.py             # → kaggle/e1d_rerank/data/pairs.jsonl (+ dataset-metadata.json)
    kaggle datasets create -p kaggle/e1d_rerank/data   # приватный датасет
    kaggle kernels push -p kaggle/e1d_rerank           # GPU: scores.jsonl
    python scripts/e1d_rerank_pairs.py --merge <scores.jsonl>   # → .cache/e1d_rerank.jsonl

Пары — ровно те, что запросит `scripts/e1d_stage_a4.py`: для каждого вопроса (20 пробных + 200
основных) предложения пула V6 и пула V2+ (гибрид без графа), запрос = текст вопроса. Уже
посчитанные в `.cache/e1d_rerank.jsonl` пропускаются. Ключи и .env не копируются.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import e1d_stage_a as A  # noqa: E402
import e1d_stage_a2 as A2  # noqa: E402

from ctxgraph.e1b import selection as SEL  # noqa: E402

DATA = ROOT / "kaggle" / "e1d_rerank" / "data"
CACHE = ROOT / ".cache" / "e1d_rerank.jsonl"


def cached() -> set[tuple[str, str]]:
    if not CACHE.exists():
        return set()
    return {tuple(json.loads(ln)[:2]) for ln in CACHE.read_text("utf-8").splitlines() if ln.strip()}


def dump() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b.backends import E5Encoder

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    enc = A.CachedEncoder(E5Encoder())
    known = json.loads(A.PREFIX.read_text("utf-8"))
    items = {pid: A.reconstruct(pid, enc, count, known) for pid in A.NEW + A.OLD}
    pairs = [(pid, it) for pid, it in A2.probe_items(items)] + [(pid, it) for pid in A.NEW for it in items[pid]]
    done, out = cached(), {}
    for _, it in pairs:
        q = it["q"]["text"]
        for kind in ("v6", "hybrid"):
            for s in SEL.pool_sentences(it["arc"], q, 30, kind):
                if (q, s.text) not in done:
                    out[(q, s.text)] = None
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "pairs.jsonl").open("w", encoding="utf-8") as fh:
        for q, t in out:
            fh.write(json.dumps([q, t], ensure_ascii=False) + "\n")
    meta = {"title": "ctxgraph-e1d-rerank", "id": "egor123hhh/ctxgraph-e1d-rerank", "licenses": [{"name": "other"}]}
    (DATA / "dataset-metadata.json").write_text(json.dumps(meta), "utf-8")
    print(f"пар к расчёту: {len(out)} (уже в кэше: {len(done)})")


def merge(path: str) -> None:
    done = cached()
    n = 0
    with CACHE.open("a", encoding="utf-8") as fh:
        for ln in Path(path).read_text("utf-8").splitlines():
            q, t, sc = json.loads(ln)
            if (q, t) not in done:
                fh.write(json.dumps([q, t, sc], ensure_ascii=False) + "\n")
                n += 1
    print(f"добавлено в кэш: {n}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--merge":
        merge(sys.argv[2])
    else:
        dump()
