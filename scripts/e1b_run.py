"""Прогон E1b: проекты × варианты × вопросы. На Kaggle — `--backend vllm`, локально — `--backend fake`.

python scripts/e1b_run.py --backend vllm --projects crypto-wallet --questions Q01,Q02,Q03,Q04,Q05 --out out/pilot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1b.chat import Limits, Turn  # noqa: E402
from ctxgraph.e1b.runner import VARIANTS, Session, load_project, run_project  # noqa: E402

PROJECTS = ("crypto-wallet", "django-shop", "rnaseq-thesis")


class FakeLLM:
    """Для сухого прогона: отвечает заглушкой, считает токены грубо."""

    model = "fake"

    def count(self, text: str) -> int:
        return max(1, len(text) // 3)

    def chat(self, messages, max_tokens, tools=None):
        n = sum(len(m.get("content") or "") for m in messages) // 3
        return {"content": "заглушка", "tool_calls": [], "usage": {"prompt_tokens": n}}


def main() -> None:
    # фиксированное зерно хэша: порядок множеств одинаков в любом запуске (кэш модели совпадает)
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        sys.exit(subprocess.call([sys.executable, *sys.argv]))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend", choices=("nvidia", "server", "auto", "vllm", "hf", "fake"), default="fake"
    )
    ap.add_argument("--projects", default=",".join(PROJECTS))
    ap.add_argument("--variants", default=",".join(map(str, VARIANTS)))
    ap.add_argument("--questions", default="")
    ap.add_argument("--context", type=int, default=6600)
    ap.add_argument("--encoder", choices=("e5", "model2vec", "hashing"), default="e5")
    ap.add_argument("--out", default="out/e1b")
    ap.add_argument(
        "--model", default="nvidia/nemotron-3-ultra-550b-a55b", help="для --backend nvidia"
    )
    ap.add_argument("--thinking", action="store_true", help="рассуждения модели (NVIDIA)")
    ap.add_argument("--workers", type=int, default=0, help="параллельных вариантов (0 — все)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import faulthandler  # при зависании — стеки всех потоков раз в 15 мин

    _stacks = (out / "stacks.log").open("w")
    faulthandler.dump_traceback_later(900, repeat=True, file=_stacks)
    llm = None
    if a.backend == "nvidia":
        from scripts.e1b_judge import load_env  # ключ — только из .env

        load_env()
        from ctxgraph.e1b.backends import NvidiaChat

        llm = NvidiaChat(a.model, str(out / "llm_cache.jsonl"), thinking=a.thinking)
    if a.backend == "server":
        from ctxgraph.e1b.backends import QwenServer

        llm = QwenServer()
    if a.backend in ("auto", "vllm"):
        try:
            from ctxgraph.e1b.backends import QwenVLLM

            llm = QwenVLLM()
        except Exception as e:  # noqa: BLE001 — на auto откатываемся на transformers
            if a.backend == "vllm":
                raise
            print(f"vLLM не поднялся ({type(e).__name__}: {e}); беру transformers", flush=True)
    if llm is None and a.backend in ("auto", "hf"):
        from ctxgraph.e1b.backends import QwenHF

        llm = QwenHF()
    if llm is None:
        llm = FakeLLM()
    if a.encoder == "e5":
        from ctxgraph.e1b.backends import E5Encoder

        enc = E5Encoder()
    elif a.encoder == "model2vec":
        from ctxgraph.embed import Model2VecEncoder

        enc = Model2VecEncoder()
    else:
        from ctxgraph.embed import HashingEncoder

        enc = HashingEncoder()
    limits = Limits(context=a.context)
    variants = tuple(int(v) for v in a.variants.split(","))
    questions = [q for q in a.questions.split(",") if q] or None
    pids = a.projects.split(",")
    meta = {
        "backend": a.backend,
        "model": getattr(llm, "model", "?"),
        "encoder": enc.name,
        "limits": limits.__dict__ | {"window": limits.window},
        "variants": variants,
        "questions": questions,
        "projects": pids,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")

    for pid in pids:
        # V5: реплики чужих проектов — источник случайных узлов той же формы
        pool = [
            Turn(t["id"], t["role"], t["text"], "load", 0)
            for other in PROJECTS
            if other != pid
            for t in load_project(ROOT, other)[1]
        ]
        sess = Session(llm, llm.count, enc, limits)
        t0 = time.time()
        # при повторном запуске ответы берутся из кэша модели — файл записей пишется заново
        (out / f"{pid}.jsonl").unlink(missing_ok=True)
        recs = run_project(
            ROOT,
            pid,
            sess,
            variants=variants,
            pool=pool,
            questions=questions,
            out=out / f"{pid}.jsonl",
            parallel=a.backend in ("server", "fake", "nvidia"),
            workers=a.workers,
        )
        (out / f"{pid}.load.json").write_text(
            json.dumps(recs[0], ensure_ascii=False, indent=1), "utf-8"
        )
        with (out / f"{pid}.calls.jsonl").open("w", encoding="utf-8") as fh:
            for c in sess.calls:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(
            f"{pid}: {len(recs) - 1} ответов, сжатий при загрузке {len(recs[0]['compressions'])}, "
            f"{time.time() - t0:.0f} с",
            flush=True,
        )


if __name__ == "__main__":
    main()
