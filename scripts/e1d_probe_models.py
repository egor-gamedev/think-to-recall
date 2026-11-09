"""E1d этап C: проба моделей-отвечающих на протоколе шагов V9 (5 вопросов на модель).

    python scripts/e1d_probe_models.py --models nvidia:mistralai/mistral-large-4-1t-a52b ...
    → reports/e1d/probe/<модель>.jsonl, reports/e1d/probe/summary.md

Вопросы — не основные: старый проект E1b crypto-wallet, 5 вопросов «связать» (Q04, Q07, Q08, Q09,
Q11). Вариант 91 (V9, рукава-фразы из шагов + общий отбор). Меряем: задержку вызова, зависания и
ошибки (жёсткий предел на вызов, 2 попытки), держит ли протокол шагов (пустые шаги, ответ вместо
рассуждения, упор в лимит токенов шага), обрывы ответа.

Модель: «nvidia:<id>[:think]» — бесплатный API NVIDIA; «timeweb:<id>» — платный timeweb с жёстким
лимитом в рублях (`--rub`, по учёту ctxgraph.e1.budget).
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

OUT = ROOT / "reports" / "e1d" / "probe"
PID, QS, VARIANT = "crypto-wallet", ["Q04", "Q07", "Q08", "Q09", "Q11"], 91
REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def load_env() -> None:
    for ln in (ROOT / ".env").read_text("utf-8").splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Timed:
    """Обёртка модели: время и исход каждого вызова (ключ API в лог не попадает)."""

    def __init__(self, llm, count):
        self.llm, self.count, self.log = llm, count, []

    def chat(self, messages, max_tokens, tools=None):
        t0 = time.time()
        try:
            r = self.llm.chat(messages, max_tokens, tools)
        except Exception as e:  # noqa: BLE001 — зависание/ошибка API фиксируется и пробрасывается
            self.log.append({"s": round(time.time() - t0, 1), "error": str(e)[:200], "max_tokens": max_tokens})
            raise
        self.log.append({"s": round(time.time() - t0, 1), "finish": r.get("finish_reason"), "max_tokens": max_tokens,
                         "out": (r.get("usage") or {}).get("completion_tokens")})  # fmt: skip
        return r


class TimewebChat:
    """timeweb (OpenAI-совместимый) с учётом в рублях и дисковым кэшем — интерфейс как у NvidiaChat."""

    def __init__(self, model: str, cache: Path, rub: float):
        from ctxgraph.e1.budget import Ledger
        from ctxgraph.e1.llm import Client

        self.model = model
        self.ledger = Ledger(OUT / "timeweb.spend.jsonl", limit_rub=rub)
        self.client = Client(model, self.ledger, phase="e1d-probe", timeout=180)
        self.cache_path, self.cache = cache, {}
        if cache.exists():
            for ln in cache.read_text("utf-8").splitlines():
                d = json.loads(ln)
                self.cache[d["k"]] = d["r"]

    def chat(self, messages, max_tokens, tools=None):
        import hashlib

        k = hashlib.sha256(json.dumps([self.model, messages, max_tokens], ensure_ascii=False).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        rep = self.client.chat(messages, purpose="probe", run_id=k[:12], extra={"max_tokens": max_tokens, "temperature": 0})
        res = {"content": rep.content, "tool_calls": [], "usage": rep.usage,
               "finish_reason": (rep.raw_message or {}).get("finish_reason")}  # fmt: skip
        self.cache[k] = res
        with self.cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "r": res}, ensure_ascii=False) + "\n")
        return res


def run_one(spec: str, rub: float) -> dict:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b import selection as SEL
    from ctxgraph.e1b.backends import E5Encoder, NvidiaChat
    from ctxgraph.e1b.chat import Limits
    from ctxgraph.e1b.runner import Session, run_project

    prov, model, *flags = spec.split(":")
    name = model.split("/")[-1] + ("-think" if "think" in flags else "")
    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    cache = OUT / f"{name}.llm_cache.jsonl"
    if prov == "nvidia":
        llm = NvidiaChat(model, str(cache), thinking="think" in flags, wall=180, tries=2)
    else:
        llm = TimewebChat(model, cache, rub)
    timed = Timed(llm, count)
    sess = Session(timed, count, E5Encoder(), Limits(context=6600))
    rr = SEL.Reranker(revision=REV)
    rr.persist(ROOT / ".cache" / "e1d_probe_rerank.jsonl")
    sess.scorer = rr.score
    out = OUT / f"{name}.jsonl"
    out.unlink(missing_ok=True)
    t0, err = time.time(), None
    try:
        run_project(ROOT, PID, sess, variants=(VARIANT,), questions=QS, out=out)
    except Exception as e:  # noqa: BLE001
        err = str(e)[:300]
    recs = [json.loads(x) for x in out.read_text("utf-8").splitlines()] if out.exists() else []
    summ = {"model": spec.replace(":think", " (рассуждения)"), "name": name, "questions_done": len(recs),
            "error": err, "minutes": round((time.time() - t0) / 60, 1), "calls": timed.log}  # fmt: skip
    (OUT / f"{name}.summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), "utf-8")
    return summ


def summary_md() -> str:
    import numpy as np

    rows = []
    for p in sorted(OUT.glob("*.summary.json")):
        s = json.loads(p.read_text("utf-8"))
        recs = [json.loads(x) for x in (OUT / f"{s['name']}.jsonl").read_text("utf-8").splitlines()] \
            if (OUT / f"{s['name']}.jsonl").exists() else []  # fmt: skip
        ok = [c for c in s["calls"] if "error" not in c]
        errs = [c for c in s["calls"] if "error" in c]
        steps = [c for c in ok if c["max_tokens"] == 250]
        prot = {k: sum((r["steps"] or {}).get("protocol", {}).get(k, 0) for r in recs)
                for k in ("empty_steps", "answer_in_step", "long_steps")}  # fmt: skip
        cut = sum(r.get("finish_reason") == "length" for r in recs)
        lat = [c["s"] for c in ok]
        rows.append(f"| {s['model']} | {s['questions_done']}/5 | {len(ok)} / {len(errs)} | "
                    f"{np.median(lat) if lat else float('nan'):.1f} / {max(lat) if lat else float('nan'):.0f} | "
                    f"{prot['empty_steps']} / {prot['answer_in_step']} / {prot['long_steps']} из {len(steps)} | {cut} | "
                    f"{s['minutes']} | {(s['error'] or '—')[:80]} |")  # fmt: skip
    head = ["# E1d этап C — проба моделей-отвечающих (V9, 5 вопросов crypto-wallet)\n",
            "| модель | вопросов | вызовов ок / ошибок | задержка, с: медиана / макс | шаги: пустые / ответ вместо шага / в лимит | обрывов ответа | минут | ошибка |",
            "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    return "\n".join(head + rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--rub", type=float, default=40.0, help="лимит ₽ на timeweb для пробы")
    ap.add_argument("--one", default="")
    a = ap.parse_args()
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.one:
        print(json.dumps({k: v for k, v in run_one(a.one, a.rub).items() if k != "calls"}, ensure_ascii=False))
        return
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1")
    for m in a.models:  # по одной модели в процессе: CPU-часть (e5, реранкер) не делится между потоками
        subprocess.call([sys.executable, __file__, "--one", m, "--rub", str(a.rub)], env=env)
    (OUT / "summary.md").write_text(summary_md(), "utf-8")
    print(summary_md())


if __name__ == "__main__":
    main()
