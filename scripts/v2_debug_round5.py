"""Debug round 5 (CPU, free NVIDIA calls): answers with/without memory, step planner, light rerankers.

    python scripts/v2_debug_round5.py [--skip-llm] [--skip-rerankers]   # -> reports/v2_debug_round5.md

A. Same question + step, same model and settings: answer with memory (hybrid main block + periphery)
   and without memory — one call each.
B. Step planner (IRCoT + Self-Ask, `ctxgraph/v2/planner.py`): the model writes <= 3 steps from the
   question and the first memory block; sleeves come from those steps; "X of 7" relevant facts.
C. Light multilingual rerankers vs bge-reranker-v2-m3 on CPU: time per query and "X of 7" (fill-to-
   budget mode — score scales differ per model, so the threshold is not comparable across models).
Debug only — no quality claims.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_debug_loss as LO  # noqa: E402
import v2_debug_pq as P  # noqa: E402
import v2_debug_round3 as R3  # noqa: E402

from ctxgraph.e1b.coverage import load_turns  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import planner as PL  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

RERANKERS = [  # (label, model id, revision, trust_remote_code, size, license)
    ("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", False, "568M", "Apache-2.0"),
    ("mmarco-mMiniLMv2-L12-H384", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "1427fd652930e4ba29e8149678df786c240d8825", False, "≈118M", "Apache-2.0"),
    ("mMiniLM-L6-v2-mmarco-v2", "unicamp-dl/mMiniLM-L6-v2-mmarco-v2", "8ed6820748716827e99e8f39505eaa121169c1a1", False, "≈107M", "MIT"),
]  # fmt: skip


def nvidia():
    import os

    from ctxgraph.e1b.backends import NvidiaChat

    for ln in (ROOT / ".env").read_text("utf-8").splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return NvidiaChat(R3.NV_MODEL, str(ROOT / ".cache" / "demo" / "answers_cache.jsonl"), wall=300, tries=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-rerankers", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    from ctxgraph.e1b.backends import E5Encoder
    from ctxgraph.e1b.selection import Reranker

    enc, count = E5Encoder(), P.counter()
    spec, ft, text = load_turns(ROOT, P.PID)
    g = Graph(Store(), enc, config=Config())
    for t in load_project(ROOT, P.PID)[1]:
        g.add_episode(t["text"], t["role"], "s1", eid=t["id"])
    ev = {k: set(LO.frags_for(g, s)) for k, s in LO.evidence(spec, ft, text).items()}

    def facts(units) -> int:
        return len({k for k, fs in ev.items() if any(fs & u for u in units)})

    def tally(tr) -> tuple[int, int, int]:
        mu = [set(x["fragments"]) for x in tr["final_lines"]]
        pu = [{it["fragment"]} for it in tr["periphery_items"]]
        return facts(mu), facts(pu), facts(mu + pu)

    bge = Reranker(revision=RERANKERS[0][2])
    bge.persist(ROOT / ".cache" / "v2_rerank.jsonl")
    L = ["# Отладка v2, раунд 5: ответ без памяти, планировщик шагов, лёгкие переранжировщики (без выводов о качестве)\n",
         f"Вопрос: «{P.Q}». Шаг (для A): «{P.STEP}».\n"]  # fmt: skip
    llm = None if a.skip_llm else nvidia()

    # A. answers
    tr = SL.sleeves_trace(g, P.STEP, P.Q, count, 200, anchor_cos=0.76, scorer=bge.score)
    mem = "\n\n".join(x for x in (tr["final"], tr["periphery"]) if x)
    L += ["## A. Память, которую получает модель (гибрид, после фильтра фрагментов)\n", "```text", mem, "```\n"]
    if llm:
        w, wo = R3.answer(mem), R3.answer("")
        L += ["| | с памятью | без памяти |", "|---|---|---|",
              f"| время, finish, рассуждения | {w['s']} с, {w['finish']}, {w['reasoning']} ток. | {wo['s']} с, {wo['finish']}, {wo['reasoning']} ток. |\n",
              "### Ответ с памятью\n", w["text"] or "(пусто)", "\n### Ответ без памяти\n", wo["text"] or "(пусто)", ""]  # fmt: skip

    # B. planner
    if llm:
        t0 = time.time()
        p = PL.plan_and_recall(g, llm, P.Q, count, scorer=bge.score, system=R3.SYSTEM)
        fm, fp, ftot = tally(p.trace)
        L += ["## B. Планировщик шагов (IRCoT + Self-Ask): модель пишет шаги сама\n",
              f"Время (вызов модели + поиск): {time.time() - t0:.1f} с.\n", "**Шаги модели:**\n"]  # fmt: skip
        L += [f"{i}. {s}" for i, s in enumerate(p.steps, 1)] or ["(не разобраны)", "```text", p.raw_plan[:800], "```"]
        L += ["\n**Рукава из шагов:** " + "; ".join(f"«{s['phrase']}»" for s in p.trace["sleeves"]),
              "\n**Пересечения:** " + ("; ".join(f"{R3.label(g, n)[:60]} ← {' + '.join(v)}" for n, v in p.trace["intersections"].items()) or "нет"),
              f"\n**Нужных фактов из 7:** основной блок {fm}, периферия {fp}, всего {ftot}. "
              f"(Для сравнения, шаг Егора: всего {tally(tr)[2]}.)\n",
              "```text", p.memory_for_model, "```\n"]  # fmt: skip

    # C. rerankers
    if not a.skip_rerankers:
        L += ["## C. Лёгкие переранжировщики на CPU (шаг Егора, режим «заполнение бюджета»)\n",
              "Время — переранжирование кандидатов основного блока (30 предложений) + весь поиск с рукавами; первый запрос "
              "прогревочный и в замер не входит.\n",
              "| модель | размер | лицензия | ревизия | переранжирование, с | весь поиск, с | фактов: основной блок | всего с периферией |",
              "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        api = HfApi()
        for lab, mid, rev, trc, size, lic in RERANKERS:
            rev = rev or api.model_info(mid).sha
            rr = Reranker(mid, revision=rev, trust_remote_code=trc)
            SL.sleeves_trace(g, "прогрев модели", P.Q, count, 200, scorer=rr.score, mode="fill")  # warm-up
            rr.cache.clear()
            cand_texts = [r[1] for r in g.s.q("SELECT id, text FROM fragments WHERE kind='sentence' LIMIT 30")]
            t0 = time.time()
            rr.score(P.Q + " " + P.STEP, cand_texts)
            t_rr = time.time() - t0
            rr.cache.clear()
            t0 = time.time()
            trm = SL.sleeves_trace(g, P.STEP, P.Q, count, 200, anchor_cos=0.76, scorer=rr.score, mode="fill")
            t_all = time.time() - t0
            fm, _, ftot = tally(trm)
            L.append(f"| {lab} | {size} | {lic} | {rev[:10]} | {t_rr:.2f} | {t_all:.2f} | {fm} | {ftot} |")
        L.append("\nБез переранжирования (порядок PPR + RRF): " + ", ".join(
            f"{k} {v}" for k, v in zip(("основной блок", "всего"), (lambda t: (t[0], t[2]))(tally(
                SL.sleeves_trace(g, P.STEP, P.Q, count, 200, anchor_cos=0.76))), strict=True)) + ".")  # fmt: skip
    (ROOT / "reports" / "v2_debug_round5.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
