"""bge-reranker-v2-m3 fp32 vs int8 (ONNX) × top-20 / top-40 candidates on the post-quantum example (debug).

    python scripts/v2_rerank_int8.py   # -> reports/v2_rerank_int8.md

Time per question = the whole search with sleeves (cold reranker cache), CPU. "X of 7" for the hybrid
(main + periphery) and for the main block alone. Debug only — the final choice is made on a benchmark.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_debug_loss as LO  # noqa: E402
import v2_debug_pq as P  # noqa: E402
import v2_onnx_reranker as OX  # noqa: E402

from ctxgraph.e1b.coverage import load_turns  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402


def main() -> None:
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

    fp32 = Reranker(revision=OX.REV)
    int8 = OX.OnnxReranker()
    models = {"bge fp32 (sentence-transformers)": fp32, "bge int8 (ONNX, dynamic)": int8}
    for m in models.values():  # warm-up
        m.score("прогрев", ["один два три"])
    L = ["# bge-reranker-v2-m3: fp32 vs int8 (ONNX) и число кандидатов (отладка, CPU, один пример)\n",
         f"Вопрос: «{P.Q}». Шаг: «{P.STEP}». Режим — гибрид (порог `Config.min_score` = {g.cfg.min_score}).\n",
         "| модель | кандидатов | время на вопрос, с (весь поиск с рукавами) | фактов: основной блок | всего с периферией |",
         "|---|---|---|---|---|"]  # fmt: skip
    for name, m in models.items():
        for k in (20, 40):
            m.cache.clear()
            t0 = time.time()
            mgs = SL.mini_graphs(g, P.STEP)
            final, info = SL.final_block(g, P.Q, mgs, count, scorer=m.score, candidates=k, min_score=g.cfg.min_score,
                                         score_query=f"{P.Q} {P.STEP}", split_low=True)  # fmt: skip
            dt = time.time() - t0
            main_u = [set(x["fragments"]) for x in info["lines"]]
            low_u = [set(x["fragments"]) for x in info["low_lines"]]
            L.append(f"| {name} | {k} | {dt:.2f} | {facts(main_u)} | {facts(main_u + low_u)} |")
    # agreement of scores
    sents = [r[0] for r in g.s.q("SELECT text FROM fragments WHERE kind='sentence' LIMIT 40")]
    a = np.array(fp32.score(f"{P.Q} {P.STEP}", sents))
    b = np.array(int8.score(f"{P.Q} {P.STEP}", sents))
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    rho = float(np.corrcoef(ra, rb)[0, 1])
    L.append(f"\nСогласие оценок int8 и fp32 на 40 предложениях: ранговая корреляция Спирмена {rho:.3f}; "
             f"макс. абсолютная разница {float(np.abs(a - b).max()):.4f}.")  # fmt: skip
    L.append("\nДля сравнения (раунд 5, режим «заполнение бюджета», 30 кандидатов): bge fp32 — 11,2 с, всего 5 из 7; "
             "mmarco-mMiniLMv2-L12 — 1,0 с, 3 из 7; mMiniLM-L6 — 0,6 с, 2 из 7; без переранжирования — 4 из 7.")  # fmt: skip
    (ROOT / "reports" / "v2_rerank_int8.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
