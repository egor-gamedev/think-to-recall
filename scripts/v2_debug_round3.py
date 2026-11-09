"""Debug round 3 (<5 min + 2 free NVIDIA calls): post-quantum example after Egor's fixes 2–5.

    python scripts/v2_debug_round3.py [--min-score 0.05]   # -> reports/v2_debug_round3.md

1. Answer with memory vs without (same model and settings, one call each).
2. Periphery must not repeat the main block (before/after).
3. Assistant turns ×0.6: roles in the block before/after.
4. Intersections: plain rule vs strict rule (threshold in each mini-graph, no hubs, single-sentence
   concepts count as their sentence).
5. Cross-encoder score of every line of the main block; cutoff instead of filling the budget.
6. "X of 7" relevant facts after the fixes.
Debug only — no quality claims.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_debug_loss as LO  # noqa: E402
import v2_debug_pq as P  # noqa: E402

from ctxgraph.e1b.coverage import load_turns  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

NV_MODEL = "mistralai/mistral-large-4-1t-a52b"
SYSTEM = "You help a user with their software project. Answer in the user's language, briefly."
REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def answer(memory: str) -> dict:
    from ctxgraph.e1b.backends import NvidiaChat

    for ln in (ROOT / ".env").read_text("utf-8").splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    llm = NvidiaChat(NV_MODEL, str(ROOT / ".cache" / "demo" / "answers_cache.jsonl"), wall=300, tries=2)
    user = (memory + "\n\n---\n" if memory else "") + f"{P.Q}\n\nМои мысли: {P.STEP}"
    t0 = time.time()
    r = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_tokens=4000)
    u = r.get("usage") or {}
    return {"text": r.get("content", ""), "s": round(time.time() - t0, 1), "finish": r.get("finish_reason"),
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}  # fmt: skip


def label(g, n):
    r = g.s.q("SELECT text FROM fragments WHERE id=?", (n,)) or g.s.q("SELECT label FROM concepts WHERE id=?", (n,))
    return (r[0][0] if r else n)[:80]


def facts_hit(g, picked, ev) -> dict:
    return {k: bool(set(LO.frags_for(g, sents)) & picked) for k, sents in ev.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=1e-3)
    ap.add_argument("--no-answers", action="store_true")
    a = ap.parse_args()
    from ctxgraph.e1b.backends import E5Encoder
    from ctxgraph.e1b.selection import Reranker

    enc, count = E5Encoder(), P.counter()
    spec, ft, text = load_turns(ROOT, P.PID)
    ev = LO.evidence(spec, ft, text)
    g = Graph(Store(), enc, config=Config())
    for t in load_project(ROOT, P.PID)[1]:
        g.add_episode(t["text"], t["role"], "s1", eid=t["id"])
    rr = Reranker(revision=REV)
    rr.persist(ROOT / ".cache" / "v2_rerank.jsonl")
    mgs = SL.mini_graphs(g, P.STEP)
    L = ["# Отладка v2, раунд 3: пост-квантовый пример после правок 2–5 (≤ 5 мин, без выводов о качестве)\n",
         f"Вопрос: «{P.Q}». Шаг: «{P.STEP}».\n"]  # fmt: skip

    # 4. intersections
    plain, strict = SL.intersections(mgs), SL.intersections(mgs, g)
    L += ["## 4. Пересечения рукавов\n", f"**До** (вес ≥ 0,05 хотя бы в двух мини-графах) — {len(plain)}:\n"]
    L += [f"- {label(g, n)} ← {' + '.join(v)}" for n, v in plain.items()] or ["- нет"]
    L += [f"\n**После** (вес ≥ 0,2 в каждом; без хабов; понятие из одного предложения = это предложение) — {len(strict)}:\n"]
    L += [f"- {label(g, n)} ← {' + '.join(v)}" for n, v in strict.items()] or ["- нет"]

    # 3 + 5. roles and cross-encoder scores
    def block(w_assistant, scorer=None, min_score=None, score_query=f"{P.Q} {P.STEP}"):
        g.cfg.w_assistant = w_assistant
        return SL.final_block(g, P.Q, mgs, count, scorer=scorer, min_score=min_score, score_query=score_query)

    b_old, i_old = block(1.0)
    b_new, i_new = block(0.6)
    roles = lambda info: (sum(x["role"] == "user" for x in info["lines"]), sum(x["role"] == "assistant" for x in info["lines"]))  # noqa: E731
    L += ["\n## 3. Роли в основном блоке (без переранжирования)\n",
          f"Без понижения реплик модели: пользователь {roles(i_old)[0]}, ассистент {roles(i_old)[1]}. "
          f"С × 0,6: пользователь {roles(i_new)[0]}, ассистент {roles(i_new)[1]}.\n"]  # fmt: skip
    b_q, i_q = block(0.6, rr.score, None, P.Q)  # previous behaviour: cross-encoder query = question only
    L += ["## 5а. Откуда шум: переранжирование только по вопросу (как было)\n", "| оценка | роль | строка |", "|---|---|---|"]
    L += [f"| {x['score']:.2e} | {x['role']} | {x['line'][:120].replace('|', '/')} |" for x in i_q["lines"][:6]]
    L.append("\nПо одному вопросу («стоит ли добавлять…») cross-encoder почти всем даёт ~10⁻⁵ и выше всех ставит шум; "
             "запрос «вопрос + текст шага» разводит нужное (~10⁻²–10⁻¹) и шум (~10⁻⁴).\n")
    b_fill, i_fill = block(0.6, rr.score)
    b_cut, i_cut = block(0.6, rr.score, a.min_score)
    L += ["## 5. Оценка переранжирования (bge-reranker-v2-m3) для каждой строки блока — заполнение бюджета до конца\n",
          "| оценка | с × 0,6 | роль | строка |", "|---|---|---|---|"]  # fmt: skip
    for x in i_fill["lines"]:
        L.append(f"| {x['score']:.2e} | {x['adjusted']:.2e} | {x['role']} | {x['line'][:150].replace('|', '/')} |")
    noisy = [x for x in i_fill["lines"] if any(k in x["line"] for k in ("скрытие балансов", "Сегодня экран создания"))]
    L += ["\nСтроки-«шум» из прошлого блока («Про скрытие балансов…», «Сегодня экран создания кошелька»): "
          + ("; ".join(f"оценка {x['score']:.2e}" for x in noisy) if noisy else "после правок в блок не попали") + ".",
          f"\n**Порог отсечения {a.min_score}** вместо заполнения бюджета: строк {len(i_cut['lines'])} (было {len(i_fill['lines'])}), "
          f"токенов {count(b_cut) if b_cut else 0} (было {count(b_fill) if b_fill else 0}).\n",
          "```text", b_cut or "(пусто)", "```\n"]  # fmt: skip

    # 2. periphery repeats: old way (no exclusion) vs new
    old_per, old_items = SL.periphery(g, mgs, P.Q, count, 200, anchor_cos=0.76)
    shown = {f for u in i_cut["picked"] for f in u}
    new_per, new_items = SL.periphery(g, mgs, P.Q, count, 200, anchor_cos=0.76, exclude=shown)
    rep_old = sorted(shown & {x["fragment"] for x in old_items})
    rep_new = sorted(shown & {x["fragment"] for x in new_items})
    L += ["## 2. Повторы основного блока в периферии\n",
          f"До: повторов {len(rep_old)} ({'; '.join(label(g, f)[:50] for f in rep_old) or '—'}). После: {len(rep_new)}.\n",
          "```text", new_per or "(пусто)", "```\n"]  # fmt: skip

    # 6. X of 7
    rows = [("основной блок, без переранжирования (×0,6)", i_new), ("основной блок, переранжирование до конца бюджета", i_fill),
            (f"основной блок, порог {a.min_score}", i_cut)]  # fmt: skip
    hits = {name: facts_hit(g, {f for u in info["picked"] for f in u}, ev) for name, info in rows}
    per_hits = facts_hit(g, {x["fragment"] for x in new_items}, ev)
    desc = {f: spec["facts"][f]["text"] for f in P.FACTS} | dict(P.LIBS)
    L += ["## 6. Нужные факты (из 7)\n", "| факт | " + " | ".join(n for n, _ in rows) + " | периферия |", "|---|" + "---|" * (len(rows) + 1)]
    for k in desc:
        L.append(f"| {k}: {desc[k][:60]} | " + " | ".join("да" if hits[n][k] else "—" for n, _ in rows) + f" | {'да' if per_hits[k] else '—'} |")
    L.append("| **итого** | " + " | ".join(str(sum(hits[n].values())) for n, _ in rows) + f" | {sum(per_hits.values())} |")
    L.append("\nДля сравнения (раунд 2, `reports/v2_debug_loss.md`): до правок 1–2 — 3 из 7, после — 3 из 7 (без переранжирования).\n")

    # 1. answers
    if not a.no_answers:
        with_mem, no_mem = answer(b_cut), answer("")
        L += ["## 1. Ответ модели с памятью и без (Mistral Large 4, NVIDIA, те же настройки)\n",
              "В обоих запросах — вопрос и текст шага («Мои мысли: …»); различие только в блоке памяти (порог отсечения).\n",
              "| | с памятью | без памяти |", "|---|---|---|",
              f"| время, finish, рассуждения | {with_mem['s']} с, {with_mem['finish']}, {with_mem['reasoning']} ток. | "
              f"{no_mem['s']} с, {no_mem['finish']}, {no_mem['reasoning']} ток. |\n",
              "### С памятью\n", with_mem["text"] or "(пусто)", "\n### Без памяти\n", no_mem["text"] or "(пусто)"]  # fmt: skip
    (ROOT / "reports" / "v2_debug_round3.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
