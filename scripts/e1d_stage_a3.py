"""E1d этап A3 (разведка до пре-регистрации V9, без API): проверки Егора перед этапом B.

    python scripts/e1d_stage_a3.py   # → reports/e1d/stage_a3.md, reports/e1d/stage_a3.json

1. Расхождение V6 на пробных (этап A 39 %, A2 35 %) — разбор в отчёте (те же вопросы, разные
   формулы: среднее по вопросам против доли по фактам).
2. Контроль размера пула: поиск по всему вопросу (пул V6 с увеличенным k, ранжирование V6),
   обрезанный до того же размера, что пул рукавов, — для каждого вопроса отдельно.
3. Порядок периферии: ранг опорных реплик среди кандидатов периферии при текущей оценке и при
   альтернативах (только пересечения по числу рукавов; без якоря; бонус / сущности; без бонуса;
   случайный); проверка, что бонус пересечения меняет порядок.
4. V9-q3: блок V6 как есть + периферия только в остаток бюджета (выдержки до 40 токенов).

Архив — как в этапе A; параметры прежние (якорь 0.76, «clause», рукавов ≤ 4).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import e1d_stage_a as A  # noqa: E402
import e1d_stage_a2 as A2  # noqa: E402

from ctxgraph.e1b import coverage as C  # noqa: E402
from ctxgraph.e1b import memory as M  # noqa: E402
from ctxgraph.e1b import v9  # noqa: E402

ANCHOR, MODE, ITEM = 0.76, "clause", 40
ORDERS = ["текущий (бонус, якорь 0.76)", "бонус / сущности", "без бонуса", "(а) только пересечения, по числу рукавов",
          "(б) без порога якоря", "(в) случайный"]  # fmt: skip


def periph_orders(arc, q, main) -> dict[str, list[str]]:
    def run(**kw):
        return v9.candidates(arc, q, query_mode=MODE, k_per_sleeve=8, **kw)

    cur = run(anchor_cos=ANCHOR)
    out = {
        ORDERS[0]: cur["ranked"],
        ORDERS[1]: run(anchor_cos=ANCHOR, bonus_norm="ents")["ranked"],
        ORDERS[2]: run(anchor_cos=ANCHOR, bonus=0.0)["ranked"],
        ORDERS[4]: run()["ranked"],
    }
    pos = {t: i for i, t in enumerate(cur["ranked"])}
    inter = [t for t in cur["ranked"] if len(cur["reached"][t]) >= 2]
    out[ORDERS[3]] = sorted(inter, key=lambda t: (-len(cur["reached"][t]), pos[t]))
    out = {k: [t for t in v if t not in main] for k, v in out.items()}
    sh = list(out[ORDERS[0]])
    random.Random(f"{q}").shuffle(sh)
    out[ORDERS[5]] = sh
    # величины: лучший RRF и бонус у кандидатов периферии (текущая оценка)
    mag = [(cur["best"][t], v9.BONUS * (len(cur["reached"][t]) - 1), len(arc.ents[t]))
           for t in out[ORDERS[0]]]  # fmt: skip
    return out, mag


def evaluate(pid, it, count, cache):
    if pid not in cache:
        cache[pid] = C.load_turns(ROOT, pid)
    _, ft, text = cache[pid]
    q, arc, anchors = it["q"]["text"], it["arc"], it["q"]["anchors"]

    def cv(mem=(), ids=()):
        return C.anchor_coverage(anchors, ft, text, mem, ids)["n"]

    row = {"pid": pid, "q": it["q"]["id"], "type": it["q"]["type"], "need": len(dict.fromkeys(anchors))}
    main, _ = M.v6_candidates(arc, q)
    # 2. размер пула
    s20 = v9.candidates(arc, q, query_mode=MODE, k_per_sleeve=20)["ranked"]
    _, lg = v9.v9q_main(arc, q, count, A.BUDGET, anchor_cos=ANCHOR, query_mode=MODE)
    p8 = lg["pool"]
    for name, pool in (("s20", s20), ("p8", p8)):
        n = len(pool)
        wide = M.v6_candidates(arc, q, k=n)[0][:n]
        row[f"{name}_size"], row[f"{name}_wide_size"] = n, len(wide)
        row[f"{name}_cov"], row[f"{name}_wide_cov"] = cv(ids=pool), cv(ids=wide)
    row["v6pool_size"], row["v6pool_cov"] = len(main), cv(ids=main)
    # 3. порядок периферии
    anchor_turns = {t for f in anchors for t in ft.get(f, [])}
    orders, mag = periph_orders(arc, q, main)
    row["orders"] = {k: [t in anchor_turns for t in v] for k, v in orders.items()}
    row["top1_ids"] = {k: (v[0] if v else None) for k, v in orders.items()}
    row["mag"] = mag
    # 4. V9-q3
    mem6 = M.v6_hybrid(arc, q, count, A.BUDGET)
    mem3, l3 = v9.v9q_rest(arc, q, count, A.BUDGET, anchor_cos=ANCHOR, query_mode=MODE, item_tokens=ITEM)
    whole = [t for t in l3["periphery_shown"] if count(arc.by_id[t].text) <= ITEM]
    row["v6"], row["v9q3"] = cv(mem6), cv(mem3)
    row["v9q3_strict"] = cv(mem6, whole)
    row["v6_tok"], row["v9q3_tok"] = count(mem6) if mem6 else 0, count(mem3) if mem3 else 0
    row["n_shown"] = len(l3["periphery_shown"])
    row["shown_anchor"] = sum(t in anchor_turns for t in l3["periphery_shown"])
    return row


def pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "—"


def report(rows, title, boot) -> list[str]:
    types = [(ty, ru) for ty, ru in A.TYPES if any(r["type"] == ty for r in rows)] + [("all", "все")]
    sel = {ty: [r for r in rows if ty == "all" or r["type"] == ty] for ty, _ in types}
    L = [f"## {title}\n", "### 2. Пул при равном размере — доля опорных фактов в пуле (до отбора)\n",
         "| тип | пул V6 | рукава k = 20 | весь вопрос, тот же размер | разность | V6 + рукава по 8 | весь вопрос, тот же размер | разность |",
         "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for ty, ru in types:
        rs = sel[ty]
        need = sum(r["need"] for r in rs)
        g = {k: sum(r[k] for r in rs) for k in ("v6pool_cov", "s20_cov", "s20_wide_cov", "p8_cov", "p8_wide_cov")}
        L.append(f"| {ru} | {pct(g['v6pool_cov'], need)} | {pct(g['s20_cov'], need)} | {pct(g['s20_wide_cov'], need)} | "
                 f"{100 * (g['s20_cov'] - g['s20_wide_cov']) / need:+.1f} п.п. | {pct(g['p8_cov'], need)} | "
                 f"{pct(g['p8_wide_cov'], need)} | {100 * (g['p8_cov'] - g['p8_wide_cov']) / need:+.1f} п.п. |")  # fmt: skip
    sz = {k: np.mean([r[k] for r in rows]) for k in ("v6pool_size", "s20_size", "s20_wide_size", "p8_size", "p8_wide_size")}
    L.append(f"\nРазмер пула в среднем: V6 {sz['v6pool_size']:.1f}; рукава k = 20 {sz['s20_size']:.1f} (весь вопрос "
             f"{sz['s20_wide_size']:.1f}); V6 + рукава по 8 {sz['p8_size']:.1f} (весь вопрос {sz['p8_wide_size']:.1f}). "
             "«Весь вопрос» — пул V6 с k, равным размеру пула рукавов, в порядке V6, обрезан до того же размера.\n")  # fmt: skip
    # 3
    has = [r for r in rows if any(r["orders"][ORDERS[0]])]
    L += ["### 3. Порядок кандидатов периферии (вне пула V6)\n",
          f"Вопросов, где среди кандидатов периферии (якорь 0.76) есть опорная реплика: {len(has)} из {len(rows)}. "
          f"Кандидатов периферии в среднем {np.mean([len(r['orders'][ORDERS[0]]) for r in rows]):.1f}, опорных среди них "
          f"{pct(sum(sum(r['orders'][ORDERS[0]]) for r in rows), sum(len(r['orders'][ORDERS[0]]) for r in rows))}.\n",
          "| порядок | опорная на 1-м месте (вопросов из всех) | в top-3 | средний отн. ранг первой опорной (0 — первая, 1 — последняя) |",
          "|---|---|---|---|"]  # fmt: skip
    for o in ORDERS:
        top1 = sum(bool(r["orders"][o]) and r["orders"][o][0] for r in rows)
        top3 = sum(any(r["orders"][o][:3]) for r in rows)
        ranks = [r["orders"][o].index(True) / max(1, len(r["orders"][o]) - 1) for r in rows if any(r["orders"][o])]
        L.append(f"| {o} | {top1} | {top3} | {np.mean(ranks) if ranks else float('nan'):.2f} (n = {len(ranks)}) |")
    exp = sum(sum(r["orders"][ORDERS[0]]) / len(r["orders"][ORDERS[0]]) for r in rows if r["orders"][ORDERS[0]])
    L.append(f"\nОжидание случайного порядка для «1-го места»: {exp:.1f} вопросов.")
    diff = Counter()
    for r in rows:
        t = r["top1_ids"]
        diff["бонус против бонус/сущности"] += t[ORDERS[0]] != t[ORDERS[1]]
        diff["бонус против без бонуса"] += t[ORDERS[0]] != t[ORDERS[2]]
    mag = [m for r in rows for m in r["mag"]]
    b = np.array([x[1] for x in mag])
    rrf = np.array([x[0] for x in mag])
    ents = np.array([x[2] for x in mag])
    L += [f"\n**Проверка бонуса.** Первое место периферии отличается: бонус против «бонус / сущности» — в {diff['бонус против бонус/сущности']} "
          f"вопросах из {len(rows)}, бонус против «без бонуса» — в {diff['бонус против без бонуса']}. "
          f"Среди кандидатов периферии: лучший RRF рукава — медиана {np.median(rrf):.4f}; бонус — медиана {np.median(b):.4f}, "
          f"доля с бонусом > 0 {pct(int((b > 0).sum()), len(b))}; бонус / сущности — медиана {np.median(b / np.maximum(ents, 1)):.4f} "
          f"(сущностей в реплике медиана {np.median(ents):.0f}).\n"]  # fmt: skip
    # 4
    L += ["### 4. V9-q3: блок V6 как есть + периферия в остаток бюджета\n",
          "| тип | V6 | V9-q3 | разность, п.п. | V9-q3 строго (периферия засчитана, только если выдержка — вся реплика) |",
          "|---|---|---|---|---|"]  # fmt: skip
    for ty, ru in types:
        rs = sel[ty]
        need = sum(r["need"] for r in rs)
        d = [(r["v9q3"] - r["v6"]) / r["need"] for r in rs if r["need"]]
        ci = A.boot(d) if boot else None
        L.append(f"| {ru} ({len(rs)}) | {pct(sum(r['v6'] for r in rs), need)} | {pct(sum(r['v9q3'] for r in rs), need)} | "
                 f"{100 * np.mean(d):+.1f}" + (f" [{100 * ci[0]:+.1f}; {100 * ci[1]:+.1f}]" if ci else "")
                 + f" | {pct(sum(r['v9q3_strict'] for r in rs), need)} |")  # fmt: skip
    L.append(f"\nТокенов: V6 {np.mean([r['v6_tok'] for r in rows]):.0f}, V9-q3 {np.mean([r['v9q3_tok'] for r in rows]):.0f} "
             f"(бюджет {A.BUDGET}); выдержек периферии в среднем {np.mean([r['n_shown'] for r in rows]):.1f} "
             f"(до {ITEM} токенов), опорных среди показанных {pct(sum(r['shown_anchor'] for r in rows), sum(r['n_shown'] for r in rows))}. "
             "Метрика полноты узнаёт реплику по её началу, а выдержка — начало реплики; факт из конца длинной реплики "
             "мог в выдержку не попасть — поэтому «строго».\n")  # fmt: skip
    return L


def render(d) -> str:
    L = ["# E1d этап A3 — проверки перед этапом B (разведка до пре-регистрации V9, без API)\n",
         "## 1. Расхождение V6 на пробных: 39 % (этап A) против 35 % (A2)\n",
         f"Те же 20 вопросов, те же блоки памяти. Разные формулы: в этапе A — средняя доля по вопросам "
         f"({100 * d['probe_v6_q']:.0f} %), в A2 — доля по фактам ({100 * d['probe_v6_f']:.0f} %). "
         f"Пересчёт A2 по формуле этапа A даёт V6 {100 * d['probe_v6_q']:.0f} %, V9-q (этап A) {100 * d['probe_v9a_q']:.0f} % — "
         "как в этапе A. Недетерминизма и изменения V6 нет. Ниже везде — доля по фактам.\n"]  # fmt: skip
    L += report(d["probe"], "Пробные 20 вопросов (старые E1b, seed 0) — пробные", boot=False)
    L += report(d["main"], "Основные 200 вопросов", boot=True)
    return "\n".join(L) + "\n"


def main() -> None:
    if "--render" in sys.argv:
        d = json.loads((A.OUT / "stage_a3.json").read_text("utf-8"))
        (A.OUT / "stage_a3.md").write_text(render(d), "utf-8")
        return
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b.backends import E5Encoder

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    enc = A.CachedEncoder(E5Encoder())
    known = json.loads(A.PREFIX.read_text("utf-8"))
    items = {pid: A.reconstruct(pid, enc, count, known) for pid in A.NEW + A.OLD}
    a2 = json.loads((A.OUT / "stage_a2.json").read_text("utf-8"))["probe"]
    cache = {}
    d = {
        "probe_v6_q": float(np.mean([r["V6"]["share"] for r in a2])),
        "probe_v6_f": sum(r["V6"]["n"] for r in a2) / sum(r["need"] for r in a2),
        "probe_v9a_q": float(np.mean([r["V9-q (этап A)"]["share"] for r in a2])),
        "probe": [evaluate(pid, it, count, cache) for pid, it in A2.probe_items(items)],
        "main": [evaluate(pid, it, count, cache) for pid in A.NEW for it in items[pid]],
    }
    (A.OUT / "stage_a3.json").write_text(json.dumps(d, ensure_ascii=False), "utf-8")
    text = render(d)
    (A.OUT / "stage_a3.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
