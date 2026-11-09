"""E1d этап A2 (разведка до пре-регистрации V9, без API): V9-q «главная линия + периферия».

    python scripts/e1d_stage_a2.py   # → reports/e1d/stage_a2.md, reports/e1d/stage_a2.json

Поправки Егора после этапа A (параметры не перебираются; порог якоря 0.76 и запрос «clause» —
из этапа A):
1. Пул = пул V6 как есть + top-8 на каждый рукав; в память 4 полных фрагмента: 3 лучших V6 +
   1 место лучшему кандидату периферии вне пула V6.
2. Бонус пересечения делится на число сущностей реплики (`bonus_norm="ents"`); сравнение с
   прежним бонусом и доля опорных реплик среди пересечений и прочих кандидатов.
3. Потолок рукавов из вопроса: опорный факт содержит хоть одну сущность / слово рукавов вопроса.
4. Верхняя граница пула: объединение всех рукавов с k = 20 на рукав, без отбора и якоря.

Сначала 20 пробных вопросов (старые E1b, те же, что в этапе A), затем 200 основных.
Архив на момент вопроса — как в этапе A (`scripts/e1d_stage_a.py`).
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

from ctxgraph.e1b import coverage as C  # noqa: E402
from ctxgraph.e1b import memory as M  # noqa: E402
from ctxgraph.e1b import v9  # noqa: E402

ANCHOR, MODE = 0.76, "clause"
VARS = ["V6", "V9-q (этап A)", "V9-q2, бонус как был", "V9-q2, бонус / сущности"]


def probe_items(items):
    pool = [(pid, it) for pid in A.OLD for it in items[pid] if it["q"].get("anchors")]
    return random.Random(0).sample(pool, 20)


def evaluate(pid, it, count, turns_cache):
    if pid not in turns_cache:
        turns_cache[pid] = C.load_turns(ROOT, pid)
    _, ft, text = turns_cache[pid]
    q, arc = it["q"], it["arc"]
    anchors = q["anchors"]

    def cv(mem=(), ids=()):
        return C.anchor_coverage(anchors, ft, text, mem, ids)

    mem = {
        "V6": M.v6_hybrid(arc, q["text"], count, A.BUDGET),
        "V9-q (этап A)": v9.v9q(arc, q["text"], count, A.BUDGET, anchor_cos=ANCHOR, query_mode=MODE)[0],
    }
    logs = {}
    for name, norm in (("V9-q2, бонус как был", None), ("V9-q2, бонус / сущности", "ents")):
        mem[name], logs[name] = v9.v9q_main(arc, q["text"], count, A.BUDGET, anchor_cos=ANCHOR,
                                            query_mode=MODE, bonus_norm=norm)  # fmt: skip
    row = {"pid": pid, "q": q["id"], "type": q["type"], "need": len(dict.fromkeys(anchors))}
    for name, m in mem.items():
        row[name] = cv(m)
    lg = logs["V9-q2, бонус / сущности"]
    row["pool_v6"] = cv(ids=lg["pool_main"])
    row["pool_q2"] = cv(ids=lg["pool"])
    # 4. верхняя граница: все рукава, k = 20 на рукав, без якоря
    c20 = v9.candidates(arc, q["text"], query_mode=MODE, k_per_sleeve=20)
    row["pool_k20"] = cv(ids=c20["ranked"])
    row["pool_k20_v6"] = cv(ids=list(lg["pool_main"]) + c20["ranked"])
    row["n_pool_k20"] = len(c20["ranked"])
    # 2. пересечения: опорные реплики среди пересечений и прочих кандидатов периферии
    anchor_turns = {t for f in anchors for t in ft.get(f, [])}
    c = v9.candidates(arc, q["text"], anchor_cos=ANCHOR, query_mode=MODE, k_per_sleeve=8)
    inter = set(c["intersections"])
    row["inter"] = [(t in anchor_turns, len(arc.ents[t])) for t in c["ranked"] if t in inter]
    row["other"] = [(t in anchor_turns, len(arc.ents[t])) for t in c["ranked"] if t not in inter]
    for name in ("V9-q2, бонус как был", "V9-q2, бонус / сущности"):
        p = logs[name]["picked_periphery"]
        row[f"pick:{name}"] = [t in anchor_turns for t in p]
        row[f"top3:{name}"] = [t in anchor_turns for t in logs[name]["periphery"][:3]]
    # 3. потолок рукавов из вопроса
    keys = [(s.id, s.kind) for s in v9.text_sleeves(arc, q["text"], query_mode=MODE)]
    in_arc, reach = 0, 0
    for f in dict.fromkeys(anchors):
        ts = [t for t in ft.get(f, []) if t in arc.by_id]
        in_arc += bool(ts)
        reach += any(
            (kind == "entity" and key in arc.ents[t]) or (kind != "entity" and key in arc.docs[arc.order[t]])
            for t in ts
            for key, kind in keys
        )
    row["facts_in_archive"], row["facts_reach_sleeves"] = in_arc, reach
    return row


def pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "—"


def report(rows, title, boot=True) -> list[str]:
    types = [(ty, ru) for ty, ru in A.TYPES if any(r["type"] == ty for r in rows)] + [("all", "все")]
    sel = {ty: [r for r in rows if ty == "all" or r["type"] == ty] for ty, _ in types}
    L = [f"## {title}\n", "### Полнота: опорные факты в памяти полным фрагментом (по фактам)\n",
         "| тип (вопросов) | " + " | ".join(VARS) + " | V9-q2 (бонус/сущн.) − V6, п.п. [95 %] |",
         "|---|" + "---|" * (len(VARS) + 1)]  # fmt: skip
    for ty, ru in types:
        rs = sel[ty]
        need = sum(r["need"] for r in rs)
        d = [r[VARS[3]]["share"] - r["V6"]["share"] for r in rs]
        ci = A.boot(d) if boot else None
        L.append(f"| {ru} ({len(rs)}) | " + " | ".join(pct(sum(r[v]["n"] for r in rs), need) for v in VARS)
                 + f" | {100 * np.mean(d):+.1f}" + (f" [{100 * ci[0]:+.1f}; {100 * ci[1]:+.1f}]" if ci else "") + " |")  # fmt: skip
    L.append(
        "\nV9-q2 (бонус/сущн.) против V6 по вопросам — лучше / хуже / так же: "
        + ", ".join(
            f"{ru} {sum(r[VARS[3]]['n'] > r['V6']['n'] for r in sel[ty])} / "
            f"{sum(r[VARS[3]]['n'] < r['V6']['n'] for r in sel[ty])} / {sum(r[VARS[3]]['n'] == r['V6']['n'] for r in sel[ty])}"
            for ty, ru in types
        )
        + ". «Все опорные в памяти»: "
        + f"V6 {sum(r['V6']['all'] for r in sel['all'])}"
        + f", V9-q2 {sum(r[VARS[3]]['all'] for r in sel['all'])}.\n"
    )  # fmt: skip
    L += ["### Пул кандидатов и верхняя граница (п. 4)\n",
          "| тип | пул V6 | пул V9-q2 (V6 + рукава по 8) | все рукава, k = 20 | рукава k = 20 ∪ пул V6 |",
          "|---|---|---|---|---|"]  # fmt: skip
    for ty, ru in types:
        rs = sel[ty]
        need = sum(r["need"] for r in rs)
        L.append(f"| {ru} | " + " | ".join(pct(sum(r[k]["n"] for r in rs), need)
                 for k in ("pool_v6", "pool_q2", "pool_k20", "pool_k20_v6")) + " |")  # fmt: skip
    L.append(f"\nРазмер пула «все рукава, k = 20» в среднем {np.mean([r['n_pool_k20'] for r in rows]):.0f} реплик.\n")
    L += ["### Потолок рукавов из вопроса (п. 3)\n",
          "| тип | опорных фактов | из них в архиве (не только в окне) | содержат сущность/слово рукава вопроса |",
          "|---|---|---|---|"]  # fmt: skip
    for ty, ru in types:
        rs = sel[ty]
        need = sum(r["need"] for r in rs)
        a, b = sum(r["facts_in_archive"] for r in rs), sum(r["facts_reach_sleeves"] for r in rs)
        L.append(f"| {ru} | {need} | {a} ({pct(a, need)}) | {b} ({pct(b, need)}) |")
    inter = [x for r in rows for x in r["inter"]]
    other = [x for r in rows for x in r["other"]]
    L += ["\n### Пересечения против «узлов» (п. 2), кандидаты периферии (рукава по 8, якорь 0.76)\n",
          f"- Пересечений {len(inter)}, опорных среди них {pct(sum(a for a, _ in inter), len(inter))}; прочих кандидатов "
          f"{len(other)}, опорных {pct(sum(a for a, _ in other), len(other))}. Сущностей в реплике в среднем: "
          f"пересечения {np.mean([n for _, n in inter] or [0]):.1f}, прочие {np.mean([n for _, n in other] or [0]):.1f}.",
          "- Множество пересечений поправкой не меняется (меняется порядок). Что меняется — выбор периферии:"]  # fmt: skip
    for name in VARS[2:]:
        pk = [x for r in rows for x in r[f"pick:{name}"]]
        t3 = [x for r in rows for x in r[f"top3:{name}"]]
        L.append(f"  - {name}: место периферии занято в {len(pk)} вопросах, опорная реплика — {sum(pk)} ({pct(sum(pk), len(pk))}); "
                 f"в top-3 периферии опорных {pct(sum(t3), len(t3))}.")  # fmt: skip
    return L + [""]


def render(probe, main_rows, n_match) -> str:
    L = ["# E1d этап A2 — V9-q «главная линия + периферия» (разведка до пре-регистрации V9, без API)\n",
         f"Параметры не перебирались: порог якоря {ANCHOR}, запрос рукава «{MODE}», рукавов ≤ {v9.MAX_SLEEVES}, "
         f"top-8 на рукав, 4 полных фрагмента (3 V6 + 1 периферия), бюджет {A.BUDGET}, бонус {v9.BONUS:.4f}. "
         f"Архив восстановлен как в этапе A (V6 совпал дословно у {n_match} из 200 основных).\n"]  # fmt: skip
    L += report(probe, "Пробные 20 вопросов (старые E1b, seed 0) — пробные", boot=False)
    L += report(main_rows, "Основные 200 вопросов")
    return "\n".join(L) + "\n"


def main() -> None:
    if "--render" in sys.argv:  # пересобрать отчёт из сохранённого json без пересчёта
        d = json.loads((A.OUT / "stage_a2.json").read_text("utf-8"))
        (A.OUT / "stage_a2.md").write_text(render(d["probe"], d["main"], d.get("match", 143)), "utf-8")
        return
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b.backends import E5Encoder

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    enc = A.CachedEncoder(E5Encoder())
    known = json.loads(A.PREFIX.read_text("utf-8")) if A.PREFIX.exists() else {}
    items = {pid: A.reconstruct(pid, enc, count, known) for pid in A.NEW + A.OLD}
    cache = {}
    probe = [evaluate(pid, it, count, cache) for pid, it in probe_items(items)]
    main_rows = [evaluate(pid, it, count, cache) for pid in A.NEW for it in items[pid]]
    match = Counter(it["match"] for pid in A.NEW for it in items[pid])
    A.OUT.mkdir(parents=True, exist_ok=True)
    data = {"probe": probe, "main": main_rows, "match": match[True]}
    (A.OUT / "stage_a2.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    text = render(probe, main_rows, match[True])
    (A.OUT / "stage_a2.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
