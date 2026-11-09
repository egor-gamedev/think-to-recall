"""E1d этап A4 (разведка до пре-регистрации этапа B, без API).

    python scripts/e1d_stage_a4.py   # → reports/e1d/stage_a4.md, sleeve_examples.md, coverage_check.md

1. Как строятся рукава сейчас: 5 вопросов «связать» (seed 0) — рукава, правило, что нашёл каждый
   рукав (top-5 своего поиска + граф, k = 8 на рукав, без якоря), опорные реплики отмечены.
2. Рукава-фразы (`v9.phrase_sleeves`): подбор max_len ∈ {4, 6} × предложный оборот ∈ {нет, да}
   только на пробных (полнота объединённого пула рукавов по репликам); те же 5 примеров.
3. Общий слой отбора (`selection.select`): пул 30 реплик → предложения → cross-encoder → 600
   токенов. Полнота на уровне факта (`coverage.fact_coverage`). V6 против V6 + новый отбор.
   30 случаев «засчитан / не засчитан» — на ручную проверку Егора (`coverage_check.md`).
Архив — как в этапе A.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import e1d_stage_a as A  # noqa: E402
import e1d_stage_a2 as A2  # noqa: E402

from ctxgraph.e1b import coverage as C  # noqa: E402
from ctxgraph.e1b import memory as M  # noqa: E402
from ctxgraph.e1b import selection as SEL  # noqa: E402
from ctxgraph.e1b import v9  # noqa: E402

MODE = "clause"
GRID = [(4, False), (4, True), (6, False), (6, True)]
REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # BAAI/bge-reranker-v2-m3, ревизия на HF


def first_words(s: str, n: int = 15) -> str:
    w = " ".join(M.excerpt(s, lambda x: len(x.split()), 10_000).split()[:n])
    return w.replace("|", "/")


def sleeve_finds(arc, q, sleeves):
    c = v9.candidates(arc, q, sleeves=sleeves, k_per_sleeve=8)
    return c


def example_block(arc, q, spec, ft, sleeves, rule) -> list[str]:
    anchors = q["anchors"]
    tag = {t: f for f in anchors for t in ft.get(f, [])}
    c = sleeve_finds(arc, q["text"], sleeves)
    L = [f"*Правило:* {rule}\n"]
    for s in sleeves:
        found = c["per_sleeve"].get(s.id, [])
        hits = sum(t in tag for t in found)
        L.append(f"- **рукав «{s.query}»** (ключ `{s.id}`, {s.kind}) — нашёл {len(found)} реплик, опорных {hits}:")
        for t in found[:5]:
            mark = f" **← опорная {tag[t]}**" if t in tag else ""
            L.append(f"  - `{t}` [{M.ROLE_RU[arc.by_id[t].role]}] {first_words(arc.by_id[t].text)}…{mark}")
    miss = [f for f in anchors if not any(t in c["ranked"] for t in ft.get(f, []))]
    L.append(f"- не найдено ни одним рукавом: {', '.join(miss) or '—'}\n")
    return L


def examples(items) -> str:
    link = [(pid, it) for pid in A.NEW for it in items[pid] if it["q"]["type"] == "link"]
    pick = random.Random(0).sample(link, 5)
    L = ["# Рукава: 5 вопросов «связать» (seed 0) — сейчас и как фразы\n",
         "Поиск рукава: эмбеддинги + BM25 → RRF, top-8 на рукав, шаг по графу (как V6), без порога якоря. "
         "Показаны первые 5 находок рукава; «опорная» — реплика загрузки, помеченная опорным фактом вопроса.\n"]  # fmt: skip
    for pid, it in pick:
        spec, ft, _ = C.load_turns(ROOT, pid)
        q = it["q"]
        facts = "; ".join(f"{f}: {spec['facts'][f]['text']}" for f in q["anchors"])
        L += [f"## {pid} {q['id']}\n", f"**Вопрос:** {q['text']}\n", f"**Опорные факты:** {facts}\n",
              "### Сейчас (сущность / лемма)\n"]  # fmt: skip
        L += example_block(arc := it["arc"], q, spec, ft, v9.text_sleeves(arc, q["text"], query_mode=MODE),
                           "редкие сущности-понятия вопроса, которые есть в архиве (по возрастанию df), затем редкие леммы; "
                           "до 4; запрос = слово + предложение вопроса с ним («clause»).")  # fmt: skip
        L.append("### Фразы (группа слов по теме)\n")
        L += example_block(arc, q, spec, ft, v9.phrase_sleeves(arc, q["text"], **PHRASE),
                           "те же головы; группа = прилагательные слева + родительный падеж справа"
                           + (" + один предложный оборот" if PHRASE["with_pp"] else "")
                           + f", до {PHRASE['max_len']} слов; пересекающиеся группы склеены; запрос = сама фраза.")  # fmt: skip
    return "\n".join(L) + "\n"


PHRASE = {"max_len": 6, "with_pp": True}


def pool_cov(arc, q, ft, text, sleeves) -> int:
    c = v9.candidates(arc, q["text"], sleeves=sleeves, k_per_sleeve=8)
    return C.anchor_coverage(q["anchors"], ft, text, shown_ids=c["ranked"])["n"], len(c["ranked"])


def tune_phrases(probe) -> list[str]:
    res = {}
    for ml, pp in GRID + [("entity", None)]:
        n = size = 0
        for pid, it in probe:
            _, ft, text = C.load_turns(ROOT, pid)
            arc, q = it["arc"], it["q"]
            ss = (v9.text_sleeves(arc, q["text"], query_mode=MODE) if ml == "entity"
                  else v9.phrase_sleeves(arc, q["text"], max_len=ml, with_pp=pp))  # fmt: skip
            a, b = pool_cov(arc, q, ft, text, ss)
            n, size = n + a, size + b
        res[(ml, pp)] = (n, size)
    need = sum(len(dict.fromkeys(it["q"]["anchors"])) for _, it in probe)
    best = max(GRID, key=lambda g: (res[g][0], -res[g][1], -g[0]))
    PHRASE.update(max_len=best[0], with_pp=best[1])
    L = ["## 2. Рукава-фразы: подбор на 20 пробных (пробные)\n",
         "Цель — полнота объединённого пула рукавов (k = 8 на рукав) по репликам; при равенстве — меньший пул, короче фраза.\n",
         "| рукава | опорных фактов в пуле | средний размер пула |", "|---|---|---|"]  # fmt: skip
    for g in GRID + [("entity", None)]:
        name = "сущность / лемма (сейчас)" if g[0] == "entity" else f"фраза ≤ {g[0]} слов, предл. оборот {'да' if g[1] else 'нет'}"
        L.append(f"| {name} | {res[g][0]}/{need} ({100 * res[g][0] / need:.0f} %) | {res[g][1] / len(probe):.1f} |")
    L.append(f"\n**Выбрано:** фраза ≤ {best[0]} слов, предложный оборот — {'да' if best[1] else 'нет'}.\n")
    return L


def coverage_rows(pairs, count, rr):
    rows = []
    for pid, it in pairs:
        spec, ft, text = C.load_turns(ROOT, pid)
        arc, q = it["arc"], it["q"]
        mem6 = M.v6_hybrid(arc, q["text"], count, A.BUDGET)
        mem2 = M.v2_rag(arc, q["text"], count, A.BUDGET)
        mems, _ = SEL.select(arc, q["text"], count, A.BUDGET, rr.score)
        memp, _ = SEL.v2_plus(arc, q["text"], count, A.BUDGET, rr.score)
        fc = {k: C.fact_coverage(q["anchors"], spec["facts"], ft, text, m)
              for k, m in (("v2", mem2), ("v6", mem6), ("v2p", memp), ("sel", mems))}  # fmt: skip
        t6 = C.anchor_coverage(q["anchors"], ft, text, mem6)
        row = {"pid": pid, "q": q["id"], "type": q["type"], "need": fc["v6"]["need"], "v6_turn": t6["n"],
               "ev6": fc["v6"]["evidence"], "evs": fc["v2p"]["evidence"], "turn_in6": t6["in"],
               "mem6": mem6, "mems": memp}  # fmt: skip
        for k, m in (("v2", mem2), ("v6", mem6), ("v2p", memp), ("sel", mems)):
            row[k], row[k + "_tok"] = fc[k]["n"], count(m) if m else 0
        rows.append(row)
    return rows


VARS = [("v2", "V2 RAG"), ("v6", "V6"), ("v2p", "V2+ сильный RAG"), ("sel", "V6 + новый отбор")]


def cov_table(rows, title, boot) -> list[str]:
    types = [(ty, ru) for ty, ru in A.TYPES if any(r["type"] == ty for r in rows)] + [("all", "все")]
    L = [f"### {title}\n",
         "| тип (вопросов) | V6 по реплике (старая метрика) | " + " | ".join(n for _, n in VARS)
         + " | V2+ − V6, п.п. [95 %] | V6+отбор − V2+, п.п. [95 %] |",
         "|---|---|" + "---|" * (len(VARS) + 2)]  # fmt: skip
    for ty, ru in types:
        rs = [r for r in rows if ty == "all" or r["type"] == ty]
        need = sum(r["need"] for r in rs)
        cells = []
        for a, b in (("v2p", "v6"), ("sel", "v2p")):
            d = [(r[a] - r[b]) / r["need"] for r in rs if r["need"]]
            ci = A.boot(d) if boot else None
            cells.append(f"{100 * np.mean(d):+.1f}" + (f" [{100 * ci[0]:+.1f}; {100 * ci[1]:+.1f}]" if ci else ""))
        L.append(f"| {ru} ({len(rs)}) | {A2.pct(sum(r['v6_turn'] for r in rs), need)} | "
                 + " | ".join(A2.pct(sum(r[k] for r in rs), need) for k, _ in VARS) + " | " + " | ".join(cells) + " |")  # fmt: skip
    L.append("\nВсе опорные факты в памяти (вопросов): "
             + ", ".join(f"{n} {sum(r[k] == r['need'] > 0 for r in rows)}" for k, n in VARS)
             + ". Токенов памяти в среднем: "
             + ", ".join(f"{n} {np.mean([r[k + '_tok'] for r in rows]):.0f}" for k, n in VARS)
             + f" (бюджет {A.BUDGET}).\n")  # fmt: skip
    return L


def check_file(rows, items_spec) -> str:
    """30 случаев: 15 засчитанных и 15 не засчитанных; среди не засчитанных — сначала те, где
    реплика факта есть в памяти V6 (старая метрика засчитала бы), — самые спорные."""
    rng = random.Random(1)
    yes, no_turn, no = [], [], []
    for r in rows:
        for var, ev, mem in (("V6", r["ev6"], r["mem6"]), ("V2+", r["evs"], r["mems"])):
            for f, e in ev.items():
                case = (r, var, f, e, mem)
                if e["hit"]:
                    yes.append(case)
                elif var == "V6" and f in r["turn_in6"]:
                    no_turn.append(case)
                else:
                    no.append(case)
    pick_no = rng.sample(no_turn, min(10, len(no_turn)))
    pick_no += rng.sample(no, 15 - len(pick_no))
    cases = rng.sample(yes, 15) + pick_no
    rng.shuffle(cases)
    L = ["# Проверка метрики полноты на уровне факта — 30 случаев (seed 1)\n",
         "Правило: опорный факт засчитан, если в блоке памяти целиком есть предложение из помеченной реплики, "
         "которое несёт факт. «Несёт» — доля лемм текста факта в предложении ≥ 0,25 и ≥ 0,75 от лучшего предложения. "
         "15 засчитанных и 15 не засчитанных; из не засчитанных до 10 — случаи, где реплика факта в памяти V6 есть "
         "(старая метрика засчитала бы по началу реплики).\n",
         "Для каждого: верно ли решение метрики? (да / нет + почему)\n"]  # fmt: skip
    for i, (r, var, f, e, mem) in enumerate(cases, 1):
        spec = items_spec[r["pid"]]
        L += [f"\n## C{i:02d} · {r['pid']} {r['q']} · {f} · {var} · метрика: **{'засчитан' if e['hit'] else 'не засчитан'}**\n",
              f"**Факт:** {spec['facts'][f]['text']}\n",
              "**Предложения, несущие факт (по правилу):**\n"]  # fmt: skip
        L += [f"- {s}" for s in e["sentences"]] or ["- (не найдено)"]
        L += [f"\n**Совпало в памяти:** {e['hit'] or '—'}\n", "**Блок памяти:**\n", "```text", mem or "(пусто)", "```",
              "\n**Ваша оценка:** ___\n"]  # fmt: skip
    return "\n".join(L) + "\n"


def main() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b.backends import E5Encoder

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    enc = A.CachedEncoder(E5Encoder())
    known = json.loads(A.PREFIX.read_text("utf-8"))
    items = {pid: A.reconstruct(pid, enc, count, known) for pid in A.NEW + A.OLD}
    probe = A2.probe_items(items)
    L = ["# E1d этап A4 — рукава-фразы и общий слой отбора по предложениям (разведка, без API)\n",
         "Примеры рукавов — `reports/e1d/sleeve_examples.md`; ручная проверка метрики — `reports/e1d/coverage_check.md`.\n"]  # fmt: skip
    L += tune_phrases(probe)
    (A.OUT / "sleeve_examples.md").write_text(examples(items), "utf-8")
    # рукава-фразы на основных: полнота пула рукавов (как в п. 2), для сравнения
    agg = {"entity": [0, 0], "phrase": [0, 0]}
    need = 0
    for pid in A.NEW:
        _, ft, text = C.load_turns(ROOT, pid)
        for it in items[pid]:
            arc, q = it["arc"], it["q"]
            need += len(dict.fromkeys(q["anchors"]))
            for k, ss in (("entity", v9.text_sleeves(arc, q["text"], query_mode=MODE)),
                          ("phrase", v9.phrase_sleeves(arc, q["text"], **PHRASE))):  # fmt: skip
                a, b = pool_cov(arc, q, ft, text, ss)
                agg[k][0] += a
                agg[k][1] += b
    L.append(f"На 200 основных (по репликам, пул рукавов k = 8): сущности {100 * agg['entity'][0] / need:.0f} % "
             f"(пул {agg['entity'][1] / 200:.1f}), фразы {100 * agg['phrase'][0] / need:.0f} % (пул {agg['phrase'][1] / 200:.1f}).\n")  # fmt: skip
    rr = SEL.Reranker(SEL.RERANKER, revision=REV)
    rr.persist(ROOT / ".cache" / "e1d_rerank.jsonl")
    L += ["## 3. Общий слой отбора и главный соперник V2+, полнота на уровне факта\n",
          "**V2+ «сильный RAG»** (главный соперник): гибрид RRF(эмбеддинги, BM25) по всему вопросу без графа, "
          "пул 30 реплик → предложения → тот же cross-encoder → 600 токенов. **V6 + новый отбор** — то же, но пул V6 (с графом). "
          "V2 и V6 — как в E1c.\n",
          f"Новый отбор: пул 30 реплик по всему вопросу (ранжирование V6) → предложения → cross-encoder `{rr.name}` (CPU) → "
          f"лучшие предложения в {A.BUDGET} токенов, с соседним предложением, если влезает. "
          "**Метрика на уровне факта предварительная — до ручной проверки Егора.**\n"]  # fmt: skip
    pr = coverage_rows(probe, count, rr)
    L += cov_table(pr, "Пробные 20 (старые E1b) — пробные", boot=False)
    mr = coverage_rows([(pid, it) for pid in A.NEW for it in items[pid]], count, rr)
    L += cov_table(mr, "Основные 200", boot=True)
    specs = {pid: C.load_turns(ROOT, pid)[0] for pid in A.NEW}
    (A.OUT / "coverage_check.md").write_text(check_file(mr, specs), "utf-8")
    slim = [{k: v for k, v in r.items() if k not in ("mem6", "mems", "ev6", "evs")} for r in pr + mr]
    (A.OUT / "stage_a4.json").write_text(json.dumps({"phrase": PHRASE, "rows": slim}, ensure_ascii=False), "utf-8")
    (A.OUT / "stage_a4.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
