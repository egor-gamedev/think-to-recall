"""English path check + build-time estimates for external benchmarks (debug, <5 min, no API).

    python scripts/v2_english_check.py   # -> reports/v2_english_and_estimates.md

1. 30 supporting MuSiQue paragraphs (seed 0): concepts per paragraph, whether the hop answer of that
   paragraph (the bridge entity of the decomposition) is a concept, time — spaCy path vs the old
   rules extractor (`ctxgraph.ingest.extract_entities`).
2. Build-time estimate for MuSiQue, 2Wiki and LongMemEval-S: passages / sessions and sentences
   counted from the data; throughput measured on this CPU (spaCy, e5-base); GPU figures are typical
   numbers to be checked on the first Kaggle run.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import text as T  # noqa: E402
from ctxgraph.ingest import extract_entities  # noqa: E402
from ctxgraph.sources.experience import is_concept  # noqa: E402
from ctxgraph.v2 import english  # noqa: E402
from ctxgraph.v2 import extract as X  # noqa: E402

B = ROOT / ".cache" / "bench"
GPU_E5 = (500, 1000)  # sentences/s, typical T4/P100 for e5-base — to be checked on Kaggle


def lemmas(xs) -> set[str]:
    return {w for x in xs for w in T.content_lemmas(x)}


def check30() -> tuple[list[str], dict]:
    qs = json.loads((B / "musique.json").read_text("utf-8"))
    items = []
    for q in qs:
        ps = {p["idx"]: p for p in q["paragraphs"]}
        for d in q["question_decomposition"]:
            p = ps.get(d.get("paragraph_support_idx"))
            if p is not None:
                items.append((p["title"], p["paragraph_text"], d["answer"]))
    rng = random.Random(0)
    pick = rng.sample(items, 30)
    res = {"spacy": [0, 0, 0.0], "rules": [0, 0, 0.0]}
    rows = []
    for title, text, ans in pick:
        t0 = time.time()
        sp = [c for s in X.sentences(text) for c in english.concepts_en(s[1])]
        t1 = time.time()
        ru = [e.surface for e in extract_entities(text) if is_concept(e.canonical, e.kind, e.surface)]
        t2 = time.time()
        al = set(T.content_lemmas(ans)) or {ans.lower()}
        hit_sp = al <= lemmas(s for _, s, _ in sp) | lemmas(k for k, _, _ in sp)
        hit_ru = al <= lemmas(ru)
        for k, n, h, dt in (("spacy", len(sp), hit_sp, t1 - t0), ("rules", len(ru), hit_ru, t2 - t1)):
            res[k][0] += n
            res[k][1] += h
            res[k][2] += dt
        rows.append(f"| {title[:40]} | {ans[:30]} | {'да' if hit_sp else '—'} | {'да' if hit_ru else '—'} | {len(sp)} | {len(ru)} |")
    L = ["## 1. Английский путь: 30 опорных абзацев MuSiQue (seed 0)\n",
         "«Ответ шага — понятие»: все леммы ответа подвопроса, к которому относится абзац (мост многохопового "
         "вопроса), есть среди извлечённого из абзаца.\n",
         "| | spaCy (en_core_web_sm 3.8.0, MIT) | старый извлекатель (правила, YAKE) |", "|---|---|---|",
         f"| понятий на абзац | {res['spacy'][0] / 30:.1f} | {res['rules'][0] / 30:.1f} |",
         f"| ответ шага — понятие | {res['spacy'][1]}/30 | {res['rules'][1]}/30 |",
         f"| секунд на абзац (CPU) | {res['spacy'][2] / 30:.3f} | {res['rules'][2] / 30:.3f} |",
         "\n<details><summary>По абзацам</summary>\n", "| абзац | ответ шага | spaCy | правила | понятий spaCy | понятий правила |",
         "|---|---|---|---|---|---|"] + rows + ["\n</details>\n"]  # fmt: skip
    return L, {"spacy_s_per_passage": res["spacy"][2] / 30}


def corpus_stats(name: str, fname: str) -> dict:
    c = json.loads((B / fname).read_text("utf-8"))
    sents = [len(X.sentences(p["text"], min_words=1)) for p in c]
    return {"name": name, "passages": len(c), "sentences": sum(sents)}


def lme_stats() -> dict:
    d = json.loads((B / "longmemeval_s_cleaned.json").read_text("utf-8"))
    total_turns = total_sents = 0
    uniq: dict[str, int] = {}
    for q in d:
        for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"], strict=True):
            if sid not in uniq:
                uniq[sid] = sum(len(X.sentences(t["content"], min_words=1)) for t in sess)
            total_turns += len(sess)
            total_sents += uniq[sid]
    return {"name": "LongMemEval-S", "questions": len(d), "turns": total_turns, "sentences": total_sents,
            "unique_sessions": len(uniq), "unique_sentences": sum(uniq.values())}  # fmt: skip


def throughput() -> dict:
    from ctxgraph.e1b.backends import E5Encoder

    c = json.loads((B / "musique_corpus.json").read_text("utf-8"))[:200]
    sents = [s for p in c for _, s in X.sentences(p["text"])][:400]
    t0 = time.time()
    for s in sents:
        english.concepts_en(s)
    sp = len(sents) / (time.time() - t0)
    enc = E5Encoder()
    enc.encode(sents[:8])
    t0 = time.time()
    enc.encode(sents[8:208])
    e5 = 200 / (time.time() - t0)
    return {"spacy_sent_s": sp, "e5_sent_s": e5}


def main() -> None:
    L = ["# Английский путь и оценка времени построения графа для чужих бенчмарков (отладка, без API)\n"]
    l1, _ = check30()
    L += l1
    tp = throughput()
    stats = [corpus_stats("MuSiQue", "musique_corpus.json"), corpus_stats("2Wiki", "2wikimultihopqa_corpus.json"), lme_stats()]
    per_s = 1 / tp["spacy_sent_s"] + 1 / tp["e5_sent_s"]  # CPU seconds per sentence (extraction + embedding)

    def hrs(n_sent: int, gpu: int | None = None) -> str:
        sec = n_sent * (1 / tp["spacy_sent_s"] + (1 / gpu if gpu else 1 / tp["e5_sent_s"]))
        return f"{sec / 3600:.1f} ч" if sec >= 3600 else f"{sec / 60:.0f} мин"

    L += ["## 2. Оценка времени построения графа\n",
          f"Замер на этом CPU: spaCy ≈ {tp['spacy_sent_s']:.0f} предложений/с, e5-base ≈ {tp['e5_sent_s']:.0f} предложений/с "
          f"(≈ {per_s:.3f} с на предложение вместе). GPU Kaggle: e5-base {GPU_E5[0]}–{GPU_E5[1]} предложений/с — "
          "типичная оценка, проверить первым запуском; spaCy остаётся на CPU. Запись в SQLite и PPR не учтены (секунды).\n",
          "| набор | единиц | предложений | CPU | GPU Kaggle (e5) |", "|---|---|---|---|---|"]  # fmt: skip
    for s in stats[:2]:
        L.append(f"| {s['name']} | {s['passages']} абзацев | {s['sentences']} | {hrs(s['sentences'])} | "
                 f"{hrs(s['sentences'], GPU_E5[1])}–{hrs(s['sentences'], GPU_E5[0])} |")  # fmt: skip
    m = stats[2]
    L += [f"| {m['name']}, без кэша | {m['questions']} вопросов, {m['turns']} реплик | {m['sentences']} | {hrs(m['sentences'])} | "
          f"{hrs(m['sentences'], GPU_E5[1])}–{hrs(m['sentences'], GPU_E5[0])} |",
          f"| {m['name']}, кэш по сессиям | {m['unique_sessions']} уникальных сессий | {m['unique_sentences']} | {hrs(m['unique_sentences'])} | "
          f"{hrs(m['unique_sentences'], GPU_E5[1])}–{hrs(m['unique_sentences'], GPU_E5[0])} |",
          f"\nКэш по сессиям для LongMemEval-S сокращает объём в {m['sentences'] / max(m['unique_sentences'], 1):.1f} раза. "
          "Граф вопроса = слияние подграфов его сессий (дозапись); рёбра «синоним» и «subphrase» между сессиями достраиваются "
          "при слиянии.\n",
          "**Поправка к ARCHITECTURE_V2 §6:** там для LongMemEval-S без кэша стояло ≈ 250 тыс. предложений — занижено; "
          f"по данным {m['sentences']}.\n",
          "Recall@5 по абзацам (как в HippoRAG 2) для MuSiQue / 2Wiki считается без вызовов LLM; переранжирование "
          "bge-reranker-v2-m3 — на GPU (на CPU ≈ 5 пар/с: 1 000 вопросов × 60 пар ≈ 3,3 ч)."]  # fmt: skip
    (ROOT / "reports" / "v2_english_and_estimates.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
