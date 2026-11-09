"""LongMemEval-S: how much a session-hash cache saves, and the time for a 100-question subsample (no API).

    python scripts/v2_lme_cache.py   # -> reports/v2_lme_cache.md

Session key = sha1 of the session content (roles + texts), not the session id: identical sessions under
different ids are built once. Sentence key = sha1 of the sentence text (an even finer cache for
embeddings). Throughput: spaCy and e5 measured on this CPU in reports/v2_english_and_estimates.md.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.v2 import extract as X  # noqa: E402

B = ROOT / ".cache" / "bench"
SPACY, E5_CPU, E5_GPU = 135.0, 13.0, (500.0, 1000.0)  # sentences/s (measured CPU; GPU — typical, to check)


def h(x: str) -> str:
    return hashlib.sha1(x.encode()).hexdigest()


def stats(qs: list[dict]) -> dict:
    inst = 0
    by_id, by_hash, sent_hash = set(), {}, set()
    for q in qs:
        for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"], strict=True):
            key = h("\n".join(f"{t['role']}:{t['content']}" for t in sess))
            if key not in by_hash:
                ss = [s for t in sess for _, s in X.sentences(t["content"], min_words=1)]
                by_hash[key] = ss
                sent_hash.update(h(s) for s in ss)
            by_id.add(sid)
            inst += len(by_hash[key])
    return {"questions": len(qs), "session_ids": len(by_id), "session_hashes": len(by_hash),
            "sentence_instances": inst, "sentences_unique_sessions": sum(len(v) for v in by_hash.values()),
            "sentences_unique": len(sent_hash)}  # fmt: skip


def t(n: int, e5: float) -> str:
    sec = n / SPACY + n / e5
    return f"{sec / 3600:.1f} ч" if sec >= 3600 else f"{sec / 60:.0f} мин"


def main() -> None:
    d = json.loads((B / "longmemeval_s_cleaned.json").read_text("utf-8"))
    full = stats(d)
    sub = stats(random.Random(0).sample(d, 100))
    L = ["# LongMemEval-S: кэш по хэшу сессии и подвыборка 100 вопросов (без API)\n",
         f"Скорость: spaCy ≈ {SPACY:.0f} предл./с (CPU), e5-base ≈ {E5_CPU:.0f} предл./с CPU, {E5_GPU[0]:.0f}–{E5_GPU[1]:.0f} на GPU "
         "(типичная оценка, проверить на Kaggle). spaCy — на CPU в обоих случаях (узкое место); в 4 процесса — ≈ ×3–4.\n",
         "| | вопросов | сессий (id) | сессий (хэш содержимого) | предложений-вхождений | предложений в уникальных сессиях | уникальных предложений (хэш) |",
         "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, s in (("весь набор", full), ("подвыборка 100 (seed 0)", sub)):
        L.append(f"| {name} | {s['questions']} | {s['session_ids']} | {s['session_hashes']} | {s['sentence_instances']} | "
                 f"{s['sentences_unique_sessions']} | {s['sentences_unique']} |")  # fmt: skip
    L += ["\n| объём | CPU (spaCy + e5) | Kaggle GPU (spaCy CPU + e5 GPU) |", "|---|---|---|"]
    for name, n in (("весь набор, без кэша", full["sentence_instances"]), ("весь набор, кэш сессий", full["sentences_unique_sessions"]),
                    ("весь набор, кэш предложений", full["sentences_unique"]), ("100 вопросов, кэш сессий", sub["sentences_unique_sessions"]),
                    ("100 вопросов, кэш предложений", sub["sentences_unique"])):  # fmt: skip
        L.append(f"| {name} ({n}) | {t(n, E5_CPU)} | {t(n, E5_GPU[1])}–{t(n, E5_GPU[0])} |")
    L.append("\nКэш предложений: эмбеддинг и извлечение считаются один раз на уникальный текст предложения; граф вопроса "
             "собирается дозаписью из готовых предложений (ссылки на кэш), без повторного кодирования.")  # fmt: skip
    (ROOT / "reports" / "v2_lme_cache.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
