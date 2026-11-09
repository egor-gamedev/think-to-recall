"""Kaggle (GPU): баллы cross-encoder BAAI/bge-reranker-v2-m3 для пар этапа A4 E1d.

Вход — pairs.jsonl из датасета ctxgraph-e1d-rerank ([запрос, предложение] на строку).
Выход — /kaggle/working/scores.jsonl ([запрос, предложение, балл]). Модель и ревизия — как на CPU
(scripts/e1d_stage_a4.py), max_length 512; fp32, чтобы баллы совпадали с CPU-расчётом.
"""

import glob
import json
import subprocess
import sys
import time

REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=3"], check=False)

import torch  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402

src = glob.glob("/kaggle/input/**/pairs.jsonl", recursive=True)[0]
pairs = [json.loads(ln) for ln in open(src, encoding="utf-8") if ln.strip()]
print("пар:", len(pairs), "GPU:", torch.cuda.is_available(), flush=True)
m = CrossEncoder("BAAI/bge-reranker-v2-m3", revision=REV, device="cuda", max_length=512)
# контроль: те же пары, что посчитаны на CPU раньше, должны дать те же баллы (сверка — локально)
t0 = time.time()
order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))  # меньше паддинга
scores = [0.0] * len(pairs)
B = 128
for s in range(0, len(order), B):
    idx = order[s : s + B]
    out = m.predict([pairs[i] for i in idx], batch_size=B)
    for i, sc in zip(idx, out):
        scores[i] = float(sc)
    if s // B % 50 == 0:
        print(f"{s + len(idx)}/{len(pairs)} за {time.time() - t0:.0f} с", flush=True)
with open("/kaggle/working/scores.jsonl", "w", encoding="utf-8") as fh:
    for (q, t), sc in zip(pairs, scores):
        fh.write(json.dumps([q, t, sc], ensure_ascii=False) + "\n")
print("готово за", round(time.time() - t0), "с", flush=True)
