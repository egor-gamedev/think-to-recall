# Paused on 2026-10-30 (Egor's decision): E1d stage B and E3 "Remembers decisions"

Reason: the graph architecture is being reviewed first (`ARCHITECTURE_REVIEW.md`). No benchmark runs
with answers or judges until Egor approves `ARCHITECTURE_V2.md`. Our E1b/E1c sets stay as debugging
sets only; conclusions will come from external benchmarks (LongMemEval, LoCoMo, ...).

Code stays in place (moving it would break imports and tests); this folder holds the task texts and
this status note. Nothing below is to be resumed without Egor's explicit go-ahead.

## E1d (V9 "thinking sleeves") — `V9_TASK.md`

- Stage A done (exploration, no API): `reports/e1d/stage_a.md`, `stage_a2.md`, `stage_a3.md`.
  Result: sleeves built from the question add nothing over whole-question search at equal pool size.
- Stage A4 (phrase sleeves, sentence-level selection with bge-reranker-v2-m3, V2+ "strong RAG",
  fact-level coverage metric) — code done (`scripts/e1d_stage_a4.py`, `ctxgraph/e1b/selection.py`,
  `coverage.fact_coverage`, `v9.phrase_sleeves`), **not computed**: the CPU run was stopped at
  ~4 300 of ~35 000 reranker pairs. Remaining pairs are dumped for a Kaggle GPU run
  (`scripts/e1d_rerank_pairs.py`, `kaggle/e1d_rerank/`); dataset not uploaded, kernel not pushed.
  `reports/e1d/coverage_check.md` (30 cases for Egor) not produced yet.
- Stage B code done, never run on a real model: `ctxgraph/e1b/steps.py` (V9 / V9-control step
  protocol), runner variants 21 / 90 / 91 / 92, stub-model tests.
- Stage C model probe (`scripts/e1d_probe_models.py`) interrupted: only Mistral Large 4 (NVIDIA,
  free) finished 5/5 questions — median call 3.4 s, max 27.5 s, no errors, but 8 of 15 reasoning
  steps came back empty (protocol problem to fix if resumed). MiniMax / Nemotron-think / DeepSeek
  (timeweb, 40 ₽ cap, ~0 ₽ spent) / GLM not run.
- Free NVIDIA models that answered a ping on 2026-10-29: Nemotron-3 Ultra, Mistral Large 4,
  MiniMax M3.1 Flash, gpt-oss-20b (slow). Hung: DeepSeek V4.1 Flash, GLM-5.3 Flash, Gemma 4. Kimi 404.

## E3 "Remembers decisions" — `DECISION_BENCH.md`

- Step 1 done for clinic-bot (no API): `benchmark_e3/projects/clinic-bot/` (20 refusal events —
  11 already in the load, 9 inserted — and 40 probe questions), loader/validator `ctxgraph/e3/data.py`,
  review table `reports/e3/clinic-bot_review.md`, cost estimate `reports/e3/cost_estimate.md`.
- Open item if resumed: filler turns to reach "after 5 compressions" for late events.

## Budget

timeweb balance ≈ 114 ₽ (Egor, 2026-10-29). Free NVIDIA API is the default for any future model calls.
