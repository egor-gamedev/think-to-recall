> **Translation of the frozen document `reports/result_stage1.md` at commit 266a380 (private development repository; the text is identical to `reports/result_stage1.md` here); the Russian original is authoritative.**

# Stage 1 (implicit links, retrieval only): result on IMLogic (2026-11-09)

> Under the pre-registration `reports/prereg_implicit.md` (frozen 2026-11-07; code 590126d; amendments 1–3 are technical, made before results were viewed).
> Run — Kaggle GPU, kernel `ctxgraph-implicit-run` v7, 2026-11-07; LLM calls — 0. Per-question rankings — `reports/implicit/main/imlogic/{A,B,C,D0,D}.jsonl`; summary — `reports/implicit/main/imlogic_main_summary.json`; timing — `imlogic_main_timing.jsonl`.

## What was and was not done

- **IMLogic, main part:** done. 18 users, **1 931 questions**, gold `memory_l`.
- **Stage 1 on ImplexConv was not run because of the weekly GPU quota; it is postponed.** Four main-part launches crashed for technical reasons (amendments 2 and 3) before results were viewed. The last, fixed one (v12) could not be launched: the weekly Kaggle GPU quota was used up.
- **The Holm correction is over the tests actually run:** 2 tests (S1 and S2 on IMLogic), not 4 as in the pre-registration (§4). This deviation from the pre-registration is forced: the ImplexConv tests were not run.
- **The invariant** "no change — ranking = base" held for C and D on every question: the run was not aborted.
- `memory_s` (descriptive under §4) was not computed: the rankings store only the `memory_l` gold.

## Recall against `memory_l`, % (1 931 questions)

| variant | R@1 | R@5 | R@10 | R@30 |
|---|---|---|---|---|
| A — dense e5 | 1.7 | 5.4 | 9.8 | 22.1 |
| B — graph only (for reference) | 0.7 | 3.0 | 4.6 | 9.7 |
| C — A + graph gate | 1.7 | 5.0 | 9.9 | 21.9 |
| D0 — EverMemOS search stack without LLM memory construction | 2.1 | **8.2** | **14.1** | 22.9 |
| D — D0 + graph gate | 2.1 | 7.3 | 13.8 | 22.9 |

## Comparisons (R@5; one-sided paired permutation test, 10 000, seed 0; Holm correction over the 2 tests run)

| | comparison | difference, pp [95 %] | wins / losses | p | Holm p (2 tests) | reverse test, uncorrected p |
|---|---|---|---|---|---|---|
| **S1** | C − A: graph on top of ordinary memory | −0.5 [−1.1; +0.2] | 17 / 26 | 0.93 | 1.00 | A > C: 0.11 |
| **S2** | D − D0: graph on top of EverMemOS-style search | −0.9 [−1.6; −0.2] | 13 / 30 | 0.997 | 1.00 | **D0 > D: 0.008** |

- Wins and losses are counted over questions whose R@5 changed.
- The gate changed the top five in 1 330 of 1 931 questions (C) and in 1 175 (D). R@5 changed in only 43 questions: most graph insertions do not hit the gold.
- The gate's signal (an intersection of ≥ 2 starts) is present almost always: 1 924 and 1 929 questions. This is the expected price of the IDF weight with no threshold, as in bench1b.

## Conclusion under the §5 criterion

- **The graph seeded from the question adds nothing** to ordinary memory: S1 is not significant, and the difference is negative.
- **On top of D0 the graph significantly hurts:** the reverse test D0 > D gives p = 0.008 < 0.05. The "D not worse than D0" condition fails on IMLogic.
- So a "continue" conclusion under §5 is impossible whatever the ImplexConv outcome: "D not worse than D0" is required on both sets.
- Under the pre-registration (§5), a negative stage 1 does not close the hypothesis of search over reasoning steps. Stage 2 tested it (`reports/result_stage2.md`).

## Descriptive

**D0 is stronger than dense:** 8.2 vs 5.4 R@5. The EverMemOS stack without LLM memory construction is the best base on IMLogic.

R@5 by IMLogic question category:

| category | n | A | C | D0 | D |
|---|---|---|---|---|---|
| Contextual Fit | 237 | 11.8 | 11.0 | 16.9 | 15.2 |
| Goal/Mission Alignment | 257 | 4.7 | 5.4 | 8.6 | 7.0 |
| Identity Shift | 253 | 4.7 | 4.7 | 2.4 | 3.6 |
| Motivational Orientation | 500 | 7.8 | 6.2 | 11.8 | 10.2 |
| Resource Constraint | 121 | 1.7 | 1.7 | 5.0 | 5.0 |
| State Constraint | 543 | 2.2 | 2.0 | 4.4 | 3.7 |
| Other | 20 | 0.0 | 0.0 | 5.0 | 5.0 |

## Time (Kaggle GPU)

- **Building the graph** (e5, spaCy, writing): 191 s for 13 521 sentences — **14 ms per sentence**. This is far from the 0.5 s-per-sentence threshold of practical applicability (§8).
- **Qwen3-Embedding-4B embeddings** for D0: 349 s for 13 431 facts. Stack index: 6.5 s.
- **Search, s per question:**
  - A — 0.018; B — 0.015; C — 0.034;
  - D0 — 5.74 (almost all of it is the Qwen3-Reranker-4B reranker); D — 5.78.
