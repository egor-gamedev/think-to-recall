> **Translation of the frozen document `reports/prereg_bench1b.md` at commit 17c0871 (private development repository; the text is identical to `reports/prereg_bench1b.md` here); the Russian original is authoritative.**

# Benchmark 1b pre-registration (second attempt): a graph gate on MuSiQue / 2WikiMultiHopQA

> **Status: FROZEN 2026-10-31 by Egor's "approved", before the run on the test sets.** Any later change is an amendment only, labelled "before / after viewing results".
> This is the **second attempt**. The first is `reports/prereg_bench1.md` and `reports/result_bench1.md`: its hypothesis was not supported. The report will show both.
> Tuning and checks are on MuSiQue dev only. The dev-100, dev-200 and dev-200b sets do not overlap with each other or with the test sets — checked by a test. Details in `reports/bench1b/dev_report.md`.

## 1. What we test

Egor's invariant:
- without a strong graph signal, the ranking is identical to the base;
- the graph only adds or raises candidates, and only on a strong signal.

We test two things:
- whether the graph with such a gate improves recall over dense;
- whether it improves over a strong competitor, dense + reranking.

## 2. Data and graph (as in benchmark 1, unchanged)

- **MuSiQue** (main set): 1 000 questions, a corpus of 11 656 passages (the HippoRAG 2 release, `osunlp/HippoRAG_2`, rev 5ec05b38de).
- **2Wiki** (secondary set): 1 000 questions.
  - Nothing was tuned on 2Wiki. The result there is a transfer check.
- The graph is built by the same code as in benchmark 1 after amendment 1. The graph-building code did not change.
- Same models:
  - e5-base `intfloat/multilingual-e5-base` @ d128750597;
  - bge-reranker-v2-m3 @ 953dc6f6f8, fp32 on Kaggle GPU.

## 3. Variants (passage Recall@2 and @5; a passage scores the maximum over its sentences)

| | variant |
|---|---|
| (0) | dense e5 (= variant (a) of benchmark 1) |
| (K) | dense: top 40 sentences → bge-reranker-v2-m3 fp32, no graph |
| (G) | (0) + gate |
| (KG) | (K) + the same gate |

The original's labels are (К), (Ш), (КШ); this translation writes them (K), (G), (KG).

**The gate** (`ctxgraph/v2/gate.py::ppr_gated`, configuration `scripts/bench1b_config.json`):
1. Starts — the base's top 3 passages and the question's concepts.
2. **Signal.** A passage outside the top five intersects ≥ 2 different starts through shared concepts.
   - A link to another start passage counts through any shared concept except the question's concepts.
   - A link to a question concept counts if the passage contains it.
3. **PPR margin.** PPR is run from the same starts. Half of the mass goes to the sentences of the start passages (in proportion to dense), half to the question's concepts (weighted by rarity). Damping 0.5, as in HippoRAG 2.
   - A candidate displaces the lowest passage of the top five only if its PPR score is **4 times** larger. At most 2 passages are displaced.
4. Without a signal, PPR is not run and the ranking is the base.

**Concept rarity is a smooth IDF weight = log(N / df)**, with no hard threshold. Here N is the number of corpus sentences and df the number of sentences containing the concept. Chosen by Egor's rule on dev-200b, see §7.
- IDF sets the start weight of the question's concepts in PPR (like node specificity in HippoRAG 1).
- There is no df cut-off. So with IDF almost every question has a signal: 200 of 200 on dev-200b.
- In practice, ranking changes are decided by the intersection of ≥ 2 starts and the **PPR margin × 4**. This departs from the letter of "strong signal = rare concept", and Egor should confirm it.
- The invariant still holds formally: without a candidate that passes the margin, the ranking = base.

**Tuned parameters.** These are our own knobs, not borrowed. Tuned on dev-100 over a grid of 200 configurations.

| parameter | value |
|---|---|
| a — number of start passages | 3 |
| PPR margin | × 4 (m = 3) |
| s — at most displaced | 2 |
| start mass split | 0.5 / 0.5 |

**Borrowed parameters:**
- damping 0.5 and the weight of dense starts (HippoRAG 2);
- the IDF weight log(N/df) (standard, like node specificity in HippoRAG 1).

## 4. Hypotheses (MuSiQue, Recall@5), Holm correction over two

**H1.** (G) beats (0).
- Test — one-sided paired sign-flip permutation test over questions: 10 000 permutations, seed 0.
- Supported if the Holm-corrected p < 0.05.

**H2.** (KG) beats (K). This is a comparison with a strong competitor, and it is **not descriptive**: the same test, the same Holm correction.

**For each hypothesis we report:**
- the R@5 difference with a 95 % interval (paired bootstrap, 10 000, seed 0);
- wins and losses on questions whose top five changed;
- the share of questions with a signal.

**Implementation check (H0).** Where the gate changed nothing, the ranking is identical to the base. This is checked on every question of the run. If violated, the run is aborted and considered invalid.

## 5. Dev expectation (written before the run)

| | dev-100 (tuning) | dev-200 | dev-200b |
|---|---|---|---|
| **H1, IDF (chosen)** | +0.8 [−1.2; +2.8]; 5 / 3 | +2.4 [+0.4; +4.5]; 24 / 10 | **+2.4 [−0.0; +4.8]; 25 / 13** |
| H1, relative threshold | +3.0 [+1.1; +5.2]; 8 / 0 | +4.2 [+2.6; +6.0]; 27 / 1 | +1.3 [−0.7; +3.3]; 18 / 10 |
| H1, absolute threshold df ≤ 10 (descriptive) | +3.0; 8 / 0 | +1.9 [+0.4; +3.4]; 14 / 2 | +0.8 [−1.1; +2.5]; 14 / 9 |
| **H2, IDF (chosen): R@5 (KG) − (K); wins / losses** | **−1.2 [−4.1; +1.2]; 2 / 6** | not computed (reranking on CPU) | — |
| H2, relative threshold | −1.6 [−4.4; +0.7]; 1 / 5 | not computed | — |
| H2, IDF, fp32 reranking (Kaggle debug kernel) | −1.8 [−4.0; +0.5]; 2 / 9 | — | — |

- **The H2 expectation is about zero or worse.**
  - The "5 wins to 4 losses" pair on dev-100 belonged to another configuration over reranking: m = 1, s = 1.
  - With the frozen configuration, dev-100 gives 2 / 6 (IDF) and 1 / 5 (threshold).
  - So, by dev, the gate on top of reranking does not help. H2 is tested because it is an honest comparison with a strong competitor.
- **By dev, H2 is negative: the graph on top of reranking violates the "not worse" principle.** The reason: the gate displaces the reranker's choice with a PPR candidate. The fix — a candidate displaces only if its reranker score is not much lower. It goes into the next stage, not into bench1b.
- **The H1 expectation** is a small gain, about +2 pp.
  - On dev-200b the lower bound of the interval is about zero, so H1 may not be supported.
  - The test corpus is 6.4 times larger than dev, and the signal may behave differently.

## 6. Run

- Kaggle GPU, private kernel `egor123hhh/ctxgraph-bench1b-run`; the code is a new version of the private dataset `ctxgraph-bench1`. LLM calls — 0.
- Before the freeze — a debug run of the kernel on dev-100 (MODE = debug).
  - It must reproduce the dev-100 numbers for (0) and (G).
  - (K) and (KG) there run in fp32 and may differ slightly from int8.
- **The debug run was done on 2026-10-31** (kernel v1, `reports/bench1b/kaggle_debug_dev100_summary.json`):
  - the same graph down to the edge: 7 599 sentences, 22 549 concepts, 92 838 edges;
  - (0) and (G) matched the local ones: R@5 59.5 and 60.2; 59 changed questions, 5 wins / 3 losses;
  - the top fives of (0) matched in 100 of 100 questions;
  - the invariant held: 41 of 41 and 34 of 34 unchanged questions = base;
  - (K) fp32 — 65.2 vs 65.0 for int8; (KG) — 63.5 vs 63.7.
- Then the freeze (commit of the pre-registration and the configuration) and **one run** (MODE = test): MuSiQue, then 2Wiki.
- Technical amendments only before viewing results, labelled.

## 7. How rarity was chosen (a decision after viewing — labelled)

- On dev-100 the absolute threshold df ≤ 10 was chosen. On dev-200 it gave 14 / 2.
- **After viewing dev-200**, the idea came up to scale the threshold to the corpus size. On dev-200 that gave 27 / 1.
- So both candidates were checked on **a new set of 200 questions, dev-200b**, unchanged:
  - (1) the relative threshold;
  - (2) a smooth IDF weight with no threshold.
- Egor's rule: if both have more wins than losses, take (2) — it is borrowed rather than tuned. If only one "holds", take it. If neither, the absolute threshold df ≤ 10.

**Result on dev-200b**, both variants unchanged:

| variant | questions with a signal | changed | wins | losses | R@5 − dense, pp [95 %] |
|---|---|---|---|---|---|
| (1) relative threshold | 163 | 110 | 18 | 10 | +1.3 [−0.7; +3.3] |
| (2) IDF | 200 | 146 | 25 | 13 | +2.4 [−0.0; +4.8] |

- Both variants hold "more wins than losses". By Egor's rule, **(2), IDF, is chosen**.
- **Important:** on dev-200b the relative threshold is clearly weaker than on dev-200 (18 / 10 vs 27 / 1). The dev-200 result was optimistic — the usual price of choosing after viewing.

## 8. Report

- A table of both attempts: benchmark 1, variants (a)–(d), and benchmark 1b, variants (0), (K), (G), (KG).
- Label: "second attempt; tuned on MuSiQue dev; both shown".
- For each hypothesis: p, Holm-corrected p, interval, wins and losses.
- By question type: the first three wins and the first three losses with their start concepts.
- 2Wiki — descriptive.

## Amendment 1 (2026-11-01, after viewing the results of version d707ea0): a rerun on code 590126d

- **Reason for the rerun.** On conversational data (IMLogic dev), a bug in English concept extraction was found: the possessor in a possessive phrase was merged with the concept, and the "'s" marker got into the key. Fix — commit `590126d`, description — `reports/implicit/fix1_changes.md`.
  - The fix changes the graph and therefore the gate's ranking. The rerun is needed so that all main results are on one code version.
- **What does not change.** The data, variants (0), (K), (G), (KG), hypotheses H1 and H2, the test, the Holm correction, the gate configuration (`scripts/bench1b_config.json`) and the models. One run, on Kaggle GPU.
- **Regression check on dev-100** (the test sets were not touched):
  - the gate's ranking changed in 18 of 100 questions;
  - R@5 60.2 → 60.8, +0.5 [0.0; +1.2], p = 0.52 — recall within noise;
  - dense did not change.
- **The original result is kept** in the report as the result of the previous version and is not replaced: H1 +2.5 pp [+1.4; +3.5] supported; H2 +0.7 [−0.2; +1.6] not supported.
- **If they diverge,** both versions go into the report, and `590126d` is considered primary.
- **Clarification 2026-11-05 (Egor).** The fix was found **after** the d707ea0 result was viewed. So the primary result is d707ea0, and 590126d is secondary, labelled "change after viewing". The previous point is cancelled.
