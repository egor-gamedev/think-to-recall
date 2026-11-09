> **Translation of the frozen document `reports/prereg_implicit.md` at commit 9caa86f (private development repository; the text is identical to `reports/prereg_implicit.md` here); the Russian original is authoritative.**

# Pre-registration: implicit links — IMLogic (memory-level) and ImplexConv-200, six variants

> **Status: FROZEN 2026-11-07 by Egor's decision ("freezing with changes"), before the run on the main questions.** This is **stage 1 — retrieval only, no LLM calls.**
> **Code: commit `590126d`.** The graph and the gate are as in bench1b, with the fix to the extraction of possessive phrases (`reports/implicit/fix1_changes.md`). The gate configuration `scripts/bench1b_config.json` does not change.
> **Combines and replaces** the drafts `prereg_imlogic.md` and `prereg_implexconv.md`: the "continue or not" criterion and the multiplicity correction are shared by both sets.
> IMLogic is **internal research only**: the repository has no license, and the underlying HaluMem is CC BY-NC-ND 4.0. ImplexConv is MIT.

## 1. Question

Does the graph find a decisive fact that is semantically far from the question better than its base:
- (a) on top of ordinary dense memory;
- (b) on top of an EverMemOS-style search stack?

## 2. Data

| | IMLogic memory-level | ImplexConv-200 |
|---|---|---|
| source | `ThisIsCosine/IMLogic`, downloaded 30.10 (`HaluMem-Medium.jsonl` and `qa/`) | HF `Kaylee0501/ImplexConv` @ baf436948f4b |
| retrieval unit | a user memory record (`memory_points[].memory_content`, deduplicated, about 750 per user) | a session: its score is the maximum over its sentences |
| gold | the decisive `memory_l`; additionally the similar `memory_s` | evidence sessions (`retrieved_conv` / `retrieved_conv_ids`), 100 % match |
| main | 18 users, **1 931 questions** | 100 opposed + 100 supportive examples (seed 0), **855 questions** (255 + 600) |
| dev (debugging only) | users 12 and 13, 285 questions | 10 other examples, 45 questions |

## 3. Stage 1 variants (all in one run)

| | variant | what it is |
|---|---|---|
| **A** | ordinary memory | dense e5-base over units; a unit scores the maximum over its sentences |
| **B** | graph only | PPR from the question's concepts only (IDF weight, damping 0.5), no dense; for reference |
| **C** | graph + RAG | A as the base + the graph gate (= (G) of bench1b) |
| **D0** | **the EverMemOS search stack without LLM memory construction** | see below |
| **D** | D0 + the graph gate | the base is the D0 ranking |

**D0 — the EverMemOS search stack without LLM memory construction.** Assembled from their own open library `everalgo` (Apache-2.0; everalgo-rank 0.3.0, everalgo-clustering 0.2.1 — the versions pinned by EverOS @ 648b34b). It reproduces their benchmark 1:1 (EverAlgo @ 7398133, `benchmarks/common/stages/search.py`), with settings from their `benchmarks/common/config.py`; **nothing is tuned**:
- **units ("episodes")** — IMLogic memory records or ImplexConv sessions; "atomic facts" — the record itself or the session's sentences. There is **no** LLM memory construction (MemCell, Foresight, profile);
- **dense** — MaxSim over fact embeddings (Qwen3-Embedding-4B @ 5cf2132, 1024 dimensions); **sparse** — MaxSim over BM25 of the facts (their NLTK + Porter tokenization); `hybrid_full` = RRF (50 candidates each, k = 40);
- **MemScene** — `cluster_by_geometry` (cosine ≥ 0.70, 7-day window; ImplexConv has no timestamps, so the window has no effect). 10 scenes are selected, their members are reranked by Qwen3-Reranker-4B @ 22e6836 with their instruction; the top 10 are taken;
- **final ranking** — round 1's top 10, then the rest of `hybrid_full` (so that Recall@k is defined beyond 10).

**D+ (D + an LLM sufficiency check) is postponed to "later"** and is not part of this run: stage 1 makes no LLM calls.

**Naming.** D0 is called "the EverMemOS search stack without LLM memory construction". The comparison of D and D0 is phrased as "a graph on top of EverMemOS-style search", **not "better than EverMemOS"**. A comparison with the real EverMemOS is possible only if the IMLogic authors send the memory they built — the letter asks for it (`reports/imlogic_letter_draft.md`, item 5; *translator's note: the draft is not published in this repository*).

## 4. Metric and pre-specified comparisons

**Main metric** — Recall@5 against the gold: `memory_l` for IMLogic, evidence sessions for ImplexConv.

**Descriptive:**
- Recall@1/10/30 (IMLogic) and @1/3/10 (ImplexConv);
- `memory_s` (IMLogic);
- opposed and supportive separately (ImplexConv);
- breakdown by category.

**Comparisons** — 2 per set, **4 tests, Holm correction over all four:**

| | comparison | what it tests |
|---|---|---|
| S1 | **C vs A** | the graph on top of ordinary memory |
| S2 | **D vs D0** | the graph on top of EverMemOS-style search |
| — | B | for reference only, no test |

- **Test** — as in bench1b: one-sided paired sign-flip permutation test over questions, 10 000 permutations, seed 0. Significant means Holm p < 0.05. Plus a 95 % bootstrap, wins and losses.
- **A tie** (an insignificant difference) means "without the module": the module is not switched on.
- **Implementation check** — the invariant "no change — ranking = base" is checked for C and D on every question. If violated, the run is invalid.

## 5. The "continue or not" criterion (fixed before the run)

- The criterion is about the **graph seeded from the question** (the graph's starts are only the question and the base ranking).
- **Continue:** C is significantly better than A on at least one set (S1 on IMLogic or ImplexConv), **and** D is not worse than D0.
  - "Not worse" means that on both sets D is not significantly worse than D0: the reverse one-sided test of the same scheme (D0 > D), p > 0.05 uncorrected. This is stricter for us than with a correction.
- **Stop or change the approach:** the graph (S1 and S2) gives **no** significant gain over its base **anywhere**.
- **The intermediate case** (a gain, but not significant, or significant on only one set without "D not worse than D0"): the report says "inconclusive", with no conclusion that "it works".
- **A negative stage 1 does not close the hypothesis of sleeves from reasoning steps — it is tested in stage 2** (`reports/prereg_stage2.md`).

## 6. Dev expectation (honest)

**On dev the graph gives no gain.** The main hypothesis is tested on the main part, and the expected result may be zero.

| set | A, R@5 | C, R@5 | C − A; wins / losses |
|---|---|---|---|
| IMLogic dev, `memory_l` (code 590126d) | 6.0 | 6.0 | 0.0 [−1.8; +1.8]; 3 / 3 |
| ImplexConv dev (code 590126d) | 16.3 | 13.7 | −2.6 [−9.8; +4.6]; 2 / 4 |

Other variants on dev (Kaggle GPU, kernel `ctxgraph-implicit-run` v3–v5; D+ not computed — it needs an NVIDIA key and a "yes" for the calls):

| set, R@5 (R@10) | A | B | C | D0 | D | D − D0; wins / losses |
|---|---|---|---|---|---|---|
| IMLogic dev, `memory_l`, 285 questions | 6.0 (11.2) | 2.8 (6.3) | 6.0 (11.6) | **11.6** (18.6) | 11.6 (17.5) | 0.0 [−1.8; +2.1]; 4 / 4 |
| ImplexConv dev, 45 questions | **16.3** (35.7) | 8.0 (14.1) | 13.7 (34.6) | 11.7 (22.9) | 16.2 (27.3) | +4.5 [−1.6; +12.0]; 4 / 3 |

- On IMLogic, the EverMemOS stack without an LLM is twice as good as dense on `memory_l` (11.6 vs 6.0). The graph on top of either changes nothing.
- **On ImplexConv dev C − A = −2.6 (2 wins / 4 losses): the graph may violate the "not worse" principle; on 45 questions this is not a conclusion.**
- On ImplexConv the stack is worse than dense (11.7 vs 16.3); the graph on top of the stack brings it up to the dense level (+4.5, not significant, only 45 questions).
- The dev numbers were recorded after debugging; **parameters did not change** (only technical failures were fixed: the time window without timestamps, the reranker running out of GPU memory).

## 7. Postponed to "later"

Answer accuracy (A and the best variant, the IMLogic protocol) and D+ are outside stage 1: a separate pre-registration, and only after Egor's "yes" for LLM calls.

## 8. Time (reported for each variant)

- **Memory construction:** seconds per sentence and in total.
  - A, C, B: e5, spaCy, graph. On Kaggle GPU in bench1b — 8–9 ms per sentence. The "0.6 s per sentence" alarm was a misreading of the log: see `fix1_changes.md`, §3.
  - D0, D: Qwen3-4B embeddings, BM25, scene clustering.
- **Search:** seconds per question for each variant.
- If writing to the graph turns out to be of the order of 0.5 s per sentence or more, this is noted as a limit on practical applicability, regardless of Recall.

## 9. Run on Kaggle

- **The kernel (A, B, C, D0, D) makes no API calls.** Internet is on only to download weights and data at pinned revisions (e5, Qwen3-Embedding-4B, Qwen3-Reranker-4B, spaCy, ImplexConv), as in bench1b; there are no keys in the kernel. (Changed at the freeze: it used to say "no internet" — that was needed for D+, which is now postponed.)
  - The Qwen3-4B models are loaded **one at a time**: first all embeddings, then unloading, then the reranker. If memory runs out — 8-bit.
- **Checkpoints per variant and per user (example).** Each variant's ranking is written to a separate file `/kaggle/working/<set>/<variant>.jsonl`, and a rerun skips what is done.
- **Time on dev (Kaggle GPU):**
  - graph (e5, spaCy) — about 20 ms per sentence;
  - Qwen embeddings — about 12 ms per fact;
  - D0 search — about 7 s per question on IMLogic and about 41 s on ImplexConv (reranking long sessions).
- **Hence two separate kernels, one per set.** IMLogic — about 4 hours, ImplexConv — about 10 hours; the Kaggle limit is 12 hours per session. If interrupted — continue from the checkpoint, noted in the report.
- **One main run** after the freeze. Technical amendments only before viewing results, labelled.

## Amendment 1 (2026-11-07, technical, before any main-part results)

- The first launch of the IMLogic main part (kernel v6) crashed at the start of the graph stage, before any rankings.
  - Reason: HaluMem has malformed timestamps like "Aug 12, 2040, 2040, 13:33:11" (the year repeated).
  - Fix: parsing takes the month, day, first year and time. Timestamps are needed only for the scene window in D0.
- In the same code version the gate gained an optional `extra_anchors` parameter for stage 2. Without it the behaviour is identical; this is checked by the gate tests.
- Everything else unchanged.

## Amendment 2 (2026-11-08, technical, before viewing any ImplexConv results)

- The ImplexConv main run (kernel v8) crashed at the embedding stage, about an hour after the start. A, B and C had been computed by then but **were not viewed**, and the partial output was deleted.
- Reason: 5 opposed examples (`opposed52`, `opposed295`, `opposed483`, `opposed894`, `opposed1272`) were left with no questions after question filtering, and stacking the embeddings of an empty list crashes.
- Fix: an empty array is written for such examples. Questions and metrics are not affected: there are still 855.
- ImplexConv is rerun in full, with a new kernel version. Everything else unchanged.
- **Clarification to amendment 2 (2026-11-08):** the first fix was wrong: a `reshape` of an empty array. The v9 rerun crashed at the same stage, also without results being viewed. Fixed: for an example with no questions, an array of the right shape (0 × dimension) is written; checked locally on an example with 0 questions. Rerun — v10. In addition, the kernel now deletes intermediate files on failure too.

## Amendment 3 (2026-11-08, technical, before viewing any ImplexConv results)

- The ImplexConv v10 rerun passed the amendment-2 fix but crashed at the embedding stage, on example 195 of 200: GPU memory ran out for a batch with very long sentences. Results were not viewed; the kernel deleted the intermediate files.
- Fix: when memory runs out, the batch is encoded again in halves, down to one sentence. The model, truncation and parameters are the same; checked locally.
- Rerun — v11. Everything else unchanged.
- **Clarification to amendment 3 (2026-11-08):** v11 crashed at the same place, also without results being viewed.
  - Reason: `supportive774` contains a "sentence" of 13 260 characters (sentence splitting failed on it). The retry with a smaller batch ran inside the exception handler, which keeps the previous attempt's memory; so free memory shrank with every retry.
  - Fixed: the retry runs after leaving the handler, with memory freed. The logic was checked locally.
  - In addition, examples with no questions (amendment 2) are skipped at every stage — the questions and metrics are the same.
  - On failure the kernel now saves checkpoints to its output, and the `ctxgraph-implicit-resume` kernel continues from them.
  - Rerun — v12.

> *Translator's note (not part of the frozen document):* v12 has not been run — the weekly Kaggle GPU quota was used up. So stage 1 has results for IMLogic only (`reports/implicit/main/`); there is no stage-1 result report for ImplexConv.
