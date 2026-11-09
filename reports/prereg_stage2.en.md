> **Translation of the frozen document `reports/prereg_stage2.md` at commit 973706d (private development repository; the text is identical to `reports/prereg_stage2.md` here); the Russian original is authoritative.**

# Stage 2 pre-registration: search over plan steps (planner) — IMLogic memory-level and ImplexConv-200

> **Status: FROZEN 2026-11-08 by Egor's decision ("I accept the draft, freeze it"), before any main-part output.** Debugging was on dev only (IMLogic dev-285, ImplexConv dev-45).
> Code: stage 1 + the gate with extra starts (`ctxgraph/v2/gate.py::ppr_gated(extra_anchors=…)`; without steps its behaviour is identical to stage 1) + `scripts/planner.py` + the `sleeves` and `sc` stages in `scripts/implicit_run.py`.
> Stage 2 does not depend on the outcome of stage 1: a negative stage 1 does not close the sleeves hypothesis (`prereg_implicit.md`, §5).

## 1. Hypothesis

The decisive fact is far from the question but close to a **reasoning step** (for example, "herniated disc" is close to the step "the user's health and constraints").

**Main hypothesis:** a search over the whole text of the plan steps, fused with the base (SC), finds the decisive fact better than base A.

**Secondary:**
- splitting the steps into sleeves (SR) adds something to the search over the whole text of the steps (SC). The dev expectation is about zero;
- SR beats base A.

The graph on top of sleeves (SG) is a variant, descriptive only. D0 variants come after the GPU quota resets.

**The hypothesis was changed before the freeze, based on dev and openly** (Egor's decision, 2026-11-08). Before: the main comparison was SR vs SC (the contribution of splitting into sleeves). On dev, splitting into sleeves gave no gain over searching the whole step (§6), so the comparison of SC with the base became the main one.

## 2. Planner

- **One LLM call per question, at most 3 steps.** A step is a short search query about a past fact of the user: a goal, constraint, preference, plan, condition or decision.
- **Prompt** — `scripts/planner.py::PROMPT`, with no examples from the data.
- **Model:** the free NVIDIA API, `mistralai/mistral-large-4-1t-a52b`, temperature 0, `max_tokens` = 3000. The model first "thinks" in hidden text: with a 300-token limit, all 20 smoke-test answers were empty.
- **Cache** — by request hash: a rerun makes no calls.
- **An unparsable answer** means "no steps": for such a question SR = SC = SG = base. The share of such answers is reported. On dev — 1 of 330.

## 3. Variants (all on top of the base; RRF k = 60)

| | variant | what it is |
|---|---|---|
| S0 | base | the base ranking for the question |
| **SR** | **sleeves** | RRF(base, base ranking for step 1, …, for step n) — one search per step |
| **SC** | **control: steps in one search** | RRF(base, base ranking for the concatenated text of all steps). Same steps, same fusion; the only difference from SR is one search instead of one per step |
| SG | sleeves + graph (variant) | base + the **old** stage-1 gate (no reranker check). The steps' concepts and the base's top result for each step are added to the starts. The rules are unchanged: intersection of ≥ 2 starts, PPR margin × 4, at most 2 displacements, add or raise only |

**The base of the main comparison is A (dense e5) on both sets** (Egor's decision, 2026-11-08): the Kaggle GPU quota is used up until it resets, and variants over A are computed without the Qwen GPU models.

**D0 variants are secondary, after the quota resets:** IMLogic only, where D0 is the strongest base (11.6 vs 6.0 for A). D0 is not computed on ImplexConv: base A is stronger there (16.3 vs 11.7).

## 4. Comparisons (Recall@5 against the gold: `memory_l` / evidence sessions)

The test is the same as in bench1b and stage 1: one-sided paired sign-flip permutation test, 10 000 permutations, seed 0; 95 % bootstrap; wins and losses.

| | comparison | role | correction |
|---|---|---|---|
| **H1** | **SC − A** | main: searching the whole text of the steps beats the base | Holm over 2 (two sets) |
| H2 | SR − SC | secondary: contribution of splitting into sleeves; dev expectation ≈ 0 | Holm over 4 (H2 and H3 × two sets) |
| H3 | SR − A | secondary: sleeves beat the base | same |
| — | SG − A, SG − SR | descriptive | — |
| — | S0 / SR / SC / SG over D0, IMLogic only | secondary, after the GPU quota resets; Holm separately, over 3 comparisons (SC − D0, SR − SC, SR − D0) | — |

All comparisons are over A; S0 = A.

**Criterion:**
- **Search over plan steps is useful:** H1 is significant on at least one set, **and** on the other set the reverse test (A > SC) is not significant (p > 0.05 uncorrected).
- **Not worse:** if H1 is significant nowhere, and the reverse test is also significant nowhere — "no gain", with no conclusion that "it works".
- **Harmful:** the reverse test is significant on at least one set.
- **Splitting into sleeves (H2):** the conclusion "sleeves are useful" only if H2 is significant. Otherwise the report says: "splitting into sleeves gave no gain over a search over the whole text of the steps".
- **The graph (SG)** is descriptive only; it does not go into the product on the basis of these data.

## 5. Cost of the main part

**LLM calls (planner):**

| | questions | calls |
|---|---|---|
| IMLogic main | 1 931 | 1 931 |
| ImplexConv main | 855 | 855 |
| **total** | **2 786** | **2 786 answers** — Egor's "yes" 2026-11-08. Only requests cut off by network errors are retried (≤ 10 attempts; they return no answer); if such errors exceed 5 %, the run stops. Per-question cache: after an interruption the run continues with no new calls |

- Tokens: about 150 in, 800–1 500 out (mostly hidden reasoning) — about 3–4 million in total.
- Time: 8 threads give about 5–6 calls a minute, about 8–9 hours in total (locally).
- Dev used **355 calls**: 20 for the smoke test with a 300-token limit, 5 with a 1 500 limit, 330 for dev itself. That is 25 more than the approved ≈ 330: the extra went into choosing `max_tokens`. Debugging rrgate and SC made no calls: the steps came from the same plans.

**Rankings over A (main, without Kaggle; Egor's decision — locally on CPU, without Modal):**
- Needed: e5 embeddings of all sentences and the graph (the graph is for SG): IMLogic main — 13 521 sentences, ImplexConv main — **696 177**. Search A / SR / SC / SG — fractions of a second per question.
- **Here, on CPU:** e5 — 33 sentences per second (measured on 1 024 ImplexConv sentences). ImplexConv embeddings — about 6 hours, plus spaCy and graph writing — about 1–2 hours; **about 8 hours in total**. IMLogic — about 15 minutes.
- **On Modal (T4 or L4):** about an hour for both sets: embeddings take minutes, the rest is spaCy and graph writing on CPU. Cost — about $1–2.
- **Checkpoints** are saved per corpus (on Modal — in a Volume), so a failure resumes rather than restarts.

**Rankings over D0 (secondary, IMLogic, after the Kaggle quota resets):** up to 5 D0 searches per question at ≈ 5.7 s each — about 15 GPU hours; two kernels of 9 users each.

## 6. Dev expectation (honest; dev is small, this is not a conclusion)

R@5; wins / losses against the base in brackets:

| set / base | S0 | SR | SC | SG |
|---|---|---|---|---|
| IMLogic dev-285, **A** (main) | 6.0 | 7.4 (9 / 5) | 6.7 (8 / 6) | 6.7 (7 / 5) |
| ImplexConv dev-45, **A** (main) | 16.3 | 28.9 (10 / 3) | 27.1 (10 / 3) | 23.7 (6 / 2) |
| IMLogic dev-285, D0 (secondary) | 11.6 | 9.8 (13 / 18) | 12.3 (11 / 9) | 13.0 (12 / 8) |
| ImplexConv dev-45, D0 (not computed on main) | 11.7 | 20.3 (9 / 2) | 20.5 (6 / 1) | 15.3 (4 / 1) |

Differences, R@5 [95 % bootstrap], wins / losses, one-sided test p (dev, uncorrected):

| | IMLogic, A | ImplexConv, A | IMLogic, D0 | ImplexConv, D0 |
|---|---|---|---|---|
| **SC − base** (H1) | +0.7 [−1.8; +3.2]; 8 / 6; p = 0.39 | **+10.8** [−0.3; +22.3]; 10 / 3; p = 0.036 | +0.7 [−2.5; +3.9]; 11 / 9; p = 0.41 | +8.8 [+1.8; +17.0]; 6 / 1; p = 0.022 |
| SR − SC (H2) | +0.7 [−1.8; +3.2]; 8 / 6; p = 0.40 | +1.9 [−9.8; +14.1]; 6 / 6; p = 0.40 | −2.5 [−6.0; +0.7]; 8 / 15; p = 0.95 | −0.2 [−9.5; +8.3]; 5 / 4; p = 0.55 |
| SR − base (H3) | +1.4 [−1.1; +3.9]; 9 / 5; p = 0.21 | **+12.7** [+3.7; +23.1]; 10 / 3; p = 0.009 | −1.8 [−5.6; +2.1]; 13 / 18; p = 0.86 | +8.6 [−0.6; +18.3]; 9 / 2; p = 0.045 |
| SG − SR (descriptive) | −0.7 [−3.2; +1.8]; 6 / 8; p = 0.79 | −5.3 [−16.7; +6.0]; 6 / 9; p = 0.82 | +3.2 [−0.4; +7.0]; 19 / 10; p = 0.068 | −5.0 [−15.6; +5.6]; 5 / 10; p = 0.81 |

- **The main point on dev: SR ≈ SC.** Splitting the plan into sleeves adds almost nothing to one search over the whole text of the same steps: +0.7 and +1.9, both not significant. Over D0, sleeves are even worse than SC: −2.5, 8 / 15.
- **The steps themselves help:** on ImplexConv both SR and SC are much better than the base — +12.7 and +10.8. On IMLogic over A the gain is small and not significant.
- **The graph on top of sleeves** loses to SR over A on both sets; over D0 on IMLogic it beats SR (+3.2, p = 0.068).
- **Dev expectation for the main part: a gain of SC over A on ImplexConv and about zero on IMLogic.** On dev: ImplexConv +10.8 (p = 0.036, 45 questions), IMLogic +0.7 (not significant). SR − SC ≈ 0.
- **On dev, splitting into sleeves gave no gain over searching the whole step; the main hypothesis was changed before the freeze for this reason.**
- Sources: `reports/implicit/stage2_dev_v1/` (S0, SR, SG) and the `ctxgraph-stage2-sc` kernel (SC). A matches in its top 5 on all questions in both kernels.

## 7. Failed approach: the gate with a reranker check (rrgate) — recorded as a finding

Tested on dev (kernel `ctxgraph-stage2-rrgate`, replay `scripts/rrgate_sweep.py`; `reports/implicit/stage2_rrgate/`).

**Rule.** A graph candidate displaces a base entry only if its Qwen3-Reranker-4B score (probability of "yes" for the question) is no lower than that of the displaced entry minus δ.

**Result.**
- At δ ≤ 0 the check blocks **all** graph insertions on both sets: D and SG equal D0.
- At δ = 1 the check never fires; this is the old gate.
- At intermediate δ (0.05–0.5), a handful of candidates pass, with no stable effect: on IMLogic, D with the new gate ranges from 1 / 2 to 3 / 3.

**The reason is structural.**
- The reranker scores similarity to the question, and the decisive memory is dissimilar to the question.
- The entries in positions 4–5 of the D0 ranking were already selected by the same reranker. A graph candidate that the reranker would score higher would usually already be in the ranking.
- So this check does not separate useful graph insertions from harmful ones.

**Egor's decision:** δ is not tuned; rrgate is not part of stage 2; SG uses the old gate, as a variant.

The replay was verified: at δ = 1 it reproduces stage-1 D on dev (IMLogic 11.6, 4 / 4; ImplexConv 16.2, 4 / 3) and SG_D0 of the debug kernel (13.0, 12 / 8).

## 8. Run

- Plans for the main part — locally, with the same `planner.py`, after the "yes".
- Rankings — on Kaggle GPU: stages graph → embed → sleeves + sc. No API calls in the kernel; plans are passed as a file.
- One main run after the freeze. Technical amendments only before results are viewed, labelled as such.

## Amendment 1 (2026-11-08, before any main-part output was viewed; Egor's decision)

**Reason.** At about 16:00 Moscow time, the free NVIDIA API for the planner model (`mistralai/mistral-large-4-1t-a52b`) almost stopped answering: requests time out after 240 s. By 17:50, 446 of 2 786 answers had been received, all for IMLogic. The model and the provider **do not change**.

**What changes:**
1. **Order.** ImplexConv first, then the rest of IMLogic.
   - ImplexConv questions are taken in the order of `random.Random(0).shuffle` over the sorted ids of all 855 main questions.
   - This makes any prefix contain opposed and supportive questions in the overall proportion. In loader order, all 255 opposed questions come first.
2. **Minimum for the analysis:**
   - **ImplexConv** — the first **300** questions of this order; the analysis uses the longest continuous prefix with received answers, but no shorter than 300.
   - **IMLogic** — **whole users only**: every question of the user has an answer. Partially processed users are excluded.
   - If all answers are received before the end of the run, the analysis uses all 2 786 questions, as in §4.
3. **A failed request** (all retries cut off) is not "no steps": such a question has no answer and is left out of the analysis. "No steps" applies only to a received but unparsable answer (§2).
4. **Where it runs.** The planner, search A / SC / SR / SG and the analysis run in a Kaggle kernel **on CPU** (no GPU, without my computer):
   - the NVIDIA key comes from Kaggle Secrets;
   - answers are written to the kernel output as they arrive;
   - at the 12-hour limit, the next kernel continues from this cache.
   - The ImplexConv e5 embeddings and the IMLogic graph were computed locally on CPU (their identity was checked on dev) and passed to the kernel as a dataset.
5. **SG on ImplexConv** is postponed: building the graph on CPU takes too long (outside the hypotheses, §4).
6. **Tests and the Holm correction** — as in §4, on the questions included in the analysis. The number of questions per set is reported.
