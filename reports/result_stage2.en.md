> **Translation of the frozen document `reports/result_stage2.md` at commit 1ec39ea (private development repository; the text is identical to `reports/result_stage2.md` here); the Russian original is authoritative.**

# Stage 2, main part: result (2026-11-09)

> Under the pre-registration `reports/prereg_stage2.md` (frozen 2026-11-08) with amendment 1. One analysis run, kernel `ctxgraph-stage2-cpu` (Kaggle CPU), 2026-11-09; tables — `reports/implicit/stage2_main/stage2_report.md`.

## Scope (amendment 1 — NVIDIA unavailability)

- **ImplexConv:** the first 300 of 855 questions in the seed-0 order (83 opposed, 217 supportive).
- **IMLogic:** 3 whole users of 18 — 397 of 1 931 questions. This is a small part of the set; the IMLogic conclusions are weaker.
- Planner steps: Mistral Large 4 via NVIDIA, the model was not changed. These questions have 10 empty plans: 8 for ImplexConv and 2 for IMLogic.

## Main hypothesis H1: SC (search over the whole text of the steps) vs A, R@5, Holm over 2

| set | A | SC | SC − A [95 %] | wins / losses | p (Holm) |
|---|---|---|---|---|---|
| ImplexConv | 25.7 | 31.0 | **+5.3** [+2.1; +8.6] | 45 / 24 | **0.0022** |
| IMLogic | 6.0 | 6.8 | +0.8 [−1.3; +2.8] | 10 / 7 | 0.32 |

**Conclusion under the §4 criterion: search over plan steps is useful.** H1 is significant on ImplexConv, and the reverse test is significant nowhere. This matches the dev expectation: a gain on ImplexConv, about zero on IMLogic.
- Descriptive, R@5 by ImplexConv type:
  - opposed — A 9.6 → SC 18.1;
  - supportive — A 31.8 → SC 35.9.

## Secondary (Holm over 4)

- **H2, SR − SC: not supported.** Splitting into sleeves gave no gain over a search over the whole text of the steps.
  - ImplexConv: −4.8 [−8.4; −1.2], 21 / 47.
  - IMLogic: −2.8 [−5.0; −0.8], 4 / 15.
  - Descriptive, uncorrected and outside the criterion: the reverse test (SC > SR) is significant on both sets (p = 0.005 and 0.011). Sleeves are worse than one search over all the steps.
- **H3, SR − A: not supported.**
  - ImplexConv: +0.5, 36 / 42.
  - IMLogic: −2.0, 8 / 16.
- **SG (descriptive, IMLogic only):**
  - SG − A: +1.0 [−0.8; +2.8], 8 / 4;
  - SG − SR: +3.0, 17 / 5.
  - SG on ImplexConv is postponed because of the time needed to build the graph on CPU.

## Caveats

- The analysis was run exactly once — in the step of the kernel chain where 300 ImplexConv questions had been collected.
- D0 variants were not computed. They are secondary and wait for the GPU quota to reset.

## Addendum 2026-11-12 (after the freeze; the result does not change): IMLogic attribution and licence

*(Translation of the addendum in the Russian original.)*

- IMLogic benchmark: Ding et al., *Towards Root Memory: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs*, arXiv 2606.23283v2; data and code — https://github.com/ThisIsCosine/IMLogic.
- Licence: data CC BY-NC-ND 4.0 (from HaluMem), code MIT. Summary Recall is published with the authors' permission (received 2026-11-12) for non-commercial research, with attribution. The data and question texts are not published.
- Data version: the copy downloaded on 2026-10-30 is identical to the v2 repository (commit `8823def`) apart from line endings; SHA-256 (LF): `HaluMem-Medium.jsonl` 39d284f936c5130a…, the 20 `qa/qa_userNN.json` files concatenated bde5fc6027e03346….
