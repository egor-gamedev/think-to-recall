# think-to-recall: summary for an outside reader (2026-11-09)

A research prototype of graph memory for LLMs, built without a generative model.

**The rule for all of this work:**
- before a run — a pre-registration of the hypotheses, metrics and test;
- parameters are tuned on dev only;
- a conclusion comes from one frozen run;
- changes made after seeing results are labelled, and the original result stays primary.

Below are all the hypotheses that were tested, failures included. Detailed pre-registrations and reports in `reports/` are in Russian.

## 1. Hypotheses and what happened to them

| # | hypothesis | where tested | result | verdict |
|---|---|---|---|---|
| 1 | A "menu" of related concepts from the graph, placed in the context, helps the model account for hidden requirements | E1, 30 tasks, DeepSeek V4.1 Flash, GPT-5.4 Mini judge | −4.8 pp vs no graph (Holm p = 0.09). Ceiling at 97 %; the model almost never called `expand` | **not supported** |
| 2 | Graph memory beats ordinary RAG in a "long project" | E1b, 54 questions, Qwen2.5-7B | the hybrid was best by direction; its gap over RAG not significant | **not supported** |
| 3 | The model choosing its own memory beats automatic expansion | E1c, 200 questions, Nemotron-3 Ultra | 12 % vs 18 % (p = 0.081), more fabrications | **not supported** (worse by direction) |
| 4 | "Sleeves" from the question give more than a search over the whole question | E1d, stage A | nothing at an equal pool size | **not supported** |
| 5 | Graph v2 + PPR instead of dense e5 (like HippoRAG 2, but without LLM triples) | bench1: MuSiQue / 2Wiki, 1 000 + 1 000 questions | −7.2 / −4.2 pp R@5 | **not supported** (worse) |
| 6 | The graph as an add-on to dense: a gate that "only adds or raises on a strong signal" | bench1b, H1, MuSiQue | **+2.5** [+1.4; +3.5], Holm p = 0.0002; 2Wiki +3.5 | **supported** |
| 7 | The same gate on top of a reranker | bench1b, H2, MuSiQue | +0.7 [−0.2; +1.6], p = 0.067 | **not supported** |
| 8 | A graph seeded from the question finds an implicit memory better than dense | stage 1, IMLogic, 1 931 questions | −0.5 [−1.1; +0.2] | **not supported** |
| 9 | The same on top of an EverMemOS-style search stack (D0) | stage 1, IMLogic | −0.9 [−1.6; −0.2]; reverse p = 0.008 | **not supported — the graph hurts** |
| 10 | A reranker as a filter on graph insertions (rrgate) | stage 2, dev | blocks everything at δ ≤ 0, nothing at δ = 1 | **failed approach** (a finding, not a parameter) |
| 11 | Searching over the whole text of the LLM's plan steps (SC) beats dense | stage 2, ImplexConv 300 + IMLogic 397 | **+5.3** [+2.1; +8.6], Holm p = 0.0022 on ImplexConv; +0.8 on IMLogic | **supported on ImplexConv** |
| 12 | Splitting the steps into separate searches ("sleeves", SR) beats SC | stage 2 | −4.8 and −2.8: sleeves are worse | **not supported** |
| 13 | SR beats dense | stage 2 | +0.5 / −2.0 | **not supported** |

**The main hypothesis was changed before the freeze.** Before stage 2 was frozen, its main hypothesis was no. 12. On dev, splitting into sleeves gave no gain over searching the whole step. So no. 11 became the main hypothesis — openly, before any main-part output (`reports/prereg_stage2.md`, §1, §6).

**In one sentence each.**
- A graph built without an LLM gives a small gain as a cautious add-on to dense retrieval on multi-hop factual questions.
- For implicit memories in conversation, the graph does not help.
- What helps is something else: a plan of steps written by the LLM, and a search over it.

## 2. Bugs found (and how they were handled)

| bug | where found | consequence | handling |
|---|---|---|---|
| Two disconnected implementations: the product graph was never benchmarked; all E1b–E1d numbers come from a separate memory rebuilt for every question | architecture review, 2026-10-30 | E1b–E1d conclusions do not apply to the product graph | rewritten as v2 (one implementation); E1b–E1c treated as a debug set |
| English possessives ("John's house") were merged into one concept | IMLogic dev, **after** the bench1b result | on bench1b, H2 moved from p = 0.067 to p = 0.004 | fixed (590126d); the rerun is **secondary**, the original d707ea0 stays primary |
| Malformed HaluMem timestamps ("2040, 2040") | first stage-1 run | crash before any results | amendment 1 to the pre-registration, technical |
| ImplexConv examples left with no questions after filtering; an OOM retry inside the exception handler | stage-1 GPU reruns | four crashes, ~9 lost GPU hours | amendments 2 and 3; results were not viewed |
| The planner returned empty answers: hidden reasoning used up the token limit | smoke test | 25 extra calls | limit raised to 3 000; the overrun of approved calls is reported |
| A script edited while spaCy worker processes were running | local run | crash on the 6th user | resumed from a checkpoint; rule: never edit running scripts |

## 3. Limitations

- **Partial stage-2 samples.**
  - ImplexConv — 300 of 855 questions: the first in a seed-0 order, with opposed and supportive in the overall proportion.
  - IMLogic — 3 of 18 users (397 of 1 931 questions).
  - Reason: the free NVIDIA API almost stopped answering (about 13 answers an hour). The size and selection rule were fixed by an amendment before the analysis.
  - The full ImplexConv sample, if it is ever completed, goes in only as a descriptive analysis.
- **Not computed:** stage 1 on ImplexConv and all variants over D0 in stage 2 — the weekly Kaggle GPU quota ran out.
- **D0 is not the full EverMemOS,** but its open retrieval stack (`everalgo`) without LLM memory construction. A comparison with the real EverMemOS is possible only if the IMLogic authors share the memory they built.
- **Embeddings are e5-base.** HippoRAG 2 / NV-Embed-v2 report higher absolute numbers (MuSiQue R@5: 74.7 / 69.7 vs our 51–60). Our conclusions are about differences against the same base.
- **Dev is small** (IMLogic 285, ImplexConv 45 questions). Dev expectations are a guide, not a conclusion.
- **One run per hypothesis,** with no repeats on other seeds. The planner is one model (Mistral Large 4 via NVIDIA), temperature 0.
- **IMLogic has no license** (the underlying HaluMem is CC BY-NC-ND 4.0) — internal research only. The repository holds only ids and numbers for it.

## 4. Where things are

- The summary table and one-command reproduction — `README.md`, `scripts/reproduce_tables.py`.
- Pre-registrations — `reports/prereg_bench1.md`, `prereg_bench1b.md`, `prereg_implicit.md`, `prereg_stage2.md`.
- Results — `reports/result_bench1.md`, `result_bench1b.md`, `result_stage2.md`; stage 1 — `reports/implicit/main/`.
- The next step (not started) — context-dependent edge weights: `ARCHITECTURE_V2.md`, §3.16.
