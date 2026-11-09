> **Translation of the frozen document `reports/result_bench1b.md` at commit 17c0871 (private development repository; the text is identical to `reports/result_bench1b.md` here); the Russian original is authoritative.**

# Benchmark 1b result — second attempt; tuned on MuSiQue dev; both attempts shown

> Pre-registration — `reports/prereg_bench1b.md` (English: `reports/prereg_bench1b.en.md`). Frozen 2026-10-31 by Egor's "approved" (commit d707ea0, in the private development repository), before the run.
> Run — Kaggle GPU, private kernel `egor123hhh/ctxgraph-bench1b-run`: version 2 — code d707ea0; version 3 — a rerun on code 590126d under amendment 1. LLM calls — 0.
> Raw data — `reports/bench1b/test/` (`*_summary.json`, `*_recall.jsonl`). Examples — `reports/bench1b/test_examples.json`.
> First attempt — `reports/result_bench1.md`. There, graph + PPR was worse than dense, and that conclusion stands.

Variant labels: (0) dense, (K) dense + reranking, (G) (0) + graph gate, (KG) (K) + graph gate; the original writes (К), (Ш), (КШ).

## Code version 590126d — SECONDARY: a change after viewing (rerun, kernel v3)

> **Reason for the rerun.** A bug in the extraction of English possessive phrases, found on conversational data, was fixed: `reports/implicit/fix1_changes.md`. The data, hypotheses, test and gate configuration did not change. One run.
> Raw data — `reports/bench1b/test_590126d/`.
> **A change after viewing.** The bug was found on 2026-11-01 on IMLogic dev (commit 590126d, 18:59) — **after** the d707ea0 test result had been obtained and recorded (commit 1ea2949). So **the primary result is the original frozen version d707ea0** (H2 not supported, p = 0.067), and 590126d is secondary. Egor's decision of 2026-11-05.

| | MuSiQue: d707ea0 | MuSiQue: **590126d** | 2Wiki: d707ea0 | 2Wiki: **590126d** |
|---|---|---|---|---|
| graph: concepts / edges | 116 030 / 640 218 | 112 773 / 634 062 | 60 876 / 304 636 | 59 388 / 302 212 |
| R@5: (0) / (K) | 51.4 / 59.7 | 51.4 / 59.7 | 70.3 / 69.4 | 70.3 / 69.4 |
| R@5: (G) / (KG) | 53.9 / 60.3 | 53.9 / **60.9** | 73.9 / 73.6 | 74.1 / 73.3 |
| **H1** (G) − (0) | +2.5 [+1.4; +3.5], Holm p 0.0002; 129 / 64 | **+2.4 [+1.3; +3.5]**, Holm p 0.0002; 138 / 70 | +3.5; 112 / 16 | +3.8 [+2.9; +4.7]; 109 / 10 |
| **H2** (KG) − (K) | +0.7 [−0.2; +1.6], Holm p 0.067 — **not supported** | **+1.3 [+0.3; +2.1]**, Holm p **0.0042 — supported** | +4.2; 123 / 14 | +3.9 [+3.0; +4.9]; 118 / 15 |
| invariant: unchanged = base | 238 / 238, 300 / 300 | 230 / 230, 298 / 298 | 238 / 238, 204 / 204 | 261 / 261, 211 / 211 |

- **H1** holds on both versions, with almost the same difference.
- **H2 diverges:**
  - on d707ea0 it is not supported;
  - on 590126d it is supported: +1.3, Holm p 0.0042, 88 wins / 60 losses.
  - This is the result of a change after viewing, **not a confirmation of H2**. The primary result remains d707ea0: H2 not supported.
- Timing of 590126d (MuSiQue): writing the graph — 374 s (7.7 ms per sentence), the gate — 0.71 s per question.

## Summary of version d707ea0 — PRIMARY (the original frozen version)

- **H1 supported.** On MuSiQue the graph with the gate beats dense: R@5 **+2.5 pp** [+1.4; +3.5], Holm p = 0.0002.
  - On changed questions — 129 wins and 64 losses (counted in supporting passages).
- **H2 not supported.** On top of reranking on MuSiQue: **+0.7** [−0.2; +1.6], Holm p = 0.067.
  - 73 wins and 60 losses.
  - The dev expectation was about −1.5.
- **The invariant held on every question.** Where the gate changed nothing, the ranking is identical to the base:
  - MuSiQue: 238 of 238 for (G) and 300 of 300 for (KG);
  - 2Wiki: 238 of 238 and 204 of 204.
- **2Wiki** (descriptive; the gate was not tuned there): (G) − (0) = **+3.5** [+2.6; +4.5], (KG) − (K) = **+4.2** [+3.2; +5.1].
  - Wins and losses: 112 / 16 and 123 / 14.
  - Here the graph adds noticeably even on top of reranking, and this is transfer without tuning.

## Both attempts, passage Recall@5, %

| attempt | variant | MuSiQue | 2Wiki |
|---|---|---|---|
| 1 (benchmark 1) | (a) dense e5 | 51.4 | 70.3 |
| 1 | (b) dense + BM25 + RRF | 49.8 | 68.3 |
| 1 | (c) graph v2 + PPR | 44.2 | 66.1 |
| 1 | (d) (c) + reranking | 60.3 | 70.4 |
| **2 (benchmark 1b)** | (0) dense e5 (= (a)) | 51.4 | 70.3 |
| 2 | (K) dense + reranking, no graph | 59.7 | 69.4 |
| 2 | **(G) (0) + graph gate** | **53.9** | **73.9** |
| 2 | **(KG) (K) + graph gate** | **60.3** | **73.6** |
| | *HippoRAG 2 (paper)* | *74.7* | *90.4* |
| | *NV-Embed-v2 (paper)* | *69.7* | *76.5* |

- **The main difference of the first attempt** ((c) − (a)): −7.2 [−9.3; −5.2] on MuSiQue and −4.2 on 2Wiki.
- **The main difference of the second attempt** ((G) − (0)): **+2.5 [+1.4; +3.5]** on MuSiQue and +3.5 on 2Wiki.
- In the second attempt the graph was built **by the same code and matched down to the edge**: 48 415 sentences, 116 030 concepts, 640 218 edges. Only the way the graph changes the ranking changed:
  - the PPR start was fixed;
  - the gate was added — intersection of starts and the PPR margin × 4.
- The gate's Recall@2 equals the base (39.0 / 46.0 and 62.5 / 63.4). The gate changes only positions 4–5 — as designed.

## Hypotheses (MuSiQue, Recall@5, Holm correction over two)

| | comparison | difference, pp [95 %] | p | Holm p | changed questions | wins / losses | verdict |
|---|---|---|---|---|---|---|---|
| H1 | (G) − (0) | **+2.5 [+1.4; +3.5]** | 0.0001 | **0.0002** | 762 | 129 / 64 | **supported** |
| H2 | (KG) − (K) | +0.7 [−0.2; +1.6] | 0.067 | 0.067 | 700 | 73 / 60 | not supported |

**Descriptive, MuSiQue:**
- (K) − (0) = +8.2 [+6.7; +9.7];
- (G) − (K) = −5.8;
- (KG) − (0) = +8.9 [+7.3; +10.5].

Reranking still gives three times more than the gate. On MuSiQue the gate adds almost nothing on top of reranking.

**2Wiki, descriptive:**
- (G) − (0) = +3.5 [+2.6; +4.5];
- (KG) − (K) = +4.2 [+3.2; +5.1];
- (K) − (0) = −0.9 [−1.9; +0.1];
- (G) − (K) = +4.5 [+3.2; +5.8].

On 2Wiki the graph with the gate beats reranking.

## By question type, Recall@5 (descriptive)

| type | n | (0) | (K) | (G) | (KG) |
|---|---|---|---|---|---|
| MuSiQue 2hop | 518 | 61.8 | 66.4 | 64.1 | 67.9 |
| MuSiQue 3hop1 | 243 | 44.2 | 58.6 | 47.5 | 58.7 |
| MuSiQue 3hop2 | 73 | 53.9 | 67.6 | 55.3 | 65.8 |
| MuSiQue 4hop1 | 108 | 27.1 | 33.3 | 28.5 | 32.6 |
| MuSiQue 4hop2 | 27 | 28.7 | 38.9 | 34.3 | 39.8 |
| MuSiQue 4hop3 | 31 | 34.7 | 46.8 | 36.3 | 49.2 |
| 2Wiki bridge_comparison | 235 | 51.3 | 53.8 | 56.6 | **59.4** |
| 2Wiki comparison | 244 | 99.6 | 99.8 | 99.4 | 99.8 |
| 2Wiki compositional | 413 | 63.6 | 61.7 | **68.9** | 67.6 |
| 2Wiki inference | 108 | 71.3 | 63.9 | **72.7** | 68.1 |

- The graph helps where a bridge between passages is needed: compositional and bridge-comparison questions on 2Wiki, 2hop on MuSiQue.
- On 2Wiki comparison questions the graph no longer hurts: 99.4 vs 90.2 in the first attempt. The gate does not touch the top three.

## Examples (the pre-registration rule: the first three MuSiQue questions each way, (G) vs (0))

**The graph found a supporting passage that dense did not have in its top five:**
1. "What month did the Tripartite discussions begin between Britain, France, and the country where … the top-ranking Warsaw Pact operatives originated?" → "Szlachta" added.
   - Intersection: the third dense passage and the question concepts "country" and "nobility commonwealth".
   - The supporting "Warsaw Pact" was displaced, so this question also has a loss.
2. "What county is Erik Hort's birthplace a part of?" → "Montebello, New York" added. It is linked to two dense passages from the top three.
3. "What is the Till dom ensamma performer's birth date?" → "Mauro Scocco" added. It is linked to three passages from the top three.

**The graph displaced a supporting dense passage:**
1. A question about a president and the ratification of the Treaty of Versailles → "Cabinet of the United States" displaced, "Felipe Berriozábal" added. Intersection through the shared concept "member".
2. A question about the party that sought to punish the southern states → a passage about the 2010 House elections displaced, "North Carolina" added (question concept "southern state").
3. A question about a majority religion → "Hindus" displaced, "Szlachta" and "Theodor Lipps" added. Intersections through the common question words "majority religion" and "meaning".

Losses come through common, uninformative question concepts. This is the expected price of the IDF variant with no threshold: there is a signal almost always — in 999 of 1 000 MuSiQue questions.

## Time (Kaggle GPU)

| | MuSiQue | 2Wiki |
|---|---|---|
| embeddings / writing the graph / synonyms | 116 / 427 / 102 s | 71 / 205 / 28 s |
| reranking, 1 000 questions | 875 s | 712 s |
| gate with PPR, 1 000 questions | 752 s (0.75 s/question) | 333 s |

## What this means and what is next

- **The "not worse than dense" principle now holds.** The graph with the gate is statistically significantly better than dense on MuSiQue and descriptively better on 2Wiki.
- **The "not worse than reranking" principle:**
  - on 2Wiki it holds descriptively;
  - on MuSiQue H2 is not supported (+0.7 [−0.2; +1.6], Holm p = 0.067).
- **The next stage's fix** is already written into the pre-registration. A graph candidate displaces the reranker's choice only if the candidate's reranker score is not much lower. This targets exactly the losses through common concepts in the examples. (Later tested on dev in stage 2 as "rrgate" and recorded as a failed approach: `reports/prereg_stage2.en.md`, §7. This note is the translator's, not part of the original.)
- The gap to HippoRAG 2 (74.7 / 90.4) remains large. Most of it lies in the encoder (e5-base vs NV-Embed-v2) and the LLM triples, not in the gate.
