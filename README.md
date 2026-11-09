# think-to-recall — graph memory for LLMs, built without a generative model

> The Python package inside this repository is called `ctxgraph`, the project's working name. Imports, scripts, Kaggle kernels and reports all use it. The main pre-registrations and results have English translations (`*.en.md`, linked below); the Russian originals are authoritative.

A research prototype. A graph is built from a user's conversations: episodes → sentences → concepts. Edges are co-occurrence and embedding synonyms. Building the graph makes **no LLM calls**. Retrieval is dense e5, with personalized PageRank over the graph on top, as in HippoRAG 2.

**The project's question:** does such a graph find the right memory better than plain dense retrieval? This matters most for a memory that is semantically far from the question — an *implicit* link.

Every conclusion below comes from **one frozen run under a pre-registration** (`reports/prereg_*.md`). Parameters were tuned on dev only. Changes made after seeing results are labelled, and the original result stays primary.

## Main results

Recall@5, difference in percentage points [95 % bootstrap]. p is a one-sided paired sign-flip permutation test (10 000 permutations, seed 0), Holm-corrected.

| experiment | comparison | result | verdict |
|---|---|---|---|
| bench1: MuSiQue / 2Wiki, 1 000 questions each | graph v2 + PPR − dense e5 | −7.2 [−9.3; −5.2] / −4.2 [−6.0; −2.5] | **not supported**: the graph instead of dense is worse |
| bench1b, MuSiQue, H1 | dense + graph gate − dense | **+2.5** [+1.4; +3.5], p = 0.0002 | **supported**; 2Wiki +3.5 (descriptive) |
| bench1b, MuSiQue, H2 | reranker + graph gate − reranker | +0.7 [−0.2; +1.6], p = 0.067 | **not supported** |
| stage 1: IMLogic, 1 931 questions | dense + graph gate − dense | −0.5 [−1.1; +0.2] | **not supported** |
| stage 1: IMLogic | graph gate over an EverMemOS-style search stack − the stack | −0.9 [−1.6; −0.2] | **significantly worse** (reverse test p = 0.008) |
| stage 2: ImplexConv, 300 questions, H1 | dense + search over the LLM's plan steps − dense | **+5.3** [+2.1; +8.6], p = 0.0022 | **supported** |
| stage 2: IMLogic, 397 questions, H1 | same | +0.8 [−1.3; +2.8], p = 0.32 | not supported |
| stage 2, H2 (both sets) | one search per step ("sleeves") − one search over all steps | −4.8 and −2.8 | **not supported**: sleeves are worse |

**Documents behind the table** (English translation · Russian original, which is authoritative):

| experiment | pre-registration | result |
|---|---|---|
| bench1b | [EN](reports/prereg_bench1b.en.md) · [RU](reports/prereg_bench1b.md) | [EN](reports/result_bench1b.en.md) · [RU](reports/result_bench1b.md) |
| stage 1 (implicit links) | [EN](reports/prereg_implicit.en.md) · [RU](reports/prereg_implicit.md) | [EN](reports/result_stage1.en.md) · [RU](reports/result_stage1.md) — IMLogic only; ImplexConv postponed (weekly GPU quota) |
| stage 2 (plan steps) | [EN](reports/prereg_stage2.en.md) · [RU](reports/prereg_stage2.md) | [EN](reports/result_stage2.en.md) · [RU](reports/result_stage2.md) |
| bench1 (first attempt) | [RU](reports/prereg_bench1.md) | [RU](reports/result_bench1.md) |

Earlier exploratory reports (E1–E1d, E3, architecture notes, debug reports) are in Russian. `docs/history/` holds the original task statements and research notes; the `reports/` of the early stages E1–E1c are a research archive.

**The gate** adds or raises a graph candidate in the dense top-5 only on a strong signal. Without a signal, its output is identical to the base ranking (an invariant checked on every question).

**The planner** is one LLM call per question (Mistral Large 4), which returns at most three short search queries about the user's past facts that could change the answer.

### What works

- **The graph as a cautious add-on to dense retrieval.** The gate gives a small gain on multi-hop factual questions (MuSiQue, 2Wiki).
- **Searching over the steps of a plan the LLM writes before answering.** This finds the implicit memory clearly better than searching by the question alone on ImplexConv. The graph is not needed for this.

### What does not work

- **The graph instead of dense** retrieval loses.
- **The graph on top of a strong reranker** adds nothing significant.
- **On conversational memory (IMLogic)**, a graph seeded from the question does not help. On top of an EverMemOS-style search stack it hurts.
- **Splitting the plan into separate "sleeves"** is worse than one search over the whole text of the steps.
- **Using a reranker to filter graph insertions** does not work (dev). The reranker scores similarity to the question, and the decisive memory is, by construction, dissimilar to the question.

### Limitations

- **Not computed:** stage 1 on ImplexConv and all variants over the EverMemOS-style stack in stage 2. The Kaggle GPU quota ran out.
- **Stage 2 used part of the sample:** 300 of 855 ImplexConv questions (the first in a seed-0 order) and 3 of 18 IMLogic users (397 of 1 931 questions). The free NVIDIA API almost stopped answering. The subset rule was fixed by an amendment before any analysis.
- **The EverMemOS-style stack (D0)** is EverMemOS's open retrieval library (`everalgo`) without LLM memory construction, not the full EverMemOS.
- **Embeddings are e5-base.** HippoRAG 2 / NV-Embed-v2 report higher absolute numbers (MuSiQue R@5 74.7 / 69.7 vs our 51–60). Our conclusions are about differences against the same base.

## Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,bench]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m spacy download en_core_web_sm
.venv/Scripts/python -c "import nltk; [nltk.download(p, download_dir='.cache/nltk') for p in ('wordnet', 'omw-1.4')]"
```

API keys are needed only for the stage-2 planner and the older E1* experiments. They go in `.env` (template: `.env.example`). The repository contains no keys.

## Reproduce with one command

```bash
.venv/Scripts/python scripts/reproduce_tables.py
```

In about 10 s, this recomputes every row of the table above except bench1. It works from the saved per-question outputs in `reports/bench1b/test/`, `reports/implicit/main/` and `reports/implicit/stage2_main/`, with the same test: 10 000 permutations, seed 0, Holm correction. No GPU and no API. It counts wins and losses per question, so they can differ from the reports, which count passages.

**Full reruns** use the Kaggle kernels in `kaggle/*/run.py`:
- `bench1b` — GPU;
- `implicit` — stage 1, GPU;
- `stage2cpu` — stage 2, CPU plus planner calls to the NVIDIA API.

Code and data reach the kernels as a dataset (`scripts/bench1_kaggle_pack.py`). Data are downloaded at pinned revisions; IMLogic is placed manually (see *Data and licenses*).

## Layout

| path | what |
|---|---|
| `ctxgraph/v2/` | graph v2: concept extraction (spaCy / pymorphy3), SQLite store, PPR, gate (`gate.py`) |
| `ctxgraph/` (top level) | first version — a "menu" of related concepts for the model (E1–E1c; not supported) |
| `scripts/bench1*_run.py` | MuSiQue / 2Wiki |
| `scripts/implicit_run.py` | IMLogic / ImplexConv: variants A, B, C, D0, D and plan-step search (SC / SR / SG) |
| `scripts/evermem_stack.py` | D0: EverMemOS-style retrieval stack on `everalgo` |
| `scripts/planner.py` | plan steps: one LLM call per question, at most 3 steps |
| `ARCHITECTURE_V2.md` | architecture (Russian); §3.16 is the next step (context-dependent edge weights) |

## Data and licenses

Code is Apache-2.0 (`LICENSE`). The repository contains no datasets; get them from:

| dataset | where | how it reaches a run |
|---|---|---|
| ImplexConv | [Hugging Face: Kaylee0501/ImplexConv](https://huggingface.co/datasets/Kaylee0501/ImplexConv), revision `baf436948f4b` | downloaded by the scripts |
| IMLogic | [authors' repository: ThisIsCosine/IMLogic](https://github.com/ThisIsCosine/IMLogic) (`HaluMem-Medium.jsonl`, `qa/`), paper [arXiv 2606.23283](https://arxiv.org/abs/2606.23283) | placed manually in `.cache/bench/imlogic/` |
| MuSiQue, 2WikiMultiHopQA | [Hugging Face: osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2) (test), [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) (dev) | downloaded by the scripts (`scripts/bench1_run.py`) |

- **IMLogic has no license**, and the underlying HaluMem is CC BY-NC-ND 4.0. So `reports/` holds only IMLogic question ids, ranks and metrics — no question or memory text.
- D0 uses `everalgo` (Apache-2.0), installed via pip.
