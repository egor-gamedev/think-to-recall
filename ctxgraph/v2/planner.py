"""Step planner for debugging and benchmarks (ARCHITECTURE_V2 §3.14).

Borrowed: IRCoT (retrieve with the question first, then let reasoning text drive further retrieval)
and Self-Ask (decompose the question into follow-up steps / sub-questions). Unlike IRCoT, the steps
are written in ONE call (cheaper), not one by one with retrieval in between. Each step is a source of
sleeves (word groups), not a single search query.

In the product the model plans by itself and calls `recall(step_text)` after each step; this module
only reproduces that for offline debugging and benchmark runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ctxgraph.v2 import sleeves as SL
from ctxgraph.v2.graph import Graph

PLAN_INSTR = (
    "Before answering, plan. Write up to {n} short reasoning steps (one line each, in the user's "
    "language): what in the project history matters for this question and what else should be "
    "recalled — facts, decisions, constraints. Do NOT answer the question. Format:\n1. ...\n2. ...\n3. ..."
)
_NUM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*(.+?)\s*$", re.M)


def parse_steps(text: str, n: int = 3) -> list[str]:
    steps = [m.group(1).strip() for m in _NUM.finditer(text or "") if len(m.group(1).split()) >= 3]
    if not steps:  # no numbering: sentences
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.split()) >= 3]
    return steps[:n]


@dataclass
class Planned:
    question: str
    first_block: str
    steps: list[str] = field(default_factory=list)
    raw_plan: str = ""
    trace: dict = field(default_factory=dict)  # sleeves_trace over all steps
    memory_for_model: str = ""


def plan_and_recall(g: Graph, llm, question: str, count: Callable[[str], int], *, scorer=None, n_steps: int = 3,
                    system: str = "You help a user with their software project.", plan_tokens: int = 3000,
                    periphery_budget: int = 200) -> Planned:  # fmt: skip
    """1) retrieve with the question; 2) the same model writes <= n steps (no answer);
    3) sleeves from the steps -> mini-graphs -> hybrid final block + periphery."""
    first, _ = g.select(question, 600, count)
    user = (first + "\n\n---\n" if first else "") + question + "\n\n" + PLAN_INSTR.format(n=n_steps)
    r = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=plan_tokens)
    raw = r.get("content") or ""
    steps = parse_steps(raw, n_steps)
    text = " ".join(steps) if steps else question
    tr = SL.sleeves_trace(g, text, question, count, periphery_budget, anchor_cos=0.76, scorer=scorer)
    mem = "\n\n".join(x for x in (tr["final"], tr["periphery"]) if x)
    return Planned(question, first, steps, raw, tr, mem)
