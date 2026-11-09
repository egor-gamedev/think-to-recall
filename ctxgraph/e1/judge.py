"""Судья E1: промпт widethink 2026-09-30.1 + поля initiative и off_topic (пре-регистрация §6).

Судья не видит вариант: получает только итоговый ответ (без меню и вызовов инструментов).
Контекст задачи передаётся только для контрольных задач.
"""

from __future__ import annotations

import json

from ctxgraph.e1.llm import Client
from ctxgraph.e1.tasks import render_item

JUDGE_PROMPT_VERSION = "2026-09-30.1+e1.1"

# Дословно widethink/bench/judge.py, JUDGE_PROMPT_VERSION = "2026-09-30.1".
WIDETHINK_JUDGE_SYSTEM = """\
You grade answers for a benchmark of hidden requirements. A task has a standard \
solution that a typical model gives, and a context (code, documents, statements of \
the user) that may imply requirements the standard solution misses. You receive the \
hidden requirements with a rubric for each, and one answer to grade.

For every hidden requirement decide, literally and strictly:
- noticed: the answer explicitly recognizes the requirement (not just something vaguely \
related);
- addressed: the proposed solution actually handles it, not merely mentions it;
- asked: the answer asks the user a question that targets it.

Then list unsupported_claims: special requirements the answer asserts about this user \
that nothing in the context supports (inventions, not reasonable questions). Finally, \
decide kept_standard: whether the answer keeps the standard solution wherever the \
context does not justify a change.

Judge only what is written in the answer."""

E1_ADDITION = """

Also decide, for the whole answer:
- initiative: Does the answer, without being asked, propose a different direction than the \
obvious one or point out something the user did not ask about but should consider, grounded \
in the task or its context? (yes/no)
- off_topic: Does the answer contain a substantial part (a paragraph or more) that is not \
relevant to the user's task? (yes/no)

Give each rationale in one sentence. Reply with json only, in this format:
{"verdicts": [{"requirement_id": "<id>", "noticed": false, "addressed": false, "asked": false, \
"rationale": "<one sentence>"}], "unsupported_claims": ["<claim>"], "kept_standard": true, \
"initiative": false, "off_topic": false}"""

JUDGE_SYSTEM = WIDETHINK_JUDGE_SYSTEM + E1_ADDITION


def judge_prompt(task: dict, answer: str) -> str:
    """Как `widethink.bench.judge.judge_prompt`; контекст — только в контрольных задачах."""
    if task.get("control"):
        context = "\n\n".join(render_item(i) for i in task.get("context", [])) or "(none)"
    else:
        context = "(omitted: each hidden requirement below is self-contained)"
    reqs = task.get("hidden_requirements") or []
    if reqs:
        rubric = "\n\n".join(
            f"[{r['id']}] {r['description'].strip()}\n  noticed? {r['noticed']}\n  addressed? {r['addressed']}"
            for r in reqs
        )
    else:
        rubric = "(none - the standard solution is right for this user)"
    return (
        f"TASK:\n{task['task']}\n\nCONTEXT:\n{context}\n\n"
        f"STANDARD SOLUTION:\n{task['standard_solution']}\n\n"
        f"HIDDEN REQUIREMENTS:\n{rubric}\n\n"
        f"ANSWER TO GRADE:\n<answer>\n{answer}\n</answer>"
    )


def parse_verdict(task: dict, text: str) -> dict:
    """JSON судьи → нормализованный вердикт; пропущенное поле = «нет» (как в widethink)."""
    data = json.loads(text)
    got = {v.get("requirement_id"): v for v in data.get("verdicts", []) if isinstance(v, dict)}
    verdicts = []
    for r in task.get("hidden_requirements") or []:
        v = got.get(r["id"], {})
        verdicts.append(
            {
                "requirement_id": r["id"],
                "noticed": bool(v.get("noticed", False)),
                "addressed": bool(v.get("addressed", False)),
                "asked": bool(v.get("asked", False)),
                "rationale": str(v.get("rationale", "(no verdict returned; counted as missed)")),
                "missing": r["id"] not in got,
            }
        )
    return {
        "verdicts": verdicts,
        "unsupported_claims": [str(c) for c in data.get("unsupported_claims", []) or []],
        "kept_standard": bool(data.get("kept_standard", False)),
        "initiative": bool(data.get("initiative", False)),
        "off_topic": bool(data.get("off_topic", False)),
        "fields_missing": [
            k for k in ("kept_standard", "initiative", "off_topic") if k not in data
        ],
    }


def grade(client: Client, task: dict, answer: str, *, run_id: str) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": judge_prompt(task, answer)},
    ]
    last_err = ""
    for attempt in range(2):
        r = client.chat(
            messages,
            purpose="judge",
            run_id=run_id,
            extra={"response_format": {"type": "json_object"}},
        )
        try:
            out = parse_verdict(task, r.content)
            out.update(
                judge_model=r.model,
                judge_usage=r.usage,
                prompt_version=JUDGE_PROMPT_VERSION,
                attempts=attempt + 1,
            )
            return out
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            last_err = f"{type(e).__name__}: {e}"
    return {
        "error": f"judge returned invalid json: {last_err}",
        "prompt_version": JUDGE_PROMPT_VERSION,
    }
