"""E1b: сжатие, память вариантов, прогон проекта на поддельной модели (без сети)."""

import json
from pathlib import Path

from ctxgraph.e1b import memory as M
from ctxgraph.e1b.chat import ChatState, Limits, Turn, compress_if_needed
from ctxgraph.e1b.runner import Session, load_project, run_project
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    """Резюме — короткий текст; ответ — эхо вопроса; в V4 один раз вызывает expand."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens, tools=None):
        self.calls.append({"n": len(messages), "tools": bool(tools)})
        if messages[0]["content"].startswith("You maintain a running summary"):
            return {
                "content": "Резюме: проект, решения.",
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            }
        if tools and not any(m["role"] == "tool" for m in messages):
            user = messages[-1]["content"]
            nid = user.split("[m:", 1)[1].split("]", 1)[0] if "[m:" in user else "x"
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "expand", "arguments": json.dumps({"id": "m:" + nid})},
                    }
                ],
                "usage": {},
            }
        return {
            "content": "Ответ: " + messages[-1]["content"][-60:],
            "usage": {"prompt_tokens": 500, "completion_tokens": 50},
        }


def test_compression_keeps_window_within_budget():
    lim = Limits(context=2000, answer=300, memory=200, summary=200, system=100)
    st = ChatState()
    for i in range(30):
        compress_if_needed(st, lim, lambda prev, ts: "сводка", approx_tokens, reserve=80)
        st.window.append(
            Turn(str(i), "user" if i % 2 == 0 else "assistant", "слово " * 60, "load", 80)
        )
        assert sum(t.tokens for t in st.window) <= lim.window
    assert st.compressions and len(st.archive) + len(st.window) == 30


def test_archive_replaces_and_variants_respect_budget():
    turns = [
        Turn("a", "user", "Историю цен храним в PostgreSQL.", "load", 10),
        Turn("b", "assistant", "Ок, схема для PostgreSQL: таблица price_points.", "load", 12),
        Turn("c", "user", "Отказались от PostgreSQL: историю цен держим в SQLite.", "load", 12),
        Turn("d", "assistant", "SQLite подходит, включите WAL.", "model", 8),
    ]
    arc = M.Archive(turns, HashingEncoder())
    assert "c" in arc.replaces and "a" in arc.replaces["c"]
    q = "Какую базу использовать для истории цен?"
    for fn in (M.v2_rag, M.v3_graph, M.v6_hybrid):
        text = fn(arc, q, approx_tokens, 150)
        assert approx_tokens(text) <= 150
    v6 = M.v6_hybrid(arc, q, approx_tokens, 400)
    assert M.HEADER in v6 and "сказано моделью" in v6
    menu, expand = M.v4_menu(arc, q, approx_tokens, 300)
    assert all(k.startswith("m:") for k in expand)


def test_run_project_end_to_end_with_fake_llm():
    spec, turns = load_project(ROOT, "rnaseq-thesis")
    lim = Limits(context=4000, answer=600, memory=400, summary=300, system=200)
    sess = Session(FakeLLM(), approx_tokens, HashingEncoder(), lim)
    _, pool_turns = load_project(ROOT, "django-shop")
    pool = [
        Turn(t["id"], t["role"], t["text"], "load", approx_tokens(t["text"])) for t in pool_turns
    ]
    recs = run_project(ROOT, "rnaseq-thesis", sess, pool=pool, questions=["Q01", "Q02", "Q10"])
    load, rest = recs[0], recs[1:]
    assert load["compressions"], "загрузка должна сжиматься"
    assert len(rest) == 6 * 3
    assert {r["variant"] for r in rest} == {1, 2, 3, 4, 5, 6}
    assert all(r["memory_tokens"] <= lim.memory for r in rest)
    assert all(r["memory"] == "" for r in rest if r["variant"] == 1)
    assert any(r["expanded"] for r in rest if r["variant"] == 4)
