"""Раннер E1 без сети: поддельный клиент, офлайн-граф (WordNet/RuWordNet локально)."""

import json
import random

import pytest

from ctxgraph.e1 import tasks as TK
from ctxgraph.e1.budget import BudgetExceeded, Ledger, cost_rub
from ctxgraph.e1.judge import JUDGE_SYSTEM, judge_prompt, parse_verdict
from ctxgraph.e1.llm import Reply
from ctxgraph.e1.variants import GraphFactory, RandomMenus, run_variant
from ctxgraph.embed import HashingEncoder
from ctxgraph.store import Store

TASK = {
    "id": "t-fish",
    "task": "Which fish suit a small aquarium for a beginner?",
    "context": [
        {"id": "note", "kind": "user", "title": "Note", "content": "Tank is 60 litres, tap water."}
    ],
    "standard_solution": "Guppies, neon tetras.",
    "hidden_requirements": [
        {"id": "r1", "description": "d", "evidence": ["note"], "noticed": "n?", "addressed": "a?"}
    ],
    "control": False,
}


class FakeClient:
    """Отвечает по сценарию: список ответов; tool_call — словарь с node_id или 'FIRST' (первый id меню)."""

    def __init__(self, script, ledger=None):
        self.script = list(script)
        self.ledger = ledger
        self.seen: list[list[dict]] = []

    def chat(self, messages, *, purpose, run_id, tools=None, extra=None):
        if self.ledger:
            self.ledger.check()
        self.seen.append(json.loads(json.dumps(messages)))
        step = self.script.pop(0)
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        if self.ledger:
            self.ledger.add(
                "deepseek/deepseek-flash", usage, phase="t", purpose=purpose, run_id=run_id
            )
        if isinstance(step, dict):
            nid = step["node_id"]
            if nid == "FIRST":
                user = messages[1]["content"]
                nid = user.split("[", 2)[1].split("]")[0] if "[" in user else "x"
            tc = [
                {
                    "id": f"c{len(self.seen)}",
                    "type": "function",
                    "function": {"name": "expand", "arguments": json.dumps({"node_id": nid})},
                }
            ]
            return Reply("", tc, "thinking...", usage, "fp1", "deepseek/deepseek-flash", 1.0)
        return Reply(step, [], None, usage, "fp1", "deepseek/deepseek-flash", 1.0)


@pytest.fixture
def graphs(tmp_path):
    db = tmp_path / "cache.sqlite"
    Store(db).close()
    return GraphFactory(db, HashingEncoder(), offline=True)


def test_budget_guard_blocks_before_request(tmp_path):
    led = Ledger(tmp_path / "spend.jsonl", limit_rub=0.01)
    c = FakeClient(["a", "b"], ledger=led)
    c.chat([{"role": "user", "content": "x"}], purpose="answer", run_id="r")
    with pytest.raises(BudgetExceeded):
        c.chat([{"role": "user", "content": "x"}], purpose="answer", run_id="r")
    assert len(c.seen) == 1
    # Сумма переживает перезапуск: новый Ledger читает файл.
    assert Ledger(tmp_path / "spend.jsonl", limit_rub=0.01).spent == pytest.approx(
        cost_rub("deepseek/deepseek-flash", {"prompt_tokens": 1000, "completion_tokens": 100})
    )


def test_variant1_and_8(graphs):
    r1 = run_variant(TASK, 1, 1, client=FakeClient(["answer one"]), graphs=graphs)
    assert r1["answer"] == "answer one" and not r1["error"] and len(r1["calls"]) == 1
    c = FakeClient(["- risk list", "final"])
    r8 = run_variant(TASK, 8, 1, client=c, graphs=graphs)
    assert r8["answer"] == "final" and r8["self_list"] == "- risk list"
    assert TK.SELF_LIST in c.seen[0][1]["content"] and c.seen[1][-1]["content"] == TK.SELF_ANSWER
    assert c.seen[0][0]["content"] == TK.SYSTEM_BASE


def test_tool_loop_limit_and_reasoning_passthrough(graphs):
    script = [{"node_id": "FIRST"}] * 4 + ["done"]
    c = FakeClient(script)
    r = run_variant(TASK, 3, 1, client=c, graphs=graphs)
    assert r["answer"] == "done" and r["expand_calls"] == 4
    assert len(r["expanded"]) <= TK.EXPAND_LIMIT
    last = c.seen[-1]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert tool_msgs[-1]["content"] == "budget exhausted" or len(r["expanded"]) < 3
    assert any(m.get("reasoning_content") for m in last if m["role"] == "assistant")
    sys_text = c.seen[0][0]["content"]
    assert TK.ADD_MENU in sys_text and TK.ADD_TOOL in sys_text
    assert "MAP OF RELATED CONCEPTS:" in c.seen[0][1]["content"]


def test_no_final_answer_is_error(graphs):
    r = run_variant(TASK, 3, 1, client=FakeClient([{"node_id": "x"}] * 6), graphs=graphs)
    assert r["error"] == "no final answer within the call limit"


def test_variant2_uses_v3_expansions_without_tool(graphs):
    c = FakeClient(["ok"])
    r = run_variant(TASK, 2, 1, client=c, graphs=graphs, v3_expanded=["wn:fish.n.01"])
    assert r["answer"] == "ok" and r["expanded"] == ["wn:fish.n.01"]
    assert TK.ADD_TOOL not in c.seen[0][0]["content"]


def test_random_menu_keeps_shape_and_hides_real_nodes(graphs):
    eng = graphs.engine()
    menu = eng.menu(TK.graph_items(TASK))
    pool = {
        "t-fish": [[i.node_id, i.label] for i in menu.items()],
        "other": [[f"o{k}", f"label{k}"] for k in range(50)],
    }
    rm = RandomMenus(pool)
    labels = rm.labels_for("t-fish")
    assert labels and all(lab.startswith("label") for lab in labels)
    rnd, mapping = rm.randomize(menu, labels, random.Random(1))
    assert len(rnd.items()) == len(menu.items())
    assert all(i.node_id.startswith("rnd:") and i.label.startswith("label") for i in rnd.items())
    assert [i.slot for i in rnd.items()] == [i.slot for i in menu.items()]
    assert [i.rel for i in rnd.items()] == [i.rel for i in menu.items()]


def test_judge_prompt_and_parse():
    p = judge_prompt(TASK, "my answer")
    assert "omitted" in p and "Tank is 60" not in p  # контекст только для контрольных
    ctrl = dict(TASK, control=True, hidden_requirements=[])
    assert "Tank is 60" in judge_prompt(ctrl, "a")
    assert JUDGE_SYSTEM.startswith("You grade answers for a benchmark of hidden requirements.")
    v = parse_verdict(
        TASK,
        json.dumps({"verdicts": [{"requirement_id": "r1", "noticed": True}], "initiative": True}),
    )
    assert v["verdicts"][0]["noticed"] is True and v["verdicts"][0]["addressed"] is False
    assert v["initiative"] is True and set(v["fields_missing"]) == {"kept_standard", "off_topic"}
    v2 = parse_verdict(TASK, "{}")
    assert v2["verdicts"][0]["missing"] is True


@pytest.mark.skipif(not TK.WIDETHINK_TASKS.exists(), reason="9 of the 30 E1 tasks live in a sibling project not in this repo")
def test_task_files_hash_and_count():
    hashes = TK.file_hashes()
    assert len(hashes) == 30 and len(TK.load_tasks()) == 30
