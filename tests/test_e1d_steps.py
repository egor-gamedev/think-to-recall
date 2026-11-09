"""Этап B: протокол шагов V9 / V9-control на заглушке модели."""

import pytest
from test_e1c_sleeves import HISTORY

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import steps as ST
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens

Q = "Как сделать вход в веб-дашборд и где хранить переводы?"


class StubLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens, tools=None):
        self.calls.append((messages, max_tokens))
        n = len(self.calls)
        text = {1: "Нужно вспомнить, как устроена JWT-сессия после подписи SIWE.",
                2: "Переводы i18next общие для мобилки и веб-дашборда, это важно.",
                3: "Проверить, где лежит общий пакет переводов в монорепе."}.get(n, "Итоговый ответ.")  # fmt: skip
        return {"content": text, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def scorer(q, texts):
    qs = set(q.lower().split())
    return [len(qs & set(t.lower().split())) + 0.001 * len(t) for t in texts]


@pytest.fixture(scope="module")
def arc():
    turns = [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(HISTORY)]
    return M.Archive(turns, HashingEncoder())


BASE = [{"role": "system", "content": "система"}]


@pytest.mark.parametrize("mode", ["control", "sleeves"])
def test_protocol_calls_budget_and_no_repeats(arc, mode):
    llm = StubLLM()
    cfg = ST.StepsConfig(anchor_cos=None)
    r = ST.run_steps(llm, arc, Q, BASE, approx_tokens, scorer, mode=mode, cfg=cfg)
    assert len(llm.calls) == cfg.steps + 1 == r["calls"]
    assert [mt for _, mt in llm.calls] == [cfg.step_tokens] * cfg.steps + [cfg.answer_tokens]
    assert r["answer"] == "Итоговый ответ." and len(r["steps"]) == cfg.steps
    assert r["memory_tokens"] <= cfg.budget_total
    seen = [p for rd in r["rounds"] for unit in rd["picked"] for p in unit]
    assert len(seen) == len(set(seen))  # показанное второй раз не подаётся
    assert [rd["query"] for rd in r["rounds"]] == [Q] + r["steps"]
    # шаг k видит прошлые шаги; ответ видит все шаги и память
    final = llm.calls[-1][0][-1]["content"]
    assert all(s in final for s in r["steps"]) and ST.FINAL_INSTR in final
    assert r["steps"][0] in llm.calls[1][0][-1]["content"]
    assert r["protocol"] == {"empty_steps": 0, "answer_in_step": 0, "long_steps": 0}


def test_round_budget_is_cumulative(arc):
    cfg = ST.StepsConfig(anchor_cos=None, budget_total=200)
    r = ST.run_steps(StubLLM(), arc, Q, BASE, approx_tokens, scorer, mode="control", cfg=cfg)
    used = 0
    for rd in r["rounds"]:
        used = approx_tokens(ST.memory_block([ln for x in r["rounds"][: rd["k"] + 1] for ln in x["lines"]]) or "")
        assert used <= 200 * (rd["k"] + 1) // (cfg.steps + 1)


def test_sleeves_mode_logs_sleeves_and_control_does_not(arc):
    rs = ST.run_steps(StubLLM(), arc, Q, BASE, approx_tokens, scorer, mode="sleeves",
                      cfg=ST.StepsConfig(anchor_cos=None))  # fmt: skip
    rc = ST.run_steps(StubLLM(), arc, Q, BASE, approx_tokens, scorer, mode="control",
                      cfg=ST.StepsConfig(anchor_cos=None))  # fmt: skip
    assert any(rd["sleeves"] for rd in rs["rounds"])
    assert all(rd["sleeves"] == [] for rd in rc["rounds"])
    assert all(len(rd["pool"]) <= 30 for rd in rs["rounds"] + rc["rounds"])


def test_entity_sleeves_variant(arc):
    r = ST.run_steps(StubLLM(), arc, Q, BASE, approx_tokens, scorer, mode="sleeves",
                     cfg=ST.StepsConfig(anchor_cos=None, sleeves="entity"))  # fmt: skip
    assert r["calls"] == 4


def test_empty_archive():
    arc = M.Archive([], HashingEncoder())
    r = ST.run_steps(StubLLM(), arc, Q, BASE, approx_tokens, scorer, mode="sleeves")
    assert r["memory"] == "" and r["answer"] == "Итоговый ответ."


def test_runner_step_and_v2plus_variants():
    from pathlib import Path

    from ctxgraph.e1b.chat import Limits
    from ctxgraph.e1b.runner import Session, run_project

    class LLM(StubLLM):
        def chat(self, messages, max_tokens, tools=None):
            r = super().chat(messages, max_tokens, tools)
            return r | {"content": "Короткое резюме."} if max_tokens == Limits().summary else r

    llm = LLM()
    sess = Session(llm, approx_tokens, HashingEncoder(), Limits())
    sess.scorer = scorer
    root = Path(__file__).resolve().parents[1]
    recs = run_project(root, "clinic-bot", sess, variants=(21, 90, 91), questions=["Q01", "Q02"])
    rs = {(r["variant"], r["question"]): r for r in recs if "variant" in r}
    assert set(rs) == {(v, q) for v in (21, 90, 91) for q in ("Q01", "Q02")}
    assert rs[(21, "Q01")]["steps"] is None and rs[(21, "Q01")]["calls"] == 1
    for v in (90, 91):
        r = rs[(v, "Q01")]
        assert r["calls"] == 4 and len(r["steps"]["steps"]) == 3
        assert r["memory_tokens"] <= Limits().memory
