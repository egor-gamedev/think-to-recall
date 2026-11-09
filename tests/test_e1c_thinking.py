"""V9 «рукава мышления» и V9-контроль: правила малого веса и равенство условий (поддельная модель)."""

import pytest

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import thinking as TH
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens
from tests.test_e1c_sleeves import HISTORY, Q

BASE = [{"role": "system", "content": "Ты — ассистент по проекту пользователя. " * 40}]


class ScriptLLM:
    """Отвечает заготовками по порядку; последний вызов — ответ. Запоминает промпты."""

    def __init__(self, steps):
        self.steps, self.prompts = list(steps), []

    def chat(self, messages, max_tokens, tools=None):
        self.prompts.append(messages[-1]["content"])
        text = self.steps.pop(0) if self.steps else "Итоговый ответ."
        return {"content": text, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


@pytest.fixture(scope="module")
def arc():
    turns = [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(HISTORY)]
    return M.Archive(turns, HashingEncoder())


CFG = TH.ThinkConfig(steps=3, anchor_cos=-1.0)  # якорь выключен, чтобы проверять остальное


def test_control_one_search_per_step_v9_per_sleeve(arc):
    step = "Вход по JWT нужен для веб-дашборда. И переводы i18next надо хранить общими."
    ctl = TH.run_thinking(
        ScriptLLM([step] * 3), arc, Q, BASE, approx_tokens, mode="control", cfg=CFG
    )
    v9 = TH.run_thinking(
        ScriptLLM([step] * 3), arc, Q, BASE, approx_tokens, mode="sleeves", cfg=CFG
    )
    assert ctl["log"]["queries_per_step"] == [1, 1, 1]
    assert ctl["log"]["intersections_per_step"] == [0, 0, 0]
    assert v9["log"]["queries_per_step"][0] == 2  # вопрос из двух тем → два поиска
    assert v9["log"]["intersections_per_step"][0] >= 1
    assert ctl["calls"] == v9["calls"] == 4  # 3 шага + ответ у обоих


def test_periphery_within_budget_and_marked_optional(arc):
    llm = ScriptLLM(["Думаю про JWT для дашборда."] * 3)
    res = TH.run_thinking(llm, arc, Q, BASE, approx_tokens, mode="sleeves", cfg=CFG)
    for s in res["steps"]:
        assert s.periphery_tokens <= s.periphery_budget
    assert TH.PERIPHERY_HEADER in llm.prompts[0]


def test_decay_and_depth_limit(arc):
    cfg = TH.ThinkConfig(steps=4, anchor_cos=-1.0, max_depth=1)
    llm = ScriptLLM(["JWT и переводы для веб-дашборда.\nБЕРУ: 1"] * 4)
    res = TH.run_thinking(llm, arc, Q, BASE, approx_tokens, mode="sleeves", cfg=cfg)
    depths = [s.depth for s in res["steps"]]
    # на глубине 2 (> max_depth) предложений нет — брать нечего, глубина сбрасывается
    assert depths == [0, 1, 2, 0]
    w0 = res["steps"][0].items[0].weight
    w1 = res["steps"][1].items[0].weight
    assert w1 <= w0 * cfg.decay + 1e-9  # вес на глубине 1 — не больше половины
    assert res["steps"][2].items == []  # глубже max_depth — предложений нет


def test_stop_rule_returns_to_main(arc):
    cfg = TH.ThinkConfig(
        steps=4, anchor_cos=-1.0, progress_cos=0.99
    )  # любой шаг «уходит от вопроса»
    llm = ScriptLLM(["Пуши Firebase.\nБЕРУ: 1", "Alchemy тариф.\nБЕРУ: 1", "Шаг.", "Шаг."])
    res = TH.run_thinking(llm, arc, Q, BASE, approx_tokens, mode="control", cfg=cfg)
    assert res["steps"][2].returned and res["steps"][2].items == []
    assert TH.RETURN_INSTR.split(".")[0] in llm.prompts[2]
    assert res["log"]["returned"]


def test_anchor_filters_far_suggestions(arc):
    qv = arc.encoder.one(Q)
    cfg = TH.ThinkConfig(steps=1, anchor_cos=0.1)
    res = TH.run_thinking(ScriptLLM(["Шаг."]), arc, Q, BASE, approx_tokens, mode="sleeves", cfg=cfg)
    for it in res["steps"][0].items:
        assert float(arc.emb[arc.order[it.tid]] @ qv) >= 0.1


def test_take_protocol_and_final_budget_same_for_both(arc):
    for mode in ("sleeves", "control"):
        llm = ScriptLLM(["Про JWT и переводы.\nБЕРУ: 1, 2"] * 3)
        res = TH.run_thinking(llm, arc, Q, BASE, approx_tokens, mode=mode, cfg=CFG)
        assert res["taken"] and all("БЕРУ" not in s.text for s in res["steps"])
        assert approx_tokens(res["memory"]) <= CFG.final_budget
        assert TH.FINAL_INSTR in llm.prompts[-1]


def test_parse_take_ignores_unknown_numbers():
    items = [TH.Item(1, "a", "x", 1.0), TH.Item(2, "b", "y", 1.0)]
    text, taken = TH.parse_take("Мысль.\nБЕРУ: 2, 9", items)
    assert text == "Мысль." and [it.tid for it in taken] == ["b"]
