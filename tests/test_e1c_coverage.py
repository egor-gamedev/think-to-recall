"""Полнота извлечения (coverage) и «последнее решение по сущности» (decisions)."""

from pathlib import Path

import pytest

from ctxgraph.e1b import coverage as C
from ctxgraph.e1b import decisions as D
from ctxgraph.e1b import memory as M
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens

ROOT = Path(__file__).resolve().parents[1]

TEXT = {
    "S1.0": "Храним записи пациентов в базе Firebird, доступ через WireGuard.",
    "S1.1": "```sql\nselect 1\n```\nСлоты врача считаем по расписанию филиала, шаг 20 минут.",
    "S2.0": "Напоминания шлём за сутки до приёма в Telegram.",
}
FACTS = {"F1": ["S1.0"], "F2": ["S1.1"], "F3": ["S2.0"], "F4": []}


def block(*tids, label=None):
    lines = [M.HEADER] + [f"- [пользователь] {TEXT[t]}" for t in tids]
    if label:
        lines.append(f"Ещё есть: {label}")
    return "\n".join(lines)


def test_full_fragment_counts_label_does_not():
    mem = block("S1.0", label=TEXT["S2.0"][:40])
    r = C.anchor_coverage(["F1", "F3"], FACTS, TEXT, mem)
    assert r["in"] == ["F1"] and r["missing"] == ["F3"]
    assert (r["n"], r["need"], r["share"], r["all"]) == (1, 2, 0.5, False)
    assert C.in_memory(TEXT["S2.0"], mem) == "label"


def test_code_block_dropped_before_matching():
    mem = block("S1.1").replace("```sql\nselect 1\n```\n", "[код] ")
    assert C.anchor_coverage(["F2"], FACTS, TEXT, mem)["all"]


def test_union_over_steps_and_ids():
    steps = [block("S1.0"), "", block("S2.0")]
    r = C.anchor_coverage(["F1", "F2", "F3"], FACTS, TEXT, steps)
    assert r["in"] == ["F1", "F3"] and not r["all"]
    r = C.anchor_coverage(["F1", "F2", "F3"], FACTS, TEXT, steps, shown_ids=["S1.1"])
    assert r["all"] and r["share"] == 1.0


def test_empty_memory_and_untagged_fact():
    r = C.anchor_coverage(["F1", "F4", "F1"], FACTS, TEXT, "")
    assert (r["n"], r["need"], r["all"]) == (0, 2, False)
    assert C.anchor_coverage([], FACTS, TEXT, block("S1.0"))["all"] is False


def test_load_turns_ids_match_runner():
    from ctxgraph.e1b.runner import load_project

    spec, ft, text = C.load_turns(ROOT, "clinic-bot")
    ids = [t["id"] for t in load_project(ROOT, "clinic-bot")[1]]
    assert list(text) == ids
    assert all(t in text for ts in ft.values() for t in ts)
    assert set(ft) <= set(spec["facts"])


# ---------- последнее решение ----------
def _arc(turns):
    return M.Archive(
        [Turn(f"S1.{i}", r, x, "load", approx_tokens(x)) for i, (r, x) in enumerate(turns)],
        HashingEncoder(),
    )


@pytest.fixture
def arc():
    return _arc(
        [
            ("user", "Бэкенд пишем на Django, база PostgreSQL."),
            ("assistant", "Хорошо, PostgreSQL подойдёт, настрою Django ORM."),
            ("user", "А Redis для очереди задач нужен?"),
            ("user", "Для очереди решили брать Redis, Celery поверх."),
            ("user", "Перешли с PostgreSQL на SQLite: проект маленький, сервер слабый."),
            ("user", "Вчера смотрел логи Redis, там пусто."),
        ]
    )


def test_latest_decision_skips_questions_and_chatter(arc):
    assert D.latest_decision(arc, "redis") == "S1.3"
    assert D.latest_decision(arc, "redis", decision_only=False) == "S1.5"


def test_replaced_decision_goes_to_chain_end(arc):
    assert arc.replaces.get("S1.4") == ["S1.0"]
    assert D.latest_decision(arc, "postgresql") == "S1.4"
    # сущность, названная только в заменённой реплике, ведёт к концу цепочки
    assert D.latest_decision(arc, "django") == "S1.4"


def test_assistant_turns_never_decisions(arc):
    assert not D.is_decision(arc, "S1.1", "postgresql")
    assert D.latest_decision(arc, "нет такой") is None


def test_render_current_marks_and_dedups(arc):
    text = D.render_current(arc, {"postgresql": "S1.4", "django": "S1.4"}, approx_tokens)
    assert text.count("\n") == 0 and text.startswith("- [актуально:")
