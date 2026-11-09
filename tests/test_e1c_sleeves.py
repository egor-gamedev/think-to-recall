"""V8 «рукава»: деление вопроса на темы, поиск и граф по рукавам, пересечения, честный объём."""

import re

import pytest

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import sleeves as S
from ctxgraph.e1b.chat import Turn
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens

#: История, где веб-дашборд связан и со входом по JWT, и с переводами.
HISTORY = [
    ("user", "Делаем мобильное приложение кошелька на React Native, бэкенд на NestJS."),
    ("assistant", "Для React Native удобно держать навигацию в одном стеке."),
    ("user", "Решили добавить веб-дашборд на Next.js только для просмотра портфеля."),
    ("assistant", "Веб-дашборд можно собрать на Next.js со статической генерацией."),
    ("user", "Для веб-дашборда бэкенд выдаёт JWT-сессию после проверки подписи SIWE."),
    ("assistant", "JWT-сессию лучше держать в httpOnly cookie, срок неделя."),
    ("user", "Интерфейс на русском и английском, переводы храним в JSON-файлах i18next."),
    ("user", "Переводы i18next должны быть общими для мобилки и веб-дашборда."),
    ("assistant", "Общий пакет с переводами i18next в монорепе подключается и в Next.js."),
    ("user", "Пуши шлём через Firebase Cloud Messaging."),
    ("assistant", "Для пушей нужен отдельный модуль в NestJS."),
    ("user", "Узлы блокчейна берём у Alchemy на бесплатном тарифе."),
]
Q = "Сделай вход по JWT. И где хранить переводы?"


@pytest.fixture(scope="module")
def arc():
    turns = [Turn(f"S1.{i}", r, t, "load", approx_tokens(t)) for i, (r, t) in enumerate(HISTORY)]
    return M.Archive(turns, HashingEncoder())


def test_two_sleeves(arc):
    sl = S.split_sleeves(arc, Q)
    assert len(sl) == 2
    assert "jwt" in sl[0].anchors and "перевод" in sl[1].anchors


def test_intersection_on_dashboard(arc):
    c = S.sleeve_candidates(arc, Q)
    assert c["intersections"], "нет узла, достижимого из обоих рукавов"
    for tid in c["intersections"]:
        assert "дашборд" in arc.by_id[tid].text
        assert c["reached"][tid] == {"s1", "s2"}
    text, log = S.v8_auto(arc, Q, approx_tokens, 600)
    assert "связывает темы: «JWT» и «переводы»" in text
    assert log["n_sleeves"] == 2 and log["n_intersections"] >= 1
    assert log["picked"][0] in c["intersections"]  # пересечения — первыми


def test_single_topic_is_one_sleeve(arc):
    c = S.sleeve_candidates(arc, "Где хранить переводы?")
    assert len(c["sleeves"]) == 1 and c["intersections"] == []


def test_piece_without_concepts_merges(arc):
    assert len(S.split_sleeves(arc, "Сделай вход по JWT. Пожалуйста, побыстрее!")) == 1


def test_at_most_four_sleeves(arc):
    q = "Про JWT. Про переводы. Про пуши Firebase. Про Alchemy. Про React Native. Про NestJS."
    assert 1 <= len(S.split_sleeves(arc, q)) <= 4


def test_same_budget_and_full_count_as_v6(arc):
    v6 = M.v6_hybrid(arc, Q, approx_tokens, 600)
    v8, _ = S.v8_auto(arc, Q, approx_tokens, 600)
    full = lambda s: len([ln for ln in s.splitlines() if ln.startswith("- [")])  # noqa: E731
    assert approx_tokens(v8) <= 600 and approx_tokens(v6) <= 600
    assert full(v8) == full(v6) == 4


def test_each_sleeve_gets_a_full_fragment(arc):
    c = S.sleeve_candidates(arc, Q)
    picked = S._pick_full(c, 4)
    for s in c["sleeves"]:
        assert any(t in c["per_sleeve"][s.id] for t in picked)


def test_menu_and_block_keep_marks(arc):
    text, menu, log = S.v8_menu(arc, Q, approx_tokens)
    assert re.search(r"^1\. ", text, re.M) and "связывает темы" in text
    inter = S.sleeve_candidates(arc, Q)["intersections"][0]
    n = next(k for k, v in menu.items() if v == inter)
    block = S.v8_block(arc, [n], menu, log, approx_tokens, 600)
    assert "связывает темы" in block and "дашборд" in block


def test_context_used_only_without_anchors(arc):
    sl = S.split_sleeves(arc, "А для него как?", context="Переводы i18next в веб-дашборде.")
    assert len(sl) == 1 and "перевод" in sl[0].query  # якоря контекста дописаны к запросу


def test_fact_coverage():
    cov = S.fact_coverage(["S1.4", "S1.7"], {"S1.4": ["F04b"], "S1.7": ["F17"]}, ["F04b", "F01b"])
    assert cov["covered"] == ["F04b"] and cov["share"] == 0.5
