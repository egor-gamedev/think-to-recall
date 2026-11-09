"""E3 «Помнит решения»: целостность данных и склейка загрузки со вставками."""

from pathlib import Path

from ctxgraph.e1b.runner import load_project
from ctxgraph.e3 import data as D

ROOT = Path(__file__).resolve().parents[1]
PID = "clinic-bot"


def test_clinic_bot_valid():
    assert D.validate(ROOT, PID) == []


def test_merged_keeps_load_and_order():
    _, base = load_project(ROOT, PID)
    m = D.merged_turns(ROOT, PID)
    assert [t["id"] for t in m if not t["inserted"]] == [t["id"] for t in base]
    pos = {t["id"]: i for i, t in enumerate(m)}
    for ins in D.load(ROOT, PID)["inserts"]:
        assert pos[ins["turns"][0]["id"]] == pos[ins["after"]] + 1


def test_mentions_word_start():
    assert D.mentions("Возьмите RedisStorage", ["Redis"]) == ["Redis"]
    assert D.mentions("оценка от 1 до 5", ["цен"]) == []
    assert D.mentions("цена чистки", ["цен"]) == ["цен"]


def test_validator_catches_trap_leak(tmp_path, monkeypatch):
    data = D.load(ROOT, PID)
    data["questions"][0]["text"] += " Может, Mini App?"
    monkeypatch.setattr(D, "load", lambda root, pid: data)
    assert any("ловушка упоминает" in p for p in D.validate(ROOT, PID))
