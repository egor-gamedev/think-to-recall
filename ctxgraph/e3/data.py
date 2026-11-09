"""Данные E3: загрузка E1c + вставки событий отказа, события, пробные вопросы; проверка целостности.

    benchmark_e3/projects/<pid>/inserts.yaml   — пары реплик, вставленные после реплики `after`
    benchmark_e3/projects/<pid>/events.yaml    — события отказа (id, тип, реплика, отвергнутое, формы)
    benchmark_e3/projects/<pid>/questions.yaml — пробные вопросы (ловушка/условный/возврат/контроль)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import yaml

from ctxgraph.e1b.runner import load_project

KINDS = ("trap", "conditional", "return", "control")
TYPES = ("refusal", "replace", "conditional", "return")
#: Доли видов вопросов из DECISION_BENCH.md (п. 2), допуск ±5 п.п.
SHARES = {"trap": 0.50, "conditional": 0.15, "return": 0.10, "control": 0.25}


def _y(path: Path) -> dict:
    return yaml.safe_load(path.read_text("utf-8"))


def project_dir(root: Path, pid: str) -> Path:
    return Path(root) / "benchmark_e3" / "projects" / pid


def load(root: Path, pid: str) -> dict:
    d = project_dir(root, pid)
    return {
        "inserts": _y(d / "inserts.yaml")["inserts"],
        "events": _y(d / "events.yaml")["events"],
        "questions": _y(d / "questions.yaml")["questions"],
    }


def merged_turns(root: Path, pid: str) -> list[dict]:
    """Загрузка E1c со вставками E3 на своих местах (порядок = время в диалоге)."""
    _, turns = load_project(Path(root), pid)
    after: dict[str, list[dict]] = {}
    for ins in load(root, pid)["inserts"]:
        after.setdefault(ins["after"], []).extend(
            {"id": t["id"], "role": t["role"], "text": t["text"].strip(), "inserted": True}
            for t in ins["turns"]
        )
    out = []
    for t in turns:
        out.append(t | {"inserted": False})
        out += after.pop(t["id"], [])
    if after:
        raise KeyError(f"вставки после несуществующих реплик: {sorted(after)}")
    return out


def mentions(text: str, forms: list[str]) -> list[str]:
    """Формы отвергнутого варианта, встречающиеся в тексте (с начала слова, без учёта регистра)."""
    return [f for f in forms if re.search(r"(?<!\w)" + f, text, re.I)]


def validate(root: Path, pid: str, min_events: int = 15, min_questions: int = 40) -> list[str]:
    """Список проблем (пустой — данные в порядке)."""
    probs: list[str] = []
    data = load(root, pid)
    _, base = load_project(Path(root), pid)
    base_ids = {t["id"]: t for t in base}
    for ins in data["inserts"]:
        a = base_ids.get(ins["after"])
        if a is None:
            probs.append(f"вставка после несуществующей {ins['after']}")
            continue
        if a["role"] != "assistant":
            probs.append(f"вставка после реплики пользователя {ins['after']}")
        roles = [t["role"] for t in ins["turns"]]
        if roles != ["user", "assistant"] * (len(roles) // 2) or len(roles) % 2:
            probs.append(f"вставка после {ins['after']}: роли не чередуются {roles}")
        for n, t in enumerate(ins["turns"], 1):
            if t["id"] != f"{ins['after']}+{n}":
                probs.append(f"id вставки {t['id']} ≠ {ins['after']}+{n}")
    merged = merged_turns(root, pid)
    by_id = {t["id"]: t for t in merged}
    roles = [t["role"] for t in merged]
    if any(a == b for a, b in zip(roles, roles[1:], strict=False)):
        probs.append("в склеенной загрузке роли не чередуются")
    ev = {e["id"]: e for e in data["events"]}
    if len(ev) != len(data["events"]):
        probs.append("повтор id события")
    if len(ev) < min_events:
        probs.append(f"событий {len(ev)} < {min_events}")
    for e in data["events"]:
        if e["type"] not in TYPES:
            probs.append(f"{e['id']}: тип {e['type']}")
        t = by_id.get(e["turn"])
        if t is None or t["role"] != "user":
            probs.append(f"{e['id']}: реплика {e['turn']} не найдена или не пользователя")
        if not e["rejected"].get("forms"):
            probs.append(f"{e['id']}: нет форм написания")
        if e["type"] == "conditional" and not (e.get("forbidden_for") and e.get("allowed_for")):
            probs.append(f"{e['id']}: условный без forbidden_for/allowed_for")
        if e["type"] == "return" and not e.get("returns_to"):
            probs.append(f"{e['id']}: возврат без returns_to")
    qs = data["questions"]
    if len({q["id"] for q in qs}) != len(qs):
        probs.append("повтор id вопроса")
    if len(qs) < min_questions:
        probs.append(f"вопросов {len(qs)} < {min_questions}")
    kinds = Counter(q["kind"] for q in qs)
    for k, share in SHARES.items():
        if abs(kinds[k] / max(len(qs), 1) - share) > 0.05:
            probs.append(f"доля «{k}» {kinds[k]}/{len(qs)} вне {share:.0%} ± 5 п.п.")
    for q in qs:
        if q["kind"] not in KINDS:
            probs.append(f"{q['id']}: вид {q['kind']}")
        if q["kind"] == "control":
            if "event" in q or not q.get("allowed"):
                probs.append(f"{q['id']}: контроль с событием или без allowed")
            continue
        e = ev.get(q.get("event"))
        if e is None:
            probs.append(f"{q['id']}: нет события {q.get('event')}")
            continue
        if e.get("superseded_by"):
            probs.append(f"{q['id']}: событие {e['id']} отменено {e['superseded_by']}")
        if q["kind"] == "trap" and (hit := mentions(q["text"], e["rejected"]["forms"])):
            probs.append(f"{q['id']}: ловушка упоминает отвергнутое {hit}")
        if q["kind"] == "conditional" and e["type"] != "conditional":
            probs.append(f"{q['id']}: условный вопрос к событию типа {e['type']}")
        if q["kind"] == "return" and e["type"] != "return":
            probs.append(f"{q['id']}: вопрос-возврат к событию типа {e['type']}")
        if not (q.get("expected") and q.get("fail")):
            probs.append(f"{q['id']}: нет expected/fail")
    return probs
