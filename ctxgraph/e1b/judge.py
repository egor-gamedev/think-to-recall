"""Судья E1b: оценка ответа по спискам из скелета проекта (CTXGRAPH_BENCH.md §5).

Судья видит факты проекта (устаревшие помечены), вопрос, опорные факты, ожидаемый вывод,
неуместные советы и ответ. Вариант памяти и блок памяти он не видит. Ответ — строгий JSON.
API — любой OpenAI-совместимый (клиент из E1); расход меряется числом запросов.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from ctxgraph.e1.llm import Client
from ctxgraph.e1b.runner import project_dir

JUDGE_PROMPT_VERSION = "e1b.3"  # e1b.2: уточнена «выдумка»; e1b.3: + «постороннее» (пилот 2)

JUDGE_SYSTEM = """\
Ты — строгий и аккуратный оценщик. Пользователь долго обсуждал с ассистентом свой проект, \
потом задал вопрос. Тебе даны факты проекта, вопрос, критерии и ответ ассистента. Оцени ответ \
ТОЛЬКО по критериям ниже, не по общему качеству и не по стилю.

Определения:
1. anchors — для каждого опорного факта: true, если ответ явно опирается на него (упоминает, \
учитывает или подстраивает совет под него). Для факта, который заменил более ранний, true \
только если ответ исходит из НОВОГО варианта.
2. inference — true, если ответ приходит к ожидаемому выводу ИЛИ задаёт уточняющий вопрос \
ровно по этому пункту. Уточнение засчитывается как правильное поведение.
3. inappropriate — true, если ответ даёт шаблонный совет из списка неуместных (или по сути \
такой же), противоречащий проекту, без оговорки. Если ответ упоминает такой вариант, но \
отвергает его из-за особенностей проекта, это не неуместный совет.
4. stale — true, если ответ исходит из устаревшего (заменённого) факта как из текущего.
5. fabrication — true, если ответ утверждает как факт о ПРОЕКТЕ ПОЛЬЗОВАТЕЛЯ что-то, чего нет \
среди фактов и в вопросе, или что противоречит фактам (выдуманные решения, ограничения, \
инструменты проекта). Выдумка — только утверждение, что в проекте УЖЕ есть, решено или \
действует X. Предложения, общие рекомендации и примеры кода — не выдумка, даже если они \
неуместны (это пункт 3). Устаревший факт отмечай в stale, а не здесь.
6. intrusion — true, если в ответе есть пункты на темы, не относящиеся ни к вопросу, ни \
к проекту (например, термины и инструменты из совсем другой области). Перечисли их в \
"intruded".
7. retention — только если дано «Ранее пользователь сказал»: true, если ответ использует это \
знание; иначе null.

Ответ ассистента мог быть обрезан по лимиту длины — оценивай то, что есть.
Верни только JSON без пояснений вокруг:
{"anchors": {"<id>": true|false, ...}, "inference": true|false, "inappropriate": true|false, \
"stale": true|false, "fabrication": true|false, "fabricated": ["<что выдумано>", ...], \
"intrusion": true|false, "intruded": ["<что постороннее>", ...], "retention": true|false|null, "why": "<одно-два предложения>"}"""


def load_spec(root: Path, pid: str) -> dict:
    import yaml

    return yaml.safe_load((project_dir(root, pid) / "spec.yaml").read_text(encoding="utf-8"))


def facts_block(spec: dict) -> str:
    replaced_by = {v["replaces"]: k for k, v in spec["facts"].items() if "replaces" in v}
    lines = []
    for fid, f in spec["facts"].items():
        mark = f" [УСТАРЕЛО, заменено {replaced_by[fid]}]" if fid in replaced_by else ""
        mark += f" [заменяет {f['replaces']}]" if "replaces" in f else ""
        lines.append(f"{fid}{mark}: {f['text']}")
    return "\n".join(lines)


#: e1c.1 (E1c, решение Егора после пробы потолка): строгий «вывод» — частичный = нет;
#: на «уточнить» — только явный вопрос или названная развилка. В промпт добавляется тип вопроса.
_INFERENCE_E1C = (
    "2. inference — true, ТОЛЬКО если в ответе есть все ключевые элементы ожидаемого вывода "
    "и ответ им не противоречит. Частичный вывод (часть элементов, верное направление без "
    "ключевого проектного факта, общие слова) — false. Если ТИП ВОПРОСА «уточнить»: true, только "
    "если ответ задаёт пользователю явный уточняющий вопрос или прямо называет развилку из "
    "ожидаемого вывода; если ответ молча выбирает одну версию и делает её — false, даже если "
    "сделано хорошо.\n"
)
JUDGE_SYSTEM_E1C = (
    JUDGE_SYSTEM[: JUDGE_SYSTEM.index("2. inference")]
    + _INFERENCE_E1C
    + JUDGE_SYSTEM[JUDGE_SYSTEM.index("3. inappropriate") :]
)
#: e1c.2 (Егор, после проверки 5 вердиктов): обязательное условие по типу вопроса проверяется
#: ПЕРВЫМ и пишется отдельной строкой; не выполнено → inference = false (движок тоже обнуляет).
_MANDATORY_E1C2 = (
    "0. mandatory — СНАЧАЛА проверь обязательное условие для ТИПА ВОПРОСА и запиши проверку "
    "одной строкой в поле mandatory, а выполнено ли — в mandatory_met:\n"
    "   - «решение менялось»: ответ НЕ предлагает отвергнутый или заменённый вариант ни как "
    "текущий, ни как опцию, проверку или альтернативу «на выбор». Предложил — не выполнено.\n"
    "   - «уточнить»: ответ задаёт пользователю явный уточняющий вопрос или прямо называет "
    "развилку из ожидаемого вывода. Молча выбрал одну версию и сделал — не выполнено.\n"
    "   - «связать»: ответ опирается на связующие факты из списка опорных (связывает их между "
    "собой в выводе). Без них — не выполнено.\n"
    "   - «вспомнить»: обязательного условия нет, mandatory_met = true.\n"
    "   Если обязательное условие не выполнено, inference = false независимо от остального.\n"
)
JUDGE_SYSTEM_E1C2 = (
    JUDGE_SYSTEM_E1C[: JUDGE_SYSTEM_E1C.index("1. anchors")]
    + _MANDATORY_E1C2
    + JUDGE_SYSTEM_E1C[JUDGE_SYSTEM_E1C.index("1. anchors") :]
).replace(
    '{"anchors":',
    '{"mandatory": "<одна строка: условие и выполнено ли>", "mandatory_met": true|false, "anchors":',
)
RUBRICS = {"e1b.3": JUDGE_SYSTEM, "e1c.1": JUDGE_SYSTEM_E1C, "e1c.2": JUDGE_SYSTEM_E1C2}
TYPE_RU = {
    "recall": "вспомнить",
    "link": "связать",
    "change": "решение менялось",
    "clarify": "уточнить",
}


def judge_prompt(spec: dict, q: dict, answer: str, *, with_type: bool = False) -> str:
    anchors = "\n".join(f"- {a}: {spec['facts'][a]['text']}" for a in q["anchors"])
    bad = "\n".join(f"- {b}" for b in q.get("inappropriate", [])) or "- (нет)"
    parts = [
        f"ФАКТЫ ПРОЕКТА «{spec['title']}»:\n{facts_block(spec)}",
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{q['text']}",
        *([f"ТИП ВОПРОСА: {TYPE_RU.get(q['type'], q['type'])}"] if with_type else []),
        f"ОПОРНЫЕ ФАКТЫ:\n{anchors}",
        f"ОЖИДАЕМЫЙ ВЫВОД:\n{q['expected']}",
        f"НЕУМЕСТНЫЕ СОВЕТЫ:\n{bad}",
    ]
    if q.get("retention_of"):
        src = next(
            x for x in spec["questions"] if (x.get("followup") or {}).get("id") == q["retention_of"]
        )
        parts.append(f"РАНЕЕ ПОЛЬЗОВАТЕЛЬ СКАЗАЛ:\n{src['followup']['text']}")
    parts.append(f"ОТВЕТ АССИСТЕНТА:\n{answer or '(пустой ответ)'}")
    return "\n\n".join(parts)


def parse_verdict(q: dict, text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("нет JSON в ответе судьи")
    d = json.loads(m.group(0))
    anchors = {a: bool((d.get("anchors") or {}).get(a, False)) for a in q["anchors"]}
    out = {
        "anchors": anchors,
        "support": sum(anchors.values()) / len(anchors) if anchors else None,
        "inference": bool(d["inference"]),
        "inappropriate": bool(d["inappropriate"]),
        "stale": bool(d.get("stale", False)),
        "fabrication": bool(d["fabrication"]),
        "fabricated": list(d.get("fabricated") or []),
        "intrusion": bool(d.get("intrusion", False)),
        "intruded": list(d.get("intruded") or []),
        "retention": None if not q.get("retention_of") else bool(d.get("retention")),
        "why": d.get("why", ""),
    }
    if "mandatory_met" in d:  # e1c.2: не выполнено обязательное условие типа → вывода нет
        out["mandatory"] = d.get("mandatory", "")
        out["mandatory_met"] = bool(d["mandatory_met"])
        out["inference_raw"] = out["inference"]
        out["inference"] = out["inference"] and out["mandatory_met"]
    return out


def grade(
    client: Client, spec: dict, q: dict, answer: str, *, run_id: str, rubric: str = "e1b.3"
) -> dict:
    msgs = [
        {"role": "system", "content": RUBRICS[rubric]},
        {"role": "user", "content": judge_prompt(spec, q, answer, with_type=rubric != "e1b.3")},
    ]
    last = None
    for attempt in range(2):  # повтор — только при битом JSON, с напоминанием
        r = client.chat(msgs, purpose="judge", run_id=run_id)
        try:
            v = parse_verdict(q, r.content)
            return v | {
                "judge": client.model,
                "prompt_version": rubric,
                "attempts": attempt + 1,
            }
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            last = e
            msgs += [
                {"role": "assistant", "content": r.content},
                {"role": "user", "content": "Верни только JSON строго по схеме."},
            ]
    return {"error": str(last), "judge": client.model, "prompt_version": rubric}


class RequestLedger:
    """Учёт для бесплатного API: считает запросы и токены, лимит — по числу запросов."""

    def __init__(self, path: Path, limit_requests: int) -> None:
        self.path, self.limit, self._lock = Path(path), limit_requests, threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = (
            sum(1 for ln in self.path.read_text("utf-8").splitlines() if ln.strip())
            if self.path.exists()
            else 0
        )

    def check(self) -> None:
        if self.n >= self.limit:
            raise RuntimeError(f"лимит запросов {self.limit} исчерпан")

    def add(self, model: str, usage: dict, *, phase: str, purpose: str, run_id: str) -> float:
        row = {"model": model, "usage": usage, "phase": phase, "purpose": purpose, "run_id": run_id}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            self.n += 1
            fh.write(json.dumps(row) + "\n")
        return 0.0


# ---------------- E1c: уместность выбранных моделью пунктов (V7) ----------------
SELECTION_PROMPT_VERSION = "e1c.sel.1"
SELECTION_SYSTEM = """\
Ты оцениваешь, помогают ли воспоминания ответить на вопрос пользователя о его проекте. \
Тебе даны факты проекта, вопрос и пункты, которые ассистент выбрал из своей памяти о прошлом \
разговоре. Для каждого пункта реши: relevant = true, если пункт содержит сведения о проекте, \
которые нужны или прямо полезны для ответа именно на этот вопрос (факт, решение, ограничение, \
смена решения по теме вопроса); false — если пункт не по теме вопроса или содержит только \
общие сведения, не меняющие ответ. Не оценивай качество самого пункта и ответа.
Верни только JSON: {"items": [{"n": <номер>, "relevant": true|false}, ...], "why": "<одно-два предложения>"}"""


def selection_prompt(spec: dict, q: dict, items: list[tuple[int, str]]) -> str:
    listed = "\n".join(f"{n}. {text}" for n, text in items)
    return (
        f"ФАКТЫ ПРОЕКТА «{spec['title']}»:\n{facts_block(spec)}\n\n"
        f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{q['text']}\n\nВЫБРАННЫЕ ПУНКТЫ ПАМЯТИ:\n{listed}"
    )


def grade_selection(
    client: Client, spec: dict, q: dict, items: list[tuple[int, str]], *, run_id: str
) -> dict:
    """items: [(номер в меню, текст раскрытого пункта)]; пусто — не вызывать."""
    msgs = [
        {"role": "system", "content": SELECTION_SYSTEM},
        {"role": "user", "content": selection_prompt(spec, q, items)},
    ]
    r = client.chat(msgs, purpose="selection", run_id=run_id)
    m = re.search(r"\{.*\}", r.content, re.S)
    if not m:
        return {"error": "нет JSON", "prompt_version": SELECTION_PROMPT_VERSION}
    d = json.loads(m.group(0))
    rel = {int(x["n"]): bool(x["relevant"]) for x in d.get("items", []) if "n" in x}
    return {
        "relevant": {n: rel.get(n, False) for n, _ in items},
        "share": sum(rel.get(n, False) for n, _ in items) / len(items),
        "why": d.get("why", ""),
        "judge": client.model,
        "prompt_version": SELECTION_PROMPT_VERSION,
    }
