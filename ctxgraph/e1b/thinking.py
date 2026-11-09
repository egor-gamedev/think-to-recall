"""V9 «рукава мышления» и контроль V9-контроль (набросок для эксперимента после E1c; без запусков).

Модель думает в 2–3 шага, потом отвечает. Перед каждым шагом движок подкладывает «сбоку» блок
«Рядом есть» — предложения из истории (RAG + граф). Различие вариантов — только в том, откуда
берутся предложения:

- **V9 (`mode="sleeves"`)**: текст предыдущего шага (перед первым — вопрос и последняя реплика)
  делится на рукава-темы (`sleeves.split_sleeves`); на каждый рукав — свой поиск и шаг по графу,
  пересечения рукавов первыми с пометкой «связывает темы»; вес предложения ×0,5^глубина.
- **V9-контроль (`mode="control"`)**: один поиск на шаг по тексту шага целиком (как IRCoT) —
  `memory.v6_candidates`; без рукавов, пересечений и затухания веса.

Общее для обоих (чтобы сравнивались только рукава): число шагов, промпты, якорь к вопросу
(предложение показывается, только если близко к вопросу: cos ≥ `anchor_cos`), бюджет периферии
(≤ 20 % токенов контекста шага), подача («Рядом есть», помечено как необязательное), протокол
взятия («БЕРУ: 2»), стоп-правило и бюджет итоговой памяти при ответе (как у V6: 600 токенов).

Правила малого веса (раздел «Рукава мышления» в CONTEXT_GRAPH.md):
- затухание 0,5^d, где d — сколько шагов подряд модель брала предложения; при d > `max_depth`
  предложения не показываются;
- якорь цели — фильтр по близости к исходному вопросу;
- бюджет периферии — `periphery_share` контекста шага;
- стоп: если два шага подряд модель брала предложения, а шаг ушёл от вопроса (cos < `anchor_cos`),
  следующий шаг идёт без периферии и с напоминанием вернуться к вопросу.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import sleeves as S

STEP_INSTR = (
    "Подумай над вопросом пользователя по шагам, ещё не отвечая. Это шаг {k} из {n}: напиши 2–4 "
    "предложения рассуждения. Если что-то из блока «Рядом есть» полезно, в конце шага отдельной "
    "строкой напиши «БЕРУ: <номера>»; если нет — ничего про этот блок не пиши."
)
RETURN_INSTR = "Ты ушёл в сторону от вопроса. Вернись к основному вопросу пользователя: «{q}»."
FINAL_INSTR = (
    "Теперь ответь пользователю. Выше твои шаги рассуждения и взятые воспоминания; используй их, "
    "только если они относятся к вопросу."
)
PERIPHERY_HEADER = "Рядом есть (необязательно, можно не использовать; это не часть задачи):"
_TAKE = re.compile(r"^\W*БЕРУ\s*[:\-—]?\s*(\d[\d\s,;и]*)\s*$", re.I | re.M)


@dataclass
class ThinkConfig:
    steps: int = 3
    max_depth: int = 2
    decay: float = 0.5
    anchor_cos: float = 0.25
    progress_cos: float | None = (
        None  # «основной рукав продвинулся»: cos шага с вопросом; None → anchor_cos
    )
    periphery_share: float = 0.2
    final_budget: int = 600  # итоговая память при ответе — как у V6
    k: int = 8
    step_tokens: int = 250
    answer_tokens: int = 1500
    item_tokens: int = 60  # длина одного предложения в блоке «Рядом есть»


@dataclass
class Item:
    n: int
    tid: str
    text: str
    weight: float
    mark: str = ""


@dataclass
class StepLog:
    k: int
    source: str
    items: list[Item] = field(default_factory=list)
    queries: int = 0  # сколько поисков сделано для этого шага (рукава / 1 у контроля)
    n_sleeves: int = 0
    n_intersections: int = 0
    periphery_budget: int = 0
    periphery_tokens: int = 0
    text: str = ""
    taken: list[str] = field(default_factory=list)
    on_topic: bool = True
    returned: bool = False
    depth: int = 0


def _q_cos(arc: M.Archive, qv, tid: str) -> float:
    return float(arc.emb[arc.order[tid]] @ qv)


def _fit_items(
    cands: list[tuple[str, float, str]], arc, count, budget: int, cfg: ThinkConfig
) -> list[Item]:
    """Взять предложения по убыванию веса, пока их текст укладывается в бюджет периферии."""
    out, used = [], 0
    for tid, w, mark in sorted(cands, key=lambda x: (-x[1], x[0])):
        text = M.excerpt(arc.by_id[tid].text, count, cfg.item_tokens)
        t = count(text) + 4
        if used + t > budget:
            continue
        out.append(Item(len(out) + 1, tid, text, w, mark))
        used += t
    return out


def suggest(
    arc: M.Archive,
    source: str,
    question: str,
    *,
    mode: str,
    depth: int,
    budget: int,
    exclude: set[str],
    count: Callable[[str], int],
    cfg: ThinkConfig,
) -> tuple[list[Item], dict]:
    """Предложения «сбоку» для следующего шага."""
    qv = arc.encoder.one(question)
    if mode == "control":  # один поиск по всему тексту шага, без рукавов и затухания
        ranked, _ = M.v6_candidates(arc, source, cfg.k)
        cands = [(t, 1.0 / (i + 1), "") for i, t in enumerate(ranked)]
        info = {"queries": 1, "n_sleeves": 0, "n_intersections": 0}
    else:
        if depth > cfg.max_depth:
            return [], {"queries": 0, "n_sleeves": 0, "n_intersections": 0}
        c = S.sleeve_candidates(arc, source, k=cfg.k)
        w = cfg.decay**depth
        pos = {t: i for i, t in enumerate(c["ranked"])}
        cands = [(t, w / (pos[t] + 1), S._marks(arc, t, c)) for t in c["ranked"]]
        info = {
            "queries": len(c["sleeves"]),
            "n_sleeves": len(c["sleeves"]),
            "n_intersections": len(c["intersections"]),
        }
    cands = [x for x in cands if x[0] not in exclude and _q_cos(arc, qv, x[0]) >= cfg.anchor_cos]
    return _fit_items(cands, arc, count, budget, cfg), info


def render_periphery(items: list[Item]) -> str:
    if not items:
        return ""
    return "\n".join([PERIPHERY_HEADER] + [f"{it.n}.{it.mark} {it.text}" for it in items])


def parse_take(text: str, items: list[Item]) -> tuple[str, list[Item]]:
    """«БЕРУ: 1, 3» в конце шага → взятые предложения; строка из текста шага убирается."""
    by_n = {it.n: it for it in items}
    taken = []
    for m in _TAKE.finditer(text or ""):
        taken += [by_n[int(x)] for x in re.findall(r"\d+", m.group(1)) if int(x) in by_n]
    return _TAKE.sub("", text or "").strip(), list({it.tid: it for it in taken}.values())


def run_thinking(
    llm,
    arc: M.Archive,
    question: str,
    base_messages: list[dict],
    count: Callable[[str], int],
    *,
    mode: str = "sleeves",
    context: str | None = None,
    cfg: ThinkConfig | None = None,
) -> dict:
    """Шаги мышления с периферией, затем ответ. `base_messages` — система + окно (как в раннере)."""
    cfg = cfg or ThinkConfig()
    assert mode in ("sleeves", "control")
    qv = arc.encoder.one(question)
    steps: list[StepLog] = []
    taken: list[Item] = []
    shown: set[str] = set()
    depth, streak, back = 0, 0, False
    source = question + (" " + context if context else "")
    usage = []
    for k in range(1, cfg.steps + 1):
        prev = "\n".join(f"Шаг {s.k}: {s.text}" for s in steps)
        head = f"{question}\n\n{prev}\n\n" if prev else f"{question}\n\n"
        ctx_tokens = sum(count(m["content"]) for m in base_messages) + count(head)
        budget = int(cfg.periphery_share * ctx_tokens)
        log = StepLog(k, source, periphery_budget=budget, depth=depth, returned=back)
        if not back and arc.turns:
            items, info = suggest(arc, source, question, mode=mode, depth=depth, budget=budget,
                                  exclude=shown | {t.tid for t in taken}, count=count, cfg=cfg)  # fmt: skip
            log.items, log.queries = items, info["queries"]
            log.n_sleeves, log.n_intersections = info["n_sleeves"], info["n_intersections"]
            log.periphery_tokens = sum(count(it.text) + 4 for it in items)
            shown |= {it.tid for it in items}
        parts = [head.rstrip()]
        if log.items:
            parts.append(render_periphery(log.items))
        if back:
            parts.append(RETURN_INSTR.format(q=question))
        parts.append(STEP_INSTR.format(k=k, n=cfg.steps))
        r = llm.chat(
            base_messages + [{"role": "user", "content": "\n\n".join(parts)}],
            max_tokens=cfg.step_tokens,
        )
        usage.append(r.get("usage", {}))
        log.text, took = parse_take(r.get("content", ""), log.items)
        log.taken = [it.tid for it in took]
        taken += took
        thr = cfg.anchor_cos if cfg.progress_cos is None else cfg.progress_cos
        log.on_topic = float(arc.encoder.one(log.text or " ") @ qv) >= thr
        streak = streak + 1 if (took and not log.on_topic) else 0
        back = (
            streak >= 2
        )  # стоп: два шага подряд только предложения, основной рукав не продвинулся
        if back:
            streak = 0
        depth = depth + 1 if (mode == "sleeves" and took) else 0
        # рекурсия: следующий шаг ищет от текста шага и от взятого (взятое раскрывается дальше)
        source = log.text + "".join(" " + arc.by_id[it.tid].text for it in took)
        steps.append(log)
    lines = [f"- {M.excerpt(arc.by_id[it.tid].text, count, 110)}" for it in taken]
    memory = M._fit(lines, count, cfg.final_budget) if lines else ""
    final = "\n\n".join(
        [question, "Твои шаги рассуждения:\n" + "\n".join(f"Шаг {s.k}: {s.text}" for s in steps)]
        + ([memory] if memory else [])
        + [FINAL_INSTR]
    )
    r = llm.chat(base_messages + [{"role": "user", "content": final}], max_tokens=cfg.answer_tokens)
    usage.append(r.get("usage", {}))
    return {
        "mode": mode,
        "answer": r.get("content", ""),
        "memory": memory,
        "steps": steps,
        "taken": [it.tid for it in taken],
        "calls": cfg.steps + 1,
        "usage": usage,
        "log": {
            "queries_per_step": [s.queries for s in steps],
            "sleeves_per_step": [s.n_sleeves for s in steps],
            "intersections_per_step": [s.n_intersections for s in steps],
            "off_topic_steps": sum(not s.on_topic for s in steps),
            "returned": any(s.returned for s in steps),
        },
    }
