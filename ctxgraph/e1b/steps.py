"""Этап B (E1d): мышление по шагам с поиском после каждого шага — V9 и V9-control.

Протокол одинаков для обоих вариантов (различается только поиск):

    раунд 0: поиск по вопросу → память
    шаг k = 1..steps: модель видит вопрос, память и прошлые шаги, пишет 2–4 предложения
             рассуждения (без ответа) → поиск по тексту шага → память пополняется
    ответ:   модель видит память, вопрос и свои шаги → ответ пользователю

Поиск раунда:
- **V9-control** (`mode="control"`): гибрид RRF(эмбеддинги, BM25) по всему тексту раунда, без
  графа, top-`pool` реплик (как V2+, IRCoT-подобно);
- **V9** (`mode="sleeves"`): рукава из текста раунда (`sleeves="phrase"` — группы слов,
  `v9.phrase_sleeves`, главный вид; `"entity"` — сущности/леммы, описательно), на рукав —
  гибридный поиск top-`k_per_sleeve` + шаг по графу, пересечения с бонусом (с учётом рукавов
  прошлых раундов), фильтр по якорю (близость к вопросу ≥ `anchor_cos`) → первые `pool` реплик.

Общий слой отбора (одинаковый): реплики пула → предложения → cross-encoder (запрос = текст
раунда) → лучшие предложения с соседним. Показанное второй раз не подаётся. Бюджет памяти —
`budget_total` токенов на вопрос суммарно по раундам: раунду k доступно `budget_total·(k+1)/
(steps+1)` нарастающим итогом (неизрасходованное переходит дальше).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import selection as SEL
from ctxgraph.e1b import v9

STEP_INSTR = (
    "Подумай над вопросом пользователя по шагам, ещё не отвечая. Это шаг {k} из {n}: напиши 2–4 "
    "предложения рассуждения — что в истории проекта важно для ответа и что ещё стоит вспомнить. "
    "Не пиши ответ пользователю."
)
FINAL_INSTR = (
    "Теперь ответь пользователю. Выше твои шаги рассуждения; воспоминания используй, только если "
    "они относятся к вопросу."
)
_ANSWERISH = re.compile(r"^\s*(?:ответ|итак, ответ)\s*[:\-—]", re.I | re.M)


@dataclass
class StepsConfig:
    steps: int = 3
    budget_total: int = 600
    pool: int = 30
    step_tokens: int = 250
    answer_tokens: int = 1500
    sleeves: str = "phrase"  # "phrase" | "entity"
    query_mode: str = "clause"  # для sleeves="entity"
    k_per_sleeve: int = 8
    anchor_cos: float | None = 0.76
    bonus: float = v9.BONUS


@dataclass
class Round:
    k: int
    query: str
    pool: list[str] = field(default_factory=list)
    sleeves: list[str] = field(default_factory=list)
    intersections: int = 0
    picked: list[list[tuple[str, int]]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


def retrieve(
    arc: M.Archive,
    mode: str,
    text: str,
    question: str,
    cfg: StepsConfig,
    reached: dict[str, set[str]],
) -> tuple[list[str], dict]:
    """Пул реплик раунда. Для V9 пополняет `reached` (рукава, которые дошли до реплики)."""
    if mode == "control":
        return SEL.hybrid_pool(arc, text, cfg.pool), {"sleeves": [], "intersections": 0}
    if cfg.sleeves == "phrase":
        ss = v9.phrase_sleeves(arc, text)
    else:
        ss = v9.text_sleeves(arc, text, query_mode=cfg.query_mode)
    c = v9.candidates(arc, text, question, sleeves=ss or None, prev_reached=reached,
                      k_per_sleeve=cfg.k_per_sleeve, anchor_cos=cfg.anchor_cos, bonus=cfg.bonus)  # fmt: skip
    for t, th in c["reached"].items():
        reached.setdefault(t, set()).update(th)
    pool = c["ranked"][: cfg.pool]
    return pool, {"sleeves": [s.query for s in c["sleeves"]], "intersections": len(c["intersections"])}


def memory_block(lines: list[str]) -> str:
    return "\n".join([M.HEADER] + lines) if lines else ""


def run_steps(
    llm,
    arc: M.Archive,
    question: str,
    base_messages: list[dict],
    count: Callable[[str], int],
    scorer: Callable[[str, list[str]], list[float]],
    *,
    mode: str = "sleeves",
    cfg: StepsConfig | None = None,
) -> dict:
    """V9 / V9-control на один вопрос. `base_messages` — система + окно (как в раннере)."""
    assert mode in ("sleeves", "control")
    cfg = cfg or StepsConfig()
    rounds: list[Round] = []
    lines: list[str] = []
    shown: set[tuple[str, int]] = set()
    reached: dict[str, set[str]] = {}
    steps: list[str] = []
    usage = []

    def do_round(k: int, text: str) -> None:
        rd = Round(k, text)
        if arc.turns:
            rd.pool, info = retrieve(arc, mode, text, question, cfg, reached)
            rd.sleeves, rd.intersections = info["sleeves"], info["intersections"]
            budget = cfg.budget_total * (k + 1) // (cfg.steps + 1)
            new, picked = SEL.pick_lines(SEL.turn_sentences(arc, rd.pool), text, count, budget, scorer,
                                         head=[M.HEADER] + lines, exclude=shown)  # fmt: skip
            rd.lines, rd.picked = new, picked
            lines.extend(new)
            shown.update(p for unit in picked for p in unit)
        rounds.append(rd)

    do_round(0, question)
    for k in range(1, cfg.steps + 1):
        mem = memory_block(lines)
        prev = "\n".join(f"Шаг {i}: {s}" for i, s in enumerate(steps, 1))
        parts = ([mem, "---"] if mem else []) + [question] + ([prev] if prev else [])
        parts.append(STEP_INSTR.format(k=k, n=cfg.steps))
        r = llm.chat(base_messages + [{"role": "user", "content": "\n\n".join(parts)}],
                     max_tokens=cfg.step_tokens)  # fmt: skip
        usage.append(r.get("usage", {}))
        text = (r.get("content") or "").strip()
        steps.append(text)
        do_round(k, text or question)
    mem = memory_block(lines)
    prev = "\n".join(f"Шаг {i}: {s}" for i, s in enumerate(steps, 1))
    final = "\n\n".join(([mem, "---"] if mem else []) + [question, "Твои шаги рассуждения:\n" + prev, FINAL_INSTR])
    r = llm.chat(base_messages + [{"role": "user", "content": final}], max_tokens=cfg.answer_tokens)
    usage.append(r.get("usage", {}))
    return {
        "mode": mode,
        "answer": r.get("content") or "",
        "finish_reason": r.get("finish_reason"),
        "memory": mem,
        "memory_tokens": count(mem) if mem else 0,
        "steps": steps,
        "rounds": [rd.__dict__ for rd in rounds],
        "calls": cfg.steps + 1,
        "usage": usage,
        "protocol": {
            "empty_steps": sum(not s for s in steps),
            "answer_in_step": sum(bool(_ANSWERISH.search(s)) for s in steps),
            "long_steps": sum(count(s) > cfg.step_tokens - 5 for s in steps),
        },
    }
