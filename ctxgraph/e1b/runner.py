"""Прогон E1b для одного проекта: загрузка (общая) → вопросы по порядку для каждого варианта.

`llm.chat(messages, max_tokens, tools=None) -> dict(content, tool_calls, usage)` — интерфейс модели:
на Kaggle это vLLM с Qwen2.5-7B-Instruct, локально в тестах — поддельная модель.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from ctxgraph.e1b import memory as M
from ctxgraph.e1b import selection as SEL
from ctxgraph.e1b import steps as ST
from ctxgraph.e1b.chat import ChatState, Limits, Turn, compress_if_needed, summary_messages

SYSTEM = (
    "Ты — ассистент, который помогает пользователю с его проектом на протяжении долгого разговора. "
    "Отвечай по-русски, по делу, не длиннее 200 слов. Если для хорошего ответа не хватает "
    "сведений, можно задать уточняющий вопрос."
)
EXPAND_TOOL = {
    "type": "function",
    "function": {
        "name": "expand",
        "description": "Показать воспоминания по пункту меню (id в квадратных скобках).",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
}
VARIANTS = (1, 2, 3, 4, 5, 6)
#: E1d этап B: 90 — V9-control (поиск по всему шагу), 91 — V9 (рукава-фразы из шагов),
#: 92 — V9 с рукавами-сущностями (описательно); 21 — V2+ «сильный RAG» (без шагов).
STEP_VARIANTS = {90: ("control", "phrase"), 91: ("sleeves", "phrase"), 92: ("sleeves", "entity")}
_CPU = threading.Lock()


BENCHMARKS = ("benchmark_e1c", "benchmark_e1b")


def project_dir(root: Path, pid: str) -> Path:
    """Проект ищется в E1c, затем в E1b (id проектов не пересекаются)."""
    for b in BENCHMARKS:
        if (root / b / "projects" / pid / "spec.yaml").exists():
            return root / b / "projects" / pid
    raise FileNotFoundError(pid)


def load_project(root: Path, pid: str) -> tuple[dict, list[dict]]:
    pdir = project_dir(root, pid)
    spec = yaml.safe_load((pdir / "spec.yaml").read_text(encoding="utf-8"))
    turns = []
    for seg in spec["segments"]:
        d = yaml.safe_load((pdir / "load" / f"{seg['id']}.yaml").read_text(encoding="utf-8"))
        for i, t in enumerate(d["turns"]):
            turns.append({"id": f"{seg['id']}.{i}", "role": t["role"], "text": t["text"].strip()})
    return spec, turns


class Session:
    def __init__(self, llm, count: Callable[[str], int], encoder, limits: Limits) -> None:
        self.llm, self.count, self.encoder, self.limits = llm, count, encoder, limits
        self.calls: list[dict] = []
        self.scorer = None  # cross-encoder для V2+ и этапа B (selection.Reranker.score)

    def summarize(self, prev: str, turns: list[Turn]) -> str:
        r = self.llm.chat(
            summary_messages(prev, turns, self.limits.summary), max_tokens=self.limits.summary
        )
        self.calls.append(
            {"purpose": "summary", "usage": r.get("usage", {}), "content": r["content"]}
        )
        return r["content"].strip()

    def add(self, state: ChatState, tid: str, role: str, text: str, origin: str) -> None:
        tok = self.count(text) + 4
        compress_if_needed(state, self.limits, self.summarize, self.count, reserve=tok)
        state.window.append(Turn(tid, role, text, origin, tok))

    def replay_load(self, turns: list[dict]) -> ChatState:
        state = ChatState()
        for t in turns:
            self.add(state, t["id"], t["role"], t["text"], "load")
        return state

    def messages(self, state: ChatState, question: str, memory: str) -> list[dict]:
        sys = SYSTEM + (
            f"\n\nКраткое содержание более ранней части разговора:\n{state.summary}"
            if state.summary
            else ""
        )
        msgs = [{"role": "system", "content": sys}]
        msgs += [{"role": t.role, "content": t.text} for t in state.window]
        user = f"{memory}\n\n---\n{question}" if memory else question
        msgs.append({"role": "user", "content": user})
        return msgs


def build_memory(variant: int, arc: M.Archive, q: str, count, budget: int, pool, seed: int):
    if variant == 1:
        return "", None
    if variant == 2:
        return M.v2_rag(arc, q, count, budget), None
    if variant == 3:
        return M.v3_graph(arc, q, count, budget), None
    if variant == 4:
        return M.v4_menu(arc, q, count, budget)
    if variant == 5:
        return M.v5_random(arc, q, count, budget, pool, seed), None
    if variant == 6:
        return M.v6_hybrid(arc, q, count, budget), None
    raise ValueError(variant)


def _ask(sess: Session, pid: str, v: int, q: dict, state: ChatState, pool: list[Turn]) -> dict:
    # сжатие до вопроса: в окне должно хватить места и для него
    compress_if_needed(
        state, sess.limits, sess.summarize, sess.count, reserve=sess.count(q["text"]) + 4
    )
    seed = int(hashlib.sha256(f"{pid}|{q['id']}|{v}".encode()).hexdigest()[:8], 16)
    selection, usage = None, []
    with _CPU:  # WordNet (NLTK) и эмбеддер не потокобезопасны; параллельны только вызовы модели
        arc = M.Archive(state.archive, sess.encoder)
        if v == 7:
            menu_text, menu = M.v7_menu(arc, q["text"], sess.count)
        elif v == 21:  # V2+ «сильный RAG»: гибрид без графа → предложения → cross-encoder
            mem = SEL.v2_plus(arc, q["text"], sess.count, sess.limits.memory, sess.scorer)[0]
            expand = None
        elif v not in STEP_VARIANTS:
            mem, expand = build_memory(
                v, arc, q["text"], sess.count, sess.limits.memory, pool, seed=seed
            )
    steps = None
    if v in STEP_VARIANTS:  # V9-control / V9: шаги мышления с поиском после каждого шага
        base = sess.messages(state, "", "")[:-1]  # система + резюме + окно
        mode, kind = STEP_VARIANTS[v]
        cfg = ST.StepsConfig(budget_total=sess.limits.memory, sleeves=kind)
        res = ST.run_steps(sess.llm, arc, q["text"], base, sess.count, sess.scorer, mode=mode, cfg=cfg)
        usage += res["usage"]
        mem, answer, expanded = res["memory"], res["answer"], []
        r = {"finish_reason": res["finish_reason"]}
        steps = {k: res[k] for k in ("steps", "rounds", "calls", "protocol")}
    if v == 7:  # шаг 1: модель видит меню с номерами и выбирает, что раскрыть (без инструментов)
        expand = None
        if menu:
            instr = M.SELECT_INSTR.format(max_pick=4)
            ask = f"{menu_text}\n\n{instr}\n\nВопрос пользователя: {q['text']}"
            r1 = sess.llm.chat(sess.messages(state, ask, ""), max_tokens=60)
            usage.append(r1.get("usage", {}))
            selection = M.parse_selection(r1.get("content", ""), menu) | {
                "menu": {n: tid for n, tid in menu.items()},
                "menu_text": menu_text,
            }
            with _CPU:
                mem = M.v7_block(arc, selection["picked"], menu, sess.count, sess.limits.memory)
            selection["picked_ids"] = [menu[n] for n in selection["picked"]]
        else:
            mem = ""
    if steps is None:
        msgs = sess.messages(state, q["text"], mem)
        tools = [EXPAND_TOOL] if v == 4 and expand else None
        expanded = []
        for _ in range(4):  # не больше 3 раскрытий + ответ
            r = sess.llm.chat(msgs, max_tokens=sess.limits.answer, tools=tools)
            usage.append(r.get("usage", {}))
            if not r.get("tool_calls"):
                break
            msgs.append(
                {"role": "assistant", "content": r.get("content") or "", "tool_calls": r["tool_calls"]}
            )
            for tc in r["tool_calls"]:
                nid = json.loads(tc["function"].get("arguments") or "{}").get("id", "")
                res = (
                    expand.get(nid, "Нет такого пункта.")
                    if len(expanded) < 3
                    else "Лимит раскрытий исчерпан."
                )
                expanded.append(nid)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": res})
            if len(expanded) >= 3:
                tools = None
        answer = r.get("content") or ""
    sess.add(state, f"{q['id']}.q", "user", q["text"], "question")
    sess.add(state, f"{q['id']}.a", "assistant", answer, "model")
    if q.get("followup"):
        sess.add(state, f"{q['id']}.f", "user", q["followup"]["text"], "followup")
    return {
        "project": pid,
        "variant": v,
        "question": q["id"],
        "question_text": q["text"],
        "type": q["type"],
        "memory": mem,
        "memory_tokens": sess.count(mem) if mem else 0,
        "expanded": expanded,
        "selection": selection,
        "steps": steps,
        "calls": len(usage),
        "answer": answer,
        "finish_reason": r.get("finish_reason"),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
        "compressions_so_far": len(state.compressions),
        "archive_turns": len(state.archive),
    }


def run_project(
    root: Path,
    pid: str,
    sess: Session,
    *,
    variants=VARIANTS,
    pool: list[Turn] | None = None,
    questions: list[str] | None = None,
    out: Path | None = None,
    parallel: bool = False,
    workers: int = 0,
) -> list[dict]:
    spec, turns = load_project(root, pid)
    t0 = time.time()
    base = sess.replay_load(turns)
    load_info = {
        "project": pid,
        "kind": "load",
        "compressions": base.compressions,
        "summary": base.summary,
        "window_tokens": sum(t.tokens for t in base.window),
        "archive_turns": len(base.archive),
        "seconds": round(time.time() - t0, 1),
    }
    records = [load_info]
    qs = [q for q in spec["questions"] if not questions or q["id"] in questions]
    lock = threading.Lock()

    def one_variant(v: int) -> list[dict]:
        state, recs = base.copy(), []
        for q in qs:
            rec = _ask(sess, pid, v, q, state, pool or [])
            recs.append(rec)
            if out:
                with lock, out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return recs

    # варианты независимы (у каждого своя копия чата); параллельно — только для батчащего сервера
    if parallel:
        with ThreadPoolExecutor(max_workers=workers or len(variants)) as ex:
            for recs in ex.map(one_variant, variants):
                records += recs
    else:
        for v in variants:
            records += one_variant(v)
    return records
