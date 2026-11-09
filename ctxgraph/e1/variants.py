"""Восемь вариантов E1 (пре-регистрация §4). Один прогон = одна задача × вариант × повтор."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ctxgraph import E1_CONFIG, build_engine
from ctxgraph.e1 import tasks as TK
from ctxgraph.e1.llm import Client
from ctxgraph.menu import HEADER, Arm, Item, Menu, render_arm
from ctxgraph.text import approx_tokens

VARIANTS = (1, 2, 3, 4, 5, 6, 7, 8)
AUTO_DEPTH = {3: 0, 4: 1, 5: 2, 6: 3, 7: 0}
MAX_MODEL_CALLS = 6
NO_MENU = "(no related concepts were found for this context)"


def seed_for(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


class GraphFactory:
    """Свежий граф на каждый прогон: копия базы, где есть только кэш источников."""

    def __init__(self, cache_db: Path, encoder, *, offline: bool = False) -> None:
        self.cache_db = Path(cache_db)
        self.encoder = encoder
        self.offline = offline
        self._tmp = Path(tempfile.mkdtemp(prefix="ctxgraph_e1_"))
        self._n = 0

    def engine(self, auto_depth: int = 0):
        self._n += 1
        db = self._tmp / f"run{self._n}.sqlite"
        shutil.copyfile(self.cache_db, db)
        cfg = replace(E1_CONFIG, auto_depth=auto_depth)
        return build_engine(
            db,
            encoder=self.encoder,
            offline=self.offline,
            experience=False,
            config=cfg,
            session="e1",
            seed=0,
        )


def far_ids(menu: Menu) -> set[str]:
    return {i.node_id for a in menu.arms for i in a.all_items() if i.slot == "far"}


class RandomMenus:
    """Вариант 7: та же разметка меню, узлы — случайные из графа других задач."""

    def __init__(self, pool: dict[str, list[list[str]]]) -> None:
        self.pool = pool  # task_id → [[node_id, label], ...] (меню и раскрытия этой задачи)

    def labels_for(self, task_id: str) -> list[str]:
        own = {nid for nid, _ in self.pool.get(task_id, [])}
        seen, out = set(), []
        for tid, items in sorted(self.pool.items()):
            if tid == task_id:
                continue
            for nid, label in items:
                if nid not in own and label not in seen:
                    seen.add(label)
                    out.append(label)
        return out

    def randomize(self, menu: Menu, labels: list[str], rng: random.Random) -> tuple[Menu, dict]:
        m = copy.deepcopy(menu)
        mapping: dict[str, str] = {}

        def fresh(it: Item) -> Item:
            rid = f"rnd:{len(mapping) + 1}"
            mapping[rid] = rng.choice(labels)
            return replace(it, node_id=rid, label=mapping[rid])

        def walk(arm: Arm) -> None:
            arm.items = {s: [fresh(i) for i in its] for s, its in arm.items.items()}
            for c in arm.children:
                c.anchor = f"rnd:{len(mapping) + 1}"
                mapping[c.anchor] = rng.choice(labels)
                c.anchor_label = mapping[c.anchor]
                walk(c)

        for a in m.arms:
            walk(a)
        for c in m.crossings:
            c.node_id = f"rnd:{len(mapping) + 1}"
            mapping[c.node_id] = c.label = rng.choice(labels)
        return m, mapping

    def expand(self, node_id: str, labels: list[str], rng: random.Random, cfg) -> str:
        arm = Arm(node_id, node_id, node_id, rng.choice(labels), depth=1)
        shape = {
            "deeper": ["подвид", "подкласс"],
            "side": ["обобщение", "брат (общий родитель)"],
            "assoc": ["связано"],
            "far": ["брат (общий родитель)"],
        }
        k = 0
        for slot, rels in shape.items():
            arm.items[slot] = []
            for rel in rels:
                k += 1
                arm.items[slot].append(
                    Item(f"{node_id}.{k}", rng.choice(labels), slot, rel, "random", 0, 0)
                )
        return f"{HEADER}\n{render_arm(arm, cfg)}"


def _usage_sum(calls: list[dict]) -> dict:
    return {
        "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
        "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
    }


def _call_rec(r, purpose: str) -> dict:
    return {
        "purpose": purpose,
        "usage": r.usage,
        "fingerprint": r.fingerprint,
        "ms": round(r.ms),
        "model": r.model,
    }


def _assistant_msg(r) -> dict:
    msg = {"role": "assistant", "content": r.content or ""}
    if r.tool_calls:
        msg["tool_calls"] = r.tool_calls
    if r.reasoning:
        # DeepSeek в режиме рассуждения с инструментами ждёт reasoning_content обратно в том же ходе.
        msg["reasoning_content"] = r.reasoning
    return msg


def run_variant(
    task: dict,
    variant: int,
    repeat: int,
    *,
    client: Client,
    graphs: GraphFactory,
    randoms: RandomMenus | None = None,
    v3_expanded: list[str] | None = None,
) -> dict:
    run_id = f"{task['id']}|v{variant}|r{repeat}"
    rec = {
        "run_id": run_id,
        "task": task["id"],
        "variant": variant,
        "repeat": repeat,
        "control": bool(task.get("control")),
        "calls": [],
        "expanded": [],
        "expand_calls": 0,
        "far_expanded": 0,
        "menu_tokens": None,
        "answer": None,
        "error": None,
        "ts": time.time(),
    }
    user = TK.task_prompt(task)
    system = TK.SYSTEM_BASE
    try:
        if variant == 1:
            r = client.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                purpose="answer",
                run_id=run_id,
            )
            rec["calls"].append(_call_rec(r, "answer"))
            rec["answer"] = r.content
        elif variant == 8:
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n\n{TK.SELF_LIST}"},
            ]
            r1 = client.chat(msgs, purpose="self_list", run_id=run_id)
            rec["calls"].append(_call_rec(r1, "self_list"))
            rec["self_list"] = r1.content
            msgs += [
                {"role": "assistant", "content": r1.content},
                {"role": "user", "content": TK.SELF_ANSWER},
            ]
            r2 = client.chat(msgs, purpose="answer", run_id=run_id)
            rec["calls"].append(_call_rec(r2, "answer"))
            rec["answer"] = r2.content
        else:
            eng = graphs.engine(AUTO_DEPTH.get(variant, 0))
            menu = eng.menu(TK.graph_items(task))
            rng = random.Random(seed_for(task["id"], repeat, variant))
            labels = randoms.labels_for(task["id"]) if variant == 7 and randoms else []
            if variant == 7 and labels:
                menu, _ = randoms.randomize(menu, labels, rng)
            menu_text = menu.render() or NO_MENU
            fars = far_ids(menu)
            if variant == 2:
                parts = [menu_text]
                for nid in v3_expanded or []:
                    arm = eng.expand(nid)
                    if arm is not None:
                        parts.append(render_arm(arm, eng.config))
                rec["expanded"] = list(v3_expanded or [])
                menu_text = "\n".join(parts)
            rec["menu_tokens"] = approx_tokens(menu_text)
            sys_text = f"{system}\n\n{TK.ADD_MENU}" + ("" if variant == 2 else f"\n{TK.ADD_TOOL}")
            msgs = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": f"{user}\n\nMAP OF RELATED CONCEPTS:\n{menu_text}"},
            ]
            if variant == 2:
                r = client.chat(msgs, purpose="answer", run_id=run_id)
                rec["calls"].append(_call_rec(r, "answer"))
                rec["answer"] = r.content
            else:
                for _ in range(MAX_MODEL_CALLS):
                    r = client.chat(msgs, purpose="answer", run_id=run_id, tools=[TK.EXPAND_TOOL])
                    rec["calls"].append(_call_rec(r, "tool_turn" if r.tool_calls else "answer"))
                    if not r.tool_calls:
                        rec["answer"] = r.content
                        break
                    msgs.append(_assistant_msg(r))
                    for tc in r.tool_calls:
                        rec["expand_calls"] += 1
                        nid = _node_arg(tc)
                        if len(rec["expanded"]) >= TK.EXPAND_LIMIT:
                            out = "budget exhausted"
                        elif variant == 7 and labels:
                            out = randoms.expand(nid, labels, rng, eng.config)
                            rec["expanded"].append(nid)
                        else:
                            arm = eng.expand(nid) if nid else None
                            out = (
                                f"{HEADER}\n{render_arm(arm, eng.config)}"
                                if arm is not None
                                else f"Unknown id {nid!r}: only ids shown in the map can be expanded."
                            )
                            if arm is not None:
                                rec["expanded"].append(nid)
                                rec["far_expanded"] += nid in fars
                        msgs.append(
                            {"role": "tool", "tool_call_id": tc.get("id", ""), "content": out}
                        )
                else:
                    rec["error"] = "no final answer within the call limit"
            eng.store.close()
    except Exception as e:  # noqa: BLE001 — ошибка прогона пишется в запись, раннер идёт дальше
        rec["error"] = f"{type(e).__name__}: {e}"
        if type(e).__name__ == "BudgetExceeded":
            raise
    rec["usage"] = _usage_sum(rec["calls"])
    rec["fingerprints"] = sorted({c["fingerprint"] for c in rec["calls"] if c["fingerprint"]})
    if rec["answer"] is not None and not rec["answer"].strip():
        rec["error"] = rec["error"] or "empty answer"
    return rec


def _node_arg(tc: dict) -> str:
    try:
        return str(json.loads(tc["function"].get("arguments") or "{}").get("node_id", "")).strip()
    except (json.JSONDecodeError, KeyError, AttributeError):
        return ""
