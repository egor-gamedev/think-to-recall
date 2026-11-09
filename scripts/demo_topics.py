"""Меню на 5 темах + раскрытие + сокращение пути. Пишет reports/data/stage1_menus.md и .json.

Без генеративной модели. Запуск: python scripts/demo_topics.py [--offline]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import build_engine  # noqa: E402
from ctxgraph.menu import render_arm  # noqa: E402
from ctxgraph.text import approx_tokens  # noqa: E402

TOPICS = {
    "рыбы": (
        "Обсуждали кормление рыб в аквариуме: живой корм, мотыль, артемия, температура воды 24 °C. "
        "Неоны и гуппи хорошо уживаются, а цихлиды агрессивны.",
        "Какие рыбы подойдут для аквариума на 100 литров начинающему?",
    ),
    "SQL-оплаты": (
        "Делали таблицу payments в PostgreSQL: idempotency_key, вебхуки от эквайера, повторные "
        "доставки уведомлений. Платёж может прийти дважды.",
        "Как хранить оплаты в SQL: таблица платежей, статусы, возвраты средств?",
    ),
    "медицина": (
        "Мама принимает варфарин, обсуждали МНО и взаимодействие с антибиотиками. Давление у неё "
        "бывает повышенным, врач назначил лизиноприл.",
        "Что можно принимать от головной боли, если давление повышенное?",
    ),
    "программирование": (
        "Писали парсер конфига parse_config на Python: YAML, валидация через pydantic, функция "
        "кэшировала прочитанные файлы в словаре.",
        "Как в Python сделать кэширование результатов функции с ограничением по памяти?",
    ),
    "история": (
        "Читали про Петра I и основание Санкт-Петербурга, Северная война со Швецией, флот на Балтике.",
        "Расскажи о причинах Крымской войны 1853 года.",
    ),
}


def reset_experience(store) -> None:
    for t in ("fragments", "mentions", "shortcuts", "events"):
        store.db.execute(f"DELETE FROM {t}")
    store.db.execute("UPDATE edges SET bonus=0, uses=0, expansions=0, shown=0, last_used=NULL")
    store.commit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / ".cache" / "demo.sqlite"))
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)

    t0 = time.perf_counter()
    eng = build_engine(args.db, offline=args.offline, seed=0)
    init_s = time.perf_counter() - t0
    reset_experience(eng.store)

    # Прошлые сессии: по одной на тему.
    for name, (past, _) in TOPICS.items():
        eng.new_session(f"past-{name}")
        eng.observe(past)

    out = {"init_s": round(init_s, 2), "encoder": eng.encoder.name, "topics": []}
    md = ["# Этап 1 — меню на 5 темах (сгенерировано scripts/demo_topics.py)\n"]
    for name, (_, current) in TOPICS.items():
        eng.new_session(f"now-{name}")
        rec = {"topic": name, "context": current}
        for depth in (0, 1):
            t = time.perf_counter()
            m = eng.menu(current, auto_depth=depth)
            ms = (time.perf_counter() - t) * 1000
            text = m.render()
            rec[f"depth{depth}"] = {
                "ms": round(ms),
                "tokens_est": approx_tokens(text),
                "arms": len(m.arms),
                "items": len(m.items()),
                "crossings": len(m.crossings),
                "menu": text,
                "arm_tokens": [approx_tokens(render_arm(a, eng.config)) for a in m.arms],
            }
        m0 = eng.menu(current, auto_depth=0)
        pick = next(
            (i for a in m0.arms for s in ("deeper", "side", "assoc") for i in a.items.get(s, [])),
            None,
        )
        if pick:
            t = time.perf_counter()
            arm = eng.expand(pick.node_id)
            rec["expand"] = {
                "node": pick.node_id,
                "label": pick.label,
                "ms": round((time.perf_counter() - t) * 1000),
                "menu": render_arm(arm, eng.config) if arm else "",
            }
        out["topics"].append(rec)
        md.append(f"## {name}\n\nКонтекст: «{current}»\n")
        for depth in (0, 1):
            d = rec[f"depth{depth}"]
            md.append(
                f"**auto_depth={depth}** — {d['ms']} мс, ~{d['tokens_est']} ток., "
                f"рукавов {d['arms']}, пунктов {d['items']}\n\n```\n{d['menu']}\n```\n"
            )
        if "expand" in rec:
            e = rec["expand"]
            md.append(
                f"**expand({e['node']})** «{e['label']}» — {e['ms']} мс\n\n```\n{e['menu']}\n```\n"
            )

    # Сокращение: путь A → B → C в одной сессии, затем похожий контекст в новой.
    ctx = "Какие рыбы подойдут для аквариума на 100 литров начинающему?"
    eng.new_session("shortcut-1")
    m = eng.menu(ctx)
    sc = {"context": ctx}
    b = next((i for a in m.arms for i in a.items.get("deeper", [])), None)
    if b:
        arm_b = eng.expand(b.node_id)
        c = next(
            (
                i
                for s in ("deeper", "side", "assoc")
                for i in (arm_b.items.get(s, []) if arm_b else [])
            ),
            None,
        )
        if c:
            sc["path"] = [m.arms[0].anchor, b.label, c.label]
            sc["mark_used"] = eng.mark_used(c.node_id)
            eng.new_session("shortcut-2")
            similar = "Посоветуйте рыб для небольшого аквариума, я новичок."
            m2 = eng.menu(similar)
            sc["similar_context"] = similar
            sc["menu_after"] = m2.render()
            sc["shortcut_shown"] = any(i.source == "shortcut" for i in m2.items())
            eng.new_session("shortcut-3")
            other = "Как приготовить рыбу в духовке с лимоном?"
            m3 = eng.menu(other)
            sc["other_context"] = other
            sc["shortcut_shown_other"] = any(i.source == "shortcut" for i in m3.items())
    out["shortcut"] = sc
    md.append("## Сокращение пути\n")
    md.append(
        f"Путь: {' → '.join(sc.get('path', []))}; mark_used: "
        f"{json.dumps(sc.get('mark_used', {}), ensure_ascii=False)}\n"
    )
    md.append(
        f"Похожий контекст «{sc.get('similar_context')}» — сокращение показано: "
        f"{sc.get('shortcut_shown')}\n\n```\n{sc.get('menu_after', '')}\n```\n"
    )
    md.append(
        f"Другой контекст «{sc.get('other_context')}» — сокращение показано: "
        f"{sc.get('shortcut_shown_other')}\n"
    )

    data = ROOT / "reports" / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "stage1_menus.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (data / "stage1_menus.md").write_text("\n".join(md), encoding="utf-8")
    eng.store.close()
    print(f"ok: {data / 'stage1_menus.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
