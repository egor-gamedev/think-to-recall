"""E1c — разведка (не предрегистрировано) по готовым ответам, без вызовов API.

    python scripts/e1c_explore.py   # → reports/e1c/main_final/explore.md, reports/e1c_link_no_sample.md

1. «Менялось», ошибки V6 (и V2): была ли в блоке памяти устаревшая реплика (помеченная в загрузке
   фактом, который заменён по эталонной цепочке `replaces`), и повторил ли ответ её как действующую
   (судья: stale = да).
2. «Связать»: сколько опорных фактов было в блоке памяти V2 и V6 (полным фрагментом), и сколько
   из них ответ учёл (судья: anchors).
3. 10 случайных вердиктов «нет» по «связать» у V6 — на ручную проверку строгости рубрики.

Ограничение: «в контексте» = в блоке памяти. Окно последних реплик и резюме в записях не сохранены;
к моменту вопросов загрузка уже сжата в резюме, так что факт мог быть и в резюме.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_e1c as E  # noqa: E402

from ctxgraph.e1b.judge import load_spec  # noqa: E402

MAIN = ROOT / "reports" / "e1c" / "main_final"


def norm(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    return re.sub(r"\W+", " ", s.lower()).strip()


def load_turns(pid: str) -> tuple[dict, dict[str, list[str]], dict[str, str]]:
    """facts → реплики загрузки, где факт помечен; id реплики → текст."""
    spec = load_spec(ROOT, pid)
    fact_turns, text = defaultdict(list), {}
    for seg in spec["segments"]:
        d = yaml.safe_load(
            (ROOT / "benchmark_e1c" / "projects" / pid / "load" / f"{seg['id']}.yaml").read_text(
                "utf-8"
            )
        )
        for i, t in enumerate(d["turns"]):
            tid = f"{seg['id']}.{i}"
            text[tid] = t["text"]
            for f in t.get("facts") or []:
                fact_turns[f].append(tid)
    return spec, fact_turns, text


def in_memory(turn_text: str, memory: str) -> str:
    """'full' — реплика есть полным фрагментом; 'label' — только ярлыком; '' — нет."""
    key = norm(turn_text)[:60]
    if not key:
        return ""
    full = "\n".join(ln for ln in memory.splitlines() if ln.startswith("- ["))
    if key[:50] in norm(full):
        return "full"
    if key[:25] in norm(memory):
        return "label"
    return ""


def ancestors(facts: dict, fid: str) -> list[str]:
    out, cur = [], fid
    while "replaces" in facts.get(cur, {}):
        cur = facts[cur]["replaces"]
        out.append(cur)
    return out


def main() -> None:
    A, J = E.answers(E.NEW), E.verdicts("deepseek")
    L = ["# E1c — разведка по готовым ответам (не предрегистрировано, без вызовов API)\n",
         "«В контексте» — в блоке памяти варианта (окно последних реплик и резюме в записях не сохранены).\n"]  # fmt: skip
    # ---------- 1. «менялось» ----------
    stats = {v: Counter() for v in (2, 6)}
    examples = []
    for pid in E.NEW:
        spec, ft, text = load_turns(pid)
        facts = spec["facts"]
        for q in spec["questions"]:
            if q["type"] != "change":
                continue
            old = {a for f in q["anchors"] for a in ancestors(facts, f)}
            old_turns = {t for f in old for t in ft.get(f, [])}
            for v in (2, 6):
                k = f"{pid}|{q['id']}|{v}"
                r, j = A[k], J[k]
                where = {in_memory(text[t], r["memory"]) for t in old_turns} - {""}
                st = stats[v]
                st["вопросов"] += 1
                inf = bool(j["inference"])
                st["вывод да" if inf else "вывод нет"] += 1
                if inf:
                    continue
                in_ctx = bool(where)
                marked = "позже заменено" in r["memory"]
                stale = bool(j["stale"])
                st["устаревшее в памяти"] += in_ctx
                st["  из них полным фрагментом"] += "full" in where
                st["  из них с пометкой «позже заменено»"] += in_ctx and marked
                st["ответ опёрся на устаревшее (судья)"] += stale
                st["устаревшее в памяти И повторено"] += in_ctx and stale
                st["повторено без устаревшего в памяти"] += stale and not in_ctx
                if v == 6 and in_ctx and stale and len(examples) < 3:
                    examples.append((pid, q, r["answer"][:260], j["why"][:200]))
    L.append("## 1. «Менялось» (60 вопросов): ошибки V6 и V2 — разведка\n")
    list(stats[6].keys() | stats[2].keys())
    order = ["вопросов", "вывод да", "вывод нет", "устаревшее в памяти", "  из них полным фрагментом",
             "  из них с пометкой «позже заменено»", "ответ опёрся на устаревшее (судья)",
             "устаревшее в памяти И повторено", "повторено без устаревшего в памяти"]  # fmt: skip
    L += ["| (среди ответов «нет», кроме первых трёх строк) | V6 | V2 |", "|---|---|---|"]
    L += [f"| {k} | {stats[6][k]} | {stats[2][k]} |" for k in order if True]
    for pid, q, ans, why in examples:
        L.append(f"\n- **{pid} {q['id']}** «{q['text']}» → V6: «{ans}…» · судья: {why}")
    # ---------- 2. «связать» ----------
    L.append("\n## 2. «Связать» (60 вопросов): опорные факты в памяти и в ответе — разведка\n")
    grid = {v: Counter() for v in (2, 6)}
    used_when = {v: Counter() for v in (2, 6)}
    for pid in E.NEW:
        spec, ft, text = load_turns(pid)
        for q in spec["questions"]:
            if q["type"] != "link":
                continue
            for v in (2, 6):
                k = f"{pid}|{q['id']}|{v}"
                r, j = A[k], J[k]
                inmem = [
                    f
                    for f in q["anchors"]
                    if any(in_memory(text[t], r["memory"]) == "full" for t in ft.get(f, []))
                ]
                n_need = len(q["anchors"])
                used = [f for f, ok in j["anchors"].items() if ok]
                grid[v][(len(inmem), n_need)] += 1
                used_when[v]["фактов нужно"] += n_need
                used_when[v]["в памяти"] += len(inmem)
                used_when[v]["учтено ответом"] += len(used)
                used_when[v]["учтено из тех, что в памяти"] += len(set(used) & set(inmem))
                used_when[v]["учтено, хотя в памяти нет"] += len(set(used) - set(inmem))
                used_when[v]["все нужные в памяти"] += len(inmem) == n_need
                used_when[v]["все нужные в памяти и вывод да"] += len(inmem) == n_need and bool(
                    j["inference"]
                )
                used_when[v]["вывод да"] += bool(j["inference"])
    for v in (6, 2):
        u = used_when[v]
        L.append(f"**V{v}:** опорных фактов нужно {u['фактов нужно']}, в памяти полным фрагментом {u['в памяти']} "
                 f"({100 * u['в памяти'] / u['фактов нужно']:.0f} %); ответ учёл {u['учтено ответом']} — из тех, что в памяти, "
                 f"{u['учтено из тех, что в памяти']} из {u['в памяти']}; учёл без памяти (из окна/резюме/своего) {u['учтено, хотя в памяти нет']}. "
                 f"Все нужные факты в памяти — у {u['все нужные в памяти']} вопросов из 60, из них «вывод да» — {u['все нужные в памяти и вывод да']}; всего «вывод да» — {u['вывод да']}.\n")  # fmt: skip
    for v in (6, 2):
        dist = Counter()
        for (got, need), n in grid[v].items():
            dist[f"{got} из {need}"] += n
        L.append(
            f"V{v}, сколько опорных фактов было в памяти (из нужных): "
            + ", ".join(f"{k}: {n}" for k, n in sorted(dist.items()))
        )
    (MAIN / "explore.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    # ---------- 3. выборка «нет» по «связать», V6 ----------
    nos = [
        k for k, j in J.items() if k.endswith("|6") and j["type"] == "link" and not j["inference"]
    ]
    pick = sorted(random.Random(0).sample(nos, 10))
    M = ["# «Связать», V6: 10 случайных вердиктов «нет» (seed 0) — на ручную проверку строгости рубрики e1c.2\n",
         f"Всего вердиктов «нет» у V6 на «связать»: {len(nos)} из 60. Для каждого: согласен ли с «нет» (да/нет) и почему.\n"]  # fmt: skip
    for i, k in enumerate(pick, 1):
        pid, qid, _ = k.split("|")
        spec = load_spec(ROOT, pid)
        q = next(x for x in spec["questions"] if x["id"] == qid)
        anchors = "; ".join(f"{a}: {spec['facts'][a]['text']}" for a in q["anchors"])
        j = J[k]
        M += [f"\n## L{i:02d} · {pid} {qid}\n", f"**Вопрос:** {q['text']}\n", f"**Опорные факты:** {anchors}\n",
              f"**Эталон:** {q['expected']}\n", f"**Ответ V6:**\n\n{A[k]['answer']}\n",
              f"**Судья (DeepSeek, e1c.2):** обязательное условие — {j.get('mandatory', '—')}; учтённые факты — "
              + ", ".join(f"{a} {'да' if ok else 'нет'}" for a, ok in j["anchors"].items()) + f"\n\n{j['why']}\n",
              "**Ваша оценка:** согласен с «нет»? ___\n"]  # fmt: skip
    (ROOT / "reports" / "e1c_link_no_sample.md").write_text("\n".join(M), encoding="utf-8")
    print("\nвыгрузка:", ROOT / "reports" / "e1c_link_no_sample.md")


if __name__ == "__main__":
    main()
