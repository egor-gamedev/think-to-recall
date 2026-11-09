"""E1d этап A (разведка до пре-регистрации V9, без вызовов API): полнота извлечения V9-q против V6.

    python scripts/e1d_stage_a.py   # → reports/e1d/stage_a.md, reports/e1d/stage_a.json

Архив на момент вопроса восстанавливается по записям V6 из E1c (`reports/e1c/main_final`):
разговор = реплики загрузки + вопросы, ответы V6 и follow-up по порядку; в архиве — префикс
разговора. Длина префикса ищется от `archive_turns` предыдущего вопроса (для первого — из
`*.load.json`) вверх, пока пересчитанный V6 не совпадёт с записанным блоком памяти дословно.
V9-q и V6 считаются на одном и том же архиве, бюджет 600 токенов, токены — токенизатором Qwen2.5,
эмбеддинги — multilingual-e5-base, как в E1c.

Параметры V9-q (порог якоря, режим запроса рукава) подбираются на 20 пробных вопросах — старые
вопросы E1b (не основные 200), seed 0, по средней полноте; при равенстве — строже порог и
режим «entity».
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1b import coverage as C  # noqa: E402
from ctxgraph.e1b import memory as M  # noqa: E402
from ctxgraph.e1b import v9  # noqa: E402
from ctxgraph.e1b.chat import Turn  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402

E1C = ROOT / "reports" / "e1c"
OUT = ROOT / "reports" / "e1d"
NEW = ["greenhouse-iot", "clinic-bot", "econ-thesis", "indie-game"]
OLD = ["crypto-wallet", "django-shop", "rnaseq-thesis"]
TYPES = [("recall", "вспомнить"), ("link", "связать"), ("change", "менялось"), ("clarify", "уточнить")]
BUDGET = 600
GRID = [None, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
MODES = ["entity", "clause"]


class CachedEncoder:
    """Обёртка над E5Encoder: один и тот же текст кодируется один раз."""

    def __init__(self, enc):
        self.enc, self.name = enc, enc.name
        self.p: dict[str, np.ndarray] = {}
        self.q: dict[str, np.ndarray] = {}

    def encode(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.p]
        if new:
            for t, v in zip(new, self.enc.encode(new), strict=True):
                self.p[t] = v
        return np.array([self.p[t] for t in texts])

    def one(self, text):
        if text not in self.q:
            self.q[text] = self.enc.one(text)
        return self.q[text]


def load_json(p: Path):
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]


def load_archive_start(pid: str) -> int:
    for d in ("main_A", "main_B"):
        p = E1C / d / f"{pid}.load.json"
        if p.exists():
            return json.loads(p.read_text("utf-8"))["archive_turns"]
    raise FileNotFoundError(pid)


PREFIX = OUT / "archive_prefix.json"


def reconstruct(pid: str, enc, count, known: dict) -> list[dict]:
    """Для каждого вопроса: архив (как в прогоне V6), запись V6, совпал ли пересчёт."""
    spec, turns = load_project(ROOT, pid)
    recs = {r["question"]: r for r in load_json(E1C / "main_final" / f"{pid}.jsonl") if r["variant"] == 6}
    conv = [Turn(t["id"], t["role"], t["text"], "load", count(t["text"]) + 4) for t in turns]
    lo = load_archive_start(pid)
    out = []
    for q in spec["questions"]:
        r = recs[q["id"]]
        found = None
        tries = [known[f"{pid}|{q['id']}"]] if f"{pid}|{q['id']}" in known else []
        for L in tries + list(range(lo, min(lo + 12, len(conv)) + 1)):
            arc = M.Archive(conv[:L], enc)
            if M.v6_hybrid(arc, q["text"], count, BUDGET) == r["memory"]:
                found = L
                break
        if found is None:
            arc = M.Archive(conv[:lo], enc)
        known[f"{pid}|{q['id']}"] = found or lo
        out.append({"q": q, "arc": arc, "rec": r, "L": found or lo, "match": found is not None})
        conv += [Turn(f"{q['id']}.q", "user", q["text"], "question", count(q["text"]) + 4),
                 Turn(f"{q['id']}.a", "assistant", r["answer"], "model", count(r["answer"]) + 4)]  # fmt: skip
        if q.get("followup"):
            f = q["followup"]["text"]
            conv.append(Turn(f"{q['id']}.f", "user", f, "followup", count(f) + 4))
        lo = r["archive_turns"]
    return out


def cov(item, ft, text, mem) -> dict:
    return C.anchor_coverage(item["q"]["anchors"], ft, text, mem)


def tune(items_by_pid, count) -> tuple[dict, list[str]]:
    rng = random.Random(0)
    pool = [(pid, it) for pid, its in items_by_pid.items() for it in its if it["q"].get("anchors")]
    probe = rng.sample(pool, 20)
    res = {}
    for mode in MODES:
        for th in GRID:
            s = []
            for pid, it in probe:
                _, ft, text = C.load_turns(ROOT, pid)
                mem, _ = v9.v9q(it["arc"], it["q"]["text"], count, BUDGET, anchor_cos=th, query_mode=mode)
                s.append(cov(it, ft, text, mem)["share"])
            res[(mode, th)] = float(np.mean(s))
    v6 = []
    for pid, it in probe:
        _, ft, text = C.load_turns(ROOT, pid)
        v6.append(cov(it, ft, text, M.v6_hybrid(it["arc"], it["q"]["text"], count, BUDGET))["share"])

    def key(mt):
        mode, th = mt
        return (round(res[mt], 6), th if th is not None else -1, mode == "entity")

    best = max(res, key=key)
    L = ["## Подбор параметров на 20 пробных вопросах (старые вопросы E1b, seed 0)\n",
         "Пробные: " + ", ".join(f"{pid} {it['q']['id']}" for pid, it in probe) + ".\n",
         f"V6 на пробных: средняя полнота {100 * np.mean(v6):.0f} %.\n",
         "| порог якоря | " + " | ".join(MODES) + " |", "|---|" + "---|" * len(MODES)]  # fmt: skip
    for th in GRID:
        L.append(f"| {th if th is not None else 'нет'} | " + " | ".join(f"{100 * res[(m, th)]:.0f} %" for m in MODES) + " |")  # noqa: E501
    L.append(f"\n**Выбрано:** режим запроса «{best[0]}», порог якоря {best[1]}, бонус пересечения {v9.BONUS:.4f} (1/60), "
             f"рукавов ≤ {v9.MAX_SLEEVES}, k = 8, полных фрагментов 4, бюджет {BUDGET}.\n")  # fmt: skip
    return {"mode": best[0], "anchor_cos": best[1]}, L


def boot(diffs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.array(diffs)
    m = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from ctxgraph.e1b.backends import E5Encoder

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))

    def count(s: str) -> int:
        return len(tok.encode(s).ids)

    enc = CachedEncoder(E5Encoder())
    OUT.mkdir(parents=True, exist_ok=True)
    known = json.loads(PREFIX.read_text("utf-8")) if PREFIX.exists() else {}
    items = {pid: reconstruct(pid, enc, count, known) for pid in NEW + OLD}
    PREFIX.write_text(json.dumps(known, indent=0), "utf-8")
    match = Counter((pid in NEW, it["match"]) for pid, its in items.items() for it in its)
    params, L_tune = tune({p: items[p] for p in OLD}, count)

    rows = []
    for pid in NEW:
        _, ft, text = C.load_turns(ROOT, pid)
        for it in items[pid]:
            q = it["q"]
            mem9, log = v9.v9q(it["arc"], q["text"], count, BUDGET, anchor_cos=params["anchor_cos"],
                               query_mode=params["mode"])  # fmt: skip
            mem6 = M.v6_hybrid(it["arc"], q["text"], count, BUDGET)
            c6, c9, c6r = cov(it, ft, text, mem6), cov(it, ft, text, mem9), cov(it, ft, text, it["rec"]["memory"])
            pool6 = M.v6_candidates(it["arc"], q["text"])[0]
            p6 = C.anchor_coverage(q["anchors"], ft, text, shown_ids=pool6)
            p9 = C.anchor_coverage(q["anchors"], ft, text, shown_ids=log.get("pool", []))
            rows.append({"pid": pid, "q": q["id"], "type": q["type"], "match": it["match"],
                         "v6": c6, "v9q": c9, "pool6": p6, "pool9": p9,
                         "npool6": len(pool6), "npool9": len(log.get("pool", [])), "v6_recorded": c6r, "tok6": count(mem6) if mem6 else 0,
                         "tok9": count(mem9) if mem9 else 0, "sleeves": log.get("sleeves"),
                         "n_inter": log.get("n_intersections", 0), "dropped": log.get("dropped_by_anchor", 0)})  # fmt: skip
    (OUT / "stage_a.json").write_text(json.dumps({"params": params, "rows": rows}, ensure_ascii=False, indent=1), "utf-8")

    L = ["# E1d этап A — полнота извлечения V9-q против V6 (разведка до пре-регистрации V9, без API)\n",
         f"Восстановление архива: пересчитанный V6 совпал с записанным в E1c дословно у {match[(True, True)]} из "
         f"{match[(True, True)] + match[(True, False)]} основных и {match[(False, True)]} из "
         f"{match[(False, True)] + match[(False, False)]} старых вопросов.\n"]  # fmt: skip
    L += L_tune
    L += ["## Полнота на 200 основных вопросах\n",
          "Доля опорных фактов в памяти полным фрагментом (по фактам); «все» — вопросов, где в памяти все опорные; "
          "разность — средняя по вопросам V9-q − V6, п.п., парный бутстреп 95 % (описательно).\n",
          "| тип (вопросов) | V6 | V9-q | разность [95 %] | V6 все / V9-q все | V9-q лучше / хуже / равно |",
          "|---|---|---|---|---|---|"]  # fmt: skip
    for ty, ru in TYPES + [("all", "все")]:
        rs = [r for r in rows if ty == "all" or r["type"] == ty]
        need = sum(r["v6"]["need"] for r in rs)
        n6, n9 = sum(r["v6"]["n"] for r in rs), sum(r["v9q"]["n"] for r in rs)
        d = [r["v9q"]["share"] - r["v6"]["share"] for r in rs]
        lo, hi = boot(d)
        L.append(f"| {ru} ({len(rs)}) | {100 * n6 / need:.0f} % ({n6}/{need}) | {100 * n9 / need:.0f} % ({n9}/{need}) | "
                 f"{100 * np.mean(d):+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}] | "
                 f"{sum(r['v6']['all'] for r in rs)} / {sum(r['v9q']['all'] for r in rs)} | "
                 f"{sum(x > 0 for x in d)} / {sum(x < 0 for x in d)} / {sum(x == 0 for x in d)} |")  # fmt: skip
    L += ["\n### Где теряется: опорные факты в пуле кандидатов (до отбора 4 полных фрагментов)\n",
          "| тип | пул V6 | пул V9-q | показано V6 | показано V9-q |", "|---|---|---|---|---|"]
    for ty, ru in TYPES + [("all", "все")]:
        rs = [r for r in rows if ty == "all" or r["type"] == ty]
        need = sum(r["v6"]["need"] for r in rs)
        L.append(f"| {ru} | " + " | ".join(f"{100 * sum(r[k]['n'] for r in rs) / need:.0f} %" for k in ("pool6", "pool9", "v6", "v9q")) + " |")
    L.append(f"\nРазмер пула в среднем: V6 {np.mean([r['npool6'] for r in rows]):.1f}, V9-q {np.mean([r['npool9'] for r in rows]):.1f} реплик.")
    t6 = np.mean([r["tok6"] for r in rows])
    t9 = np.mean([r["tok9"] for r in rows])
    rec = sum(r["v6_recorded"]["n"] for r in rows)
    L += [f"\nТокенов памяти в среднем: V6 {t6:.0f}, V9-q {t9:.0f} (бюджет {BUDGET}). "
          f"V6 из записей E1c: {rec} фактов в памяти (пересчитанный — {sum(r['v6']['n'] for r in rows)}).",
          f"Рукавов в среднем {np.mean([len(r['sleeves'] or []) for r in rows]):.1f}; пересечений в пуле в среднем "
          f"{np.mean([r['n_inter'] for r in rows]):.1f}; отброшено якорем в среднем {np.mean([r['dropped'] for r in rows]):.1f}."]  # fmt: skip
    (OUT / "stage_a.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
