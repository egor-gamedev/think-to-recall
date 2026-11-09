"""Debug check (<5 min, CPU): hub flattening and duplicate concepts on crypto-wallet, before/after.

    python scripts/v2_debug_pq.py   # -> reports/v2_debug_postquantum.md

before = first demo config (IDF concept seeds, no nested-phrase merge);
after  = node specificity 1/|fragments| (HippoRAG 1) + nested key phrases merged into the fullest form.
Question «Стоит ли добавлять пост-квантовые кошельки?», thinking step with 2–3 topics (Egor's example).
Relevant load facts are picked by hand (networks, key storage, signing libraries, deadline, team size);
we only check which of them reach the memory block / periphery. Debug only — no quality claims.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1b.coverage import fact_coverage, load_turns, norm  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

PID = "crypto-wallet"
Q = "Стоит ли добавлять пост-квантовые кошельки?"
STEP = ("Пост-квантовые подписи пока поддерживают не все сети. У нас Ethereum и Solana, ключи хранятся в "
        "Secure Enclave. Команда маленькая, релиз скоро.")  # fmt: skip
FACTS = ["F03", "F03r", "F02", "F07", "F06"]  # networks, Solana refusal, keys on device, Secure Enclave, team+deadline
LIBS = {  # relevant sentences without a fact id in the spec (signing / key libraries)
    "viem (S1.9)": "Библиотека для работы с сетью — `viem`",
    "@scure/bip39 (S2.1)": "Генерацию делайте через `@scure/bip39`",
}


def counter():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))
    return lambda s: len(tok.encode(s).ids)


def hits(spec, ft, text, block: str) -> dict:
    fc = fact_coverage(FACTS, spec["facts"], ft, text, block)
    out = {f: bool(fc["evidence"][f]["hit"]) for f in FACTS}
    nb = norm(block)
    out |= {k: norm(v) in nb for k, v in LIBS.items()}
    return out


def run(name: str, cfg: Config, enc, count, spec, ft, text) -> dict:
    t0 = time.time()
    g = Graph(Store(), enc, config=cfg)
    for t in load_project(ROOT, PID)[1]:
        g.add_episode(t["text"], t["role"], "s1", eid=t["id"])
    build = time.time() - t0
    mem, info = g.select(Q, 600, count)
    tr = SL.sleeves_trace(g, STEP, Q, count, 200, anchor_cos=0.76)
    lab = {r[0]: r[1] for r in g.s.q("SELECT id, label FROM concepts")}
    txt = {r[0]: r[1] for r in g.s.q("SELECT id, text FROM fragments")}
    dup = sorted(r[0][2:] for r in g.s.q("SELECT id FROM concepts WHERE id LIKE '%создани%'"))
    return {"name": name, "build_s": round(build, 1), "counts": g.s.counts(), "memory": tr["final"], "memory_q": mem, "trace": tr,
            "concept_seeds": {lab.get(k, k): round(v, 4) for k, v in info["trace"].concept_seeds.items()},
            "hits_memory": hits(spec, ft, text, tr["final"]), "hits_q": hits(spec, ft, text, mem),
            "hits_periphery": hits(spec, ft, text, tr["periphery"]),
            "lab": lab, "txt": txt, "dup": dup}  # fmt: skip


def name_of(r, n):
    return r["txt"].get(n, r["lab"].get(n, n))[:70]


def main() -> None:
    from ctxgraph.e1b.backends import E5Encoder

    enc, count = E5Encoder(), counter()
    spec, ft, text = load_turns(ROOT, PID)
    rs = [run("до (IDF, без слияния)", Config(specificity="idf", merge_nested=False), enc, count, spec, ft, text),
          run("после (1/|фрагменты|, слияние вложенных)", Config(), enc, count, spec, ft, text)]  # fmt: skip
    b, a = rs
    L = ["# Отладка v2: хабы и дубли понятий на crypto-wallet (≤ 5 мин, без выводов о качестве)\n",
         f"Вопрос: «{Q}»\n", f"Текст шага: «{STEP}»\n",
         "## Размер графа\n", "| | понятий | рёбер | упоминаний |", "|---|---|---|---|"]  # fmt: skip
    for r in rs:
        c = r["counts"]
        L.append(f"| {r['name']} | {c['concepts']} | {c['edges']} | {c['mentions']} |")
    dc = 1 - a["counts"]["concepts"] / b["counts"]["concepts"]
    de = 1 - a["counts"]["edges"] / b["counts"]["edges"]
    L += [f"\nПонятий меньше на {dc:.0%}, рёбер на {de:.0%}.\n",
          f"Понятия с «создани…»: до — {len(b['dup'])} ({', '.join(b['dup'][:12])}); после — {len(a['dup'])} ({', '.join(a['dup'][:12])}).\n"]  # fmt: skip
    for r in rs:
        L += [f"## {r['name']}\n", f"Стартовые веса понятий вопроса: {r['concept_seeds'] or '—'}\n",
              "### Рукава из текста шага: мини-графы (без стартовых узлов) и разброс весов\n"]  # fmt: skip
        for s in r["trace"]["sleeves"]:
            st = s["stats"]
            top = "; ".join(f"{name_of(r, n)} {w:.2f}" for n, w in list(s["nodes"].items())[:6])
            L.append(f"- **«{s['phrase']}»** — разброс: max/min {st.get('max_min')}, CV {st.get('cv')}, "
                     f"доля массы PPR в top-15 {st.get('mass')}. Узлы: {top}")  # fmt: skip
        inter = r["trace"]["intersections"]
        L.append("\n**Пересечения:** " + ("; ".join(f"{name_of(r, n)} ← {' + '.join(sorted(set(v)))}" for n, v in inter.items()) or "нет"))
        L += ["\n### Блок памяти для модели\n", "```text", r["memory"] or "(пусто)", "```",
              "\n### Периферия рукавов\n", "```text", r["trace"]["periphery"] or "(пусто)", "```\n"]  # fmt: skip
    L += ["## Проверка смысла: относящиеся к делу факты загрузки\n",
          "| факт | " + " | ".join(f"только вопрос: {r['name']}" for r in rs) + " | "
          + " | ".join(f"итоговый блок: {r['name']}" for r in rs) + " | " + " | ".join(f"периферия: {r['name']}" for r in rs) + " |",
          "|---|" + "---|" * 6]  # fmt: skip
    desc = {f: spec["facts"][f]["text"] for f in FACTS} | {k: v for k, v in LIBS.items()}
    for f in FACTS + list(LIBS):
        L.append(f"| {f}: {desc[f][:80]} | " + " | ".join("да" if r["hits_q"][f] else "—" for r in rs) + " | "
                 + " | ".join("да" if r["hits_memory"][f] else "—" for r in rs) + " | "
                 + " | ".join("да" if r["hits_periphery"][f] else "—" for r in rs) + " |")  # fmt: skip
    for r in rs:
        L.append(f"\n{r['name']}: только по вопросу {sum(r['hits_q'].values())} из {len(desc)}, "
                 f"в итоговом блоке {sum(r['hits_memory'].values())}, "
                 f"в периферии {sum(r['hits_periphery'].values())}; построение {r['build_s']} с.")  # fmt: skip
    (ROOT / "reports" / "v2_debug_postquantum.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
