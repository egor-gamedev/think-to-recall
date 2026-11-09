"""Debug check (<5 min): where each relevant fact of the post-quantum example is lost, before/after.

    python scripts/v2_debug_loss.py   # -> reports/v2_debug_loss.md

Categories (Egor):
  (а) no node: the fact's sentence is not a fragment, or it shares no concept with the question/step;
  (б) node exists but is neither among the question's seeds/candidates nor in any sleeve mini-graph;
  (в) in a sleeve mini-graph, but its weight is too low to reach the fused candidate list;
  (г) in the fused candidates, but dropped by ranking/budget when the block was filled.
before = IDF seeds, no nested merge, no subphrase edges; after = current v2 defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v2_debug_pq as P  # noqa: E402

from ctxgraph.e1b.coverage import fact_sentences, load_turns, norm  # noqa: E402
from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Config, Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402


def evidence(spec, ft, text) -> dict[str, list[str]]:
    ev = {f: fact_sentences(spec["facts"][f]["text"], [text[t] for t in ft.get(f, [])]) for f in P.FACTS}
    return ev | {k: [v] for k, v in P.LIBS.items()}


def frags_for(g: Graph, sents: list[str]) -> list[str]:
    out = []
    for fid, txt in g.s.q("SELECT id, text FROM fragments WHERE kind='sentence'"):
        nf = norm(txt)
        for s in sents:
            ns = norm(s)
            if len(nf) > 15 and (ns[:60] in nf or nf in ns):
                out.append(fid)
                break
    return out


def analyse(name: str, cfg: Config, enc, count, spec, ft, text) -> dict:
    g = Graph(Store(), enc, config=cfg)
    for t in load_project(ROOT, P.PID)[1]:
        g.add_episode(t["text"], t["role"], "s1", eid=t["id"])
    mgs = SL.mini_graphs(g, P.STEP)
    final, info = SL.final_block(g, P.Q, mgs, count)
    tr = info["trace"]
    qrank = g.ranked_fragments(tr, 10**6)
    picked = {f for u in info["picked"] for f in u}
    cands = info["candidates"]
    bridge = {c for c, _, _ in X.concepts_of(P.Q + " " + P.STEP)} | {k[2:] for m in mgs for k in m.seeds if k.startswith("c:")}
    rows = {}
    for item, sents in evidence(spec, ft, text).items():
        fs = frags_for(g, sents)
        d = {"fragments": fs}
        if not fs:
            rows[item] = ("а", "предложение факта не стало фрагментом", d)
            continue
        if picked & set(fs):
            rows[item] = ("+", "в итоговом блоке", d)
            continue
        concepts = {r[0][2:] for f in fs for r in g.s.q("SELECT concept_id FROM mentions WHERE fragment_id=?", (f,))}
        shared = sorted(concepts & bridge)
        qr = min((qrank.index(f) + 1 for f in fs if f in qrank), default=None)
        sl = []
        for m in mgs:
            full = [k for k, _ in sorted(g.ppr(m.seeds).items(), key=lambda x: -x[1]) if k.startswith("f:")]
            r = min((full.index(f) + 1 for f in fs if f in full), default=None)
            w = max((m.nodes[f] for f in fs if f in m.nodes), default=None)
            sl.append((m.sleeve, r, w))
        d |= {"shared_concepts": shared, "question_rank": qr, "sleeves": sl,
              "concepts": sorted(concepts)[:12]}  # fmt: skip
        in_cands = [cands.index(f) + 1 for f in fs if f in cands]
        if in_cands:
            rows[item] = ("г", f"в кандидатах (место {min(in_cands)} из {len(cands)}), не влез в бюджет 600 токенов", d)
        elif any(w is not None for _, _, w in sl):
            best = max((w, s) for s, _, w in sl if w is not None)
            rows[item] = ("в", f"в мини-графе «{best[1]}», вес {best[0]:.2f} — ниже порога попадания в кандидаты", d)
        elif not shared:
            rows[item] = ("а", "фрагмент есть, но нет общего понятия с вопросом/шагом — нет моста в графе", d)
        else:
            rows[item] = ("б", "узел есть, но не дошёл ни до стартовых/кандидатов вопроса, ни до мини-графов рукавов", d)
    return {"name": name, "rows": rows, "final": final, "n_ok": sum(v[0] == "+" for v in rows.values()),
            "counts": g.s.counts()}  # fmt: skip


def main() -> None:
    from ctxgraph.e1b.backends import E5Encoder

    enc, count = E5Encoder(), P.counter()
    spec, ft, text = load_turns(ROOT, P.PID)
    rs = [analyse("до (IDF, без слияния, без subphrase)", Config(specificity="idf", merge_nested=False, subphrase=False),
                  enc, count, spec, ft, text),
          analyse("после (текущий v2)", Config(), enc, count, spec, ft, text)]  # fmt: skip
    desc = {f: spec["facts"][f]["text"] for f in P.FACTS} | dict(P.LIBS)
    L = ["# Где теряются факты (пост-квантовый пример, отладка ≤ 5 мин, без выводов о качестве)\n",
         f"Вопрос: «{P.Q}». Шаг: «{P.STEP}».\n",
         "(а) нет узла / нет моста; (б) узел есть, но не достигнут; (в) в мини-графе, но вес низкий; (г) в кандидатах, отброшен бюджетом/ранжированием; + — в итоговом блоке.\n",
         "| | " + " | ".join(r["name"] for r in rs) + " |", "|---|" + "---|" * len(rs),
         "| в итоговом блоке, из 7 | " + " | ".join(str(r["n_ok"]) for r in rs) + " |",
         "| понятий / рёбер | " + " | ".join(f"{r['counts']['concepts']} / {r['counts']['edges']}" for r in rs) + " |", ""]  # fmt: skip
    L += ["| факт | " + " | ".join(r["name"] for r in rs) + " |", "|---|" + "---|" * len(rs)]
    for item in desc:
        L.append(f"| {item}: {desc[item][:70]} | " + " | ".join(f"**{r['rows'][item][0]}** {r['rows'][item][1]}" for r in rs) + " |")
    for r in rs:
        L.append(f"\n## Подробности: {r['name']}\n")
        for item, (cat, _why, d) in r["rows"].items():
            if cat == "+":
                continue
            sl = "; ".join(f"«{s}»: место {rk}, вес в мини-графе {w if w is None else round(w, 2)}" for s, rk, w in d.get("sleeves", []))
            L.append(f"- **{item}** ({cat}) — фрагменты {d['fragments'] or '—'}; место в ранжировании по вопросу: {d.get('question_rank')}; "
                     f"общие понятия с вопросом/шагом: {d.get('shared_concepts') or '—'}; понятия фрагмента: {d.get('concepts') or '—'}; рукава: {sl or '—'}")  # fmt: skip
        L += ["\n```text", r["final"] or "(пусто)", "```"]
    (ROOT / "reports" / "v2_debug_loss.md").write_text("\n".join(L) + "\n", "utf-8")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
