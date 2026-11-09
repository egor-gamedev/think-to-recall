"""Quick check (<5 min, CPU): GLiNER multi-v2.1 entities on 30 Russian user turns vs current extractor.

    python scripts/arch_gliner_ru_check.py   # -> reports/arch_gliner_ru.md

Debug-only check for ARCHITECTURE_V2 part 2 (extraction). Not a benchmark: 30 turns, eyeball +
two proxies: share of fact-text lemmas covered by extracted entity lemmas (turns tagged with facts),
and time per turn.
"""

from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import text as T  # noqa: E402
from ctxgraph.e1b.coverage import load_turns  # noqa: E402
from ctxgraph.e1b.memory import _concepts  # noqa: E402

PROJECTS = ["greenhouse-iot", "clinic-bot", "econ-thesis", "indie-game"]
LABELS = ["person", "organization", "location", "software", "technology", "product", "hardware",
          "date", "quantity", "concept"]  # fmt: skip
MODEL = "urchade/gliner_multi-v2.1"
CODE = re.compile(r"```.*?```", re.S)


def sample():
    rows = []
    for pid in PROJECTS:
        spec, ft, text = load_turns(ROOT, pid)
        tag = {t: f for f, ts in ft.items() for t in ts}
        for tid, tx in text.items():
            if tid in tag:
                rows.append((pid, tid, CODE.sub(" ", tx).strip(), spec["facts"][tag[tid]]["text"]))
    return random.Random(0).sample(rows, 30)


def lemmas(xs):
    return {w for x in xs for w in T.content_lemmas(x)}


def main() -> None:
    from gliner import GLiNER

    m = GLiNER.from_pretrained(MODEL)
    rows = sample()
    L = ["# GLiNER multi-v2.1 on Russian: 30 user turns (debug check, not a benchmark)\n",
         f"Model `{MODEL}`, labels {LABELS}, threshold 0.4, CPU. Turns: random 30 fact-tagged user turns of the 4 E1c projects "
         "(seed 0), code blocks removed. Proxy = share of the fact text's content lemmas found among extracted entity lemmas.\n"]  # fmt: skip
    cov_g = cov_c = 0.0
    n_g = n_c = 0
    t_g = 0.0
    body = []
    for pid, tid, tx, fact in rows:
        t0 = time.time()
        ents = m.predict_entities(tx[:2000], LABELS, threshold=0.4)
        t_g += time.time() - t0
        g = [f"{e['text']} ({e['label']})" for e in ents]
        c = [s for _, s in _concepts(tx)]
        fl = set(T.content_lemmas(fact))
        cg = len(fl & lemmas(e["text"] for e in ents)) / max(len(fl), 1)
        cc = len(fl & lemmas(c)) / max(len(fl), 1)
        cov_g, cov_c, n_g, n_c = cov_g + cg, cov_c + cc, n_g + len(g), n_c + len(c)
        body += [f"\n## {pid} {tid}\n", f"**Turn:** {tx[:400]}{'…' if len(tx) > 400 else ''}\n",
                 f"**Fact:** {fact}\n", f"**GLiNER ({len(g)}, fact lemmas {cg:.0%}):** {'; '.join(g) or '—'}\n",
                 f"**Current extractor ({len(c)}, fact lemmas {cc:.0%}):** {'; '.join(c[:40]) or '—'}\n"]  # fmt: skip
    L += ["| | GLiNER | current (`_concepts`) |", "|---|---|---|",
          f"| entities per turn | {n_g / 30:.1f} | {n_c / 30:.1f} |",
          f"| fact-lemma coverage, mean | {cov_g / 30:.0%} | {cov_c / 30:.0%} |",
          f"| seconds per turn (CPU) | {t_g / 30:.2f} | — |", ""]  # fmt: skip
    (ROOT / "reports" / "arch_gliner_ru.md").write_text("\n".join(L + body) + "\n", "utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
