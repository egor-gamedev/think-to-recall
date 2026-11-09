"""Память строится одинаково при любом PYTHONHASHSEED (порядок множеств не влияет на результат)."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import json, sys
sys.path.insert(0, r"{root}")
from pathlib import Path
from ctxgraph.e1b import memory as M, sleeves as S, v9
from ctxgraph.e1b.chat import Turn
from ctxgraph.e1b.runner import load_project
from ctxgraph.embed import HashingEncoder
from ctxgraph.text import approx_tokens
spec, turns = load_project(Path(r"{root}"), "econ-thesis")
arc = M.Archive([Turn(t["id"], t["role"], t["text"], "load", approx_tokens(t["text"])) for t in turns], HashingEncoder())
out = {{"replaces": arc.replaces}}
for q in spec["questions"][:12]:
    t = q["text"]
    out[q["id"]] = [
        M.v2_rag(arc, t, approx_tokens, 600),
        M.v3_graph(arc, t, approx_tokens, 600),
        M.v6_hybrid(arc, t, approx_tokens, 600),
        M.v7_menu(arc, t, approx_tokens)[0],
        S.v8_auto(arc, t, approx_tokens, 600)[0],
        v9.v9q(arc, t, approx_tokens, 600, anchor_cos=0.0, query_mode="clause")[0],
    ]
print(json.dumps(out, ensure_ascii=False, sort_keys=True))
"""


def _build(seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONIOENCODING="utf-8")
    code = SCRIPT.format(root=str(ROOT))
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8"
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def test_memory_identical_across_hash_seeds():
    a, b = _build("1"), _build("12345")
    assert json.loads(a) == json.loads(b)
