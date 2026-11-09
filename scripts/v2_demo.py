"""ctxgraph v2 demo: local server + one-page UI (ARCHITECTURE_REVIEW §3).

    python scripts/v2_demo.py            # build (first run, ~2–3 min CPU) and serve http://127.0.0.1:8765
    python scripts/v2_demo.py --rebuild  # rebuild the demo databases

Project: crypto-wallet (E1b debug set), ingested turn by turn into .cache/demo/crypto-wallet.sqlite
(append-only). The optional model answer uses the free NVIDIA API (key from .env, server side only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.e1b.runner import load_project  # noqa: E402
from ctxgraph.v2 import extract as X  # noqa: E402
from ctxgraph.v2 import sleeves as SL  # noqa: E402
from ctxgraph.v2.graph import Graph  # noqa: E402
from ctxgraph.v2.store import Store  # noqa: E402

OUT = ROOT / ".cache" / "demo"
PID = "crypto-wallet"
HTML = ROOT / "demo" / "index.html"
NV_MODEL = "mistralai/mistral-large-4-1t-a52b"
REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def counter():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json"))
    return lambda s: len(tok.encode(s).ids)


class Demo:
    def __init__(self, rebuild: bool) -> None:
        from ctxgraph.e1b.backends import E5Encoder

        OUT.mkdir(parents=True, exist_ok=True)
        db = OUT / f"{PID}.sqlite"
        if rebuild:
            for p in OUT.glob("*.sqlite*"):
                p.unlink()
        fresh = not db.exists()
        self.g = Graph(Store(db), E5Encoder())
        self.count = counter()
        self.log: list[dict] = []
        logp = OUT / f"{PID}.ingest.json"
        if fresh:
            spec, turns = load_project(ROOT, PID)
            for t in turns:
                r = self.g.add_episode(t["text"], t["role"], "s1", source=t["id"], eid=t["id"])
                self.log.append({k: r.get(k) for k in ("episode", "seq", "pieces", "new_concepts", "redactions", "changes")}
                                | {"counts": self.g.s.counts()})  # fmt: skip
            logp.write_text(json.dumps(self.log, ensure_ascii=False), "utf-8")
        else:
            self.log = json.loads(logp.read_text("utf-8"))
        self.spec = load_project(ROOT, PID)[0]
        self.reranker = None
        self.g.select("warm-up query", 600, self.count)  # first query builds the PPR adjacency (~7 s)

    # ---------- API ----------
    def meta(self) -> dict:
        eps = self.g.s.q("SELECT id, seq, role, substr(text,1,140), redactions FROM episodes ORDER BY seq")
        return {"project": PID, "episodes": [{"id": e, "seq": s, "role": r, "preview": t, "redactions": json.loads(x)}
                                             for e, s, r, t, x in eps],
                "log": self.log, "counts": self.g.s.counts(),
                "questions": [q["text"] for q in self.spec["questions"]]}  # fmt: skip

    def graph(self, upto: int, top: int) -> dict:
        s = self.g.s
        cons = s.q("SELECT c.id, c.label, c.kind, c.first_seq, COUNT(m.fragment_id) n FROM concepts c "
                   "JOIN mentions m ON m.concept_id=c.id JOIN fragments f ON f.id=m.fragment_id "
                   "WHERE f.seq<=? GROUP BY c.id ORDER BY n DESC, c.id LIMIT ?", (upto, top))  # fmt: skip
        keep = {c[0] for c in cons}
        # invalidated edges are the point of the time view: always show their ends
        inv_ends = {x for a, b in s.q("SELECT src, dst FROM edges WHERE t_invalid IS NOT NULL AND t_invalid<=?", (upto,))
                    for x in (a, b) if x.startswith("c:")} - keep  # fmt: skip
        if inv_ends:
            cons += s.q(f"SELECT id, label, kind, first_seq, n_fragments FROM concepts WHERE id IN ({','.join('?' * len(inv_ends))})",
                        tuple(sorted(inv_ends)))  # fmt: skip
            keep |= inv_ends
        nodes = [{"id": c, "label": lab, "type": "concept", "kind": k, "first_seq": fs, "n": n}
                 for c, lab, k, fs, n in cons]  # fmt: skip
        edges = []
        for a, b, rel, w, tv, ti, eps in s.q(
                "SELECT src, dst, rel, weight, t_valid, t_invalid, episodes FROM edges "
                "WHERE rel IN ('cooccur','synonym') AND t_valid<=?", (upto,)):  # fmt: skip
            if a in keep and b in keep:
                edges.append({"source": a, "target": b, "rel": rel, "w": w, "t_valid": tv,
                              "invalid": ti is not None and ti <= upto, "t_invalid": ti, "episodes": json.loads(eps)})  # fmt: skip
        # decision changes: sentence nodes + replaces edges, always shown (they are the point)
        for n_f, o_f, sq, ev in s.q("SELECT new_fragment, old_fragment, seq, evidence FROM supersedes WHERE seq<=?", (upto,)):
            for f in (n_f, o_f):
                if not any(x["id"] == f for x in nodes):
                    t = s.q("SELECT text, seq FROM fragments WHERE id=?", (f,))[0]
                    nodes.append({"id": f, "label": t[0][:60], "type": "fragment", "first_seq": t[1], "n": 1})
            edges.append({"source": n_f, "target": o_f, "rel": "replaces", "w": 1, "t_valid": sq, "invalid": False,
                          "t_invalid": None, "evidence": json.loads(ev)})  # fmt: skip
            for c in [r[0] for r in s.q("SELECT concept_id FROM mentions WHERE fragment_id=?", (o_f,))]:
                if c in keep:
                    edges.append({"source": o_f, "target": c, "rel": "contains", "w": 1, "t_valid": sq,
                                  "invalid": False, "t_invalid": None})  # fmt: skip
        return {"upto": upto, "nodes": nodes, "edges": edges, "counts_at": self.log[upto]["counts"] if upto < len(self.log) else s.counts()}

    def node(self, nid: str) -> dict:
        s = self.g.s
        if nid.startswith("f:"):
            r = s.q("SELECT f.text, f.episode_id, f.seq, e.role FROM fragments f JOIN episodes e ON e.id=f.episode_id WHERE f.id=?", (nid,))
            return {"id": nid, "type": "fragment", "text": r[0][0], "episode": r[0][1], "seq": r[0][2], "role": r[0][3]} if r else {}
        c = s.q("SELECT label, kind, first_seq, n_fragments, qid FROM concepts WHERE id=?", (nid,))
        frs = s.q("SELECT f.id, f.text, f.episode_id, f.seq FROM mentions m JOIN fragments f ON f.id=m.fragment_id "
                  "WHERE m.concept_id=? ORDER BY f.seq LIMIT 12", (nid,))  # fmt: skip
        return {"id": nid, "type": "concept", "label": c[0][0] if c else nid, "kind": c[0][1] if c else "",
                "first_seq": c[0][2] if c else None, "n_fragments": c[0][3] if c else 0, "qid": c[0][4] if c else None,
                "fragments": [{"id": a, "text": b, "episode": e, "seq": q} for a, b, e, q in frs]}  # fmt: skip

    def query(self, q: str, step: str, rerank: bool) -> dict:
        t0 = time.time()
        scorer = None
        if rerank:
            if self.reranker is None:
                from ctxgraph.e1b.selection import Reranker

                self.reranker = Reranker(revision=REV)
            scorer = self.reranker.score
        mem, info = self.g.select(q, 600, self.count, scorer=scorer, candidates=20 if rerank else 30)
        tr = info["trace"]
        t_ret = time.time() - t0
        txt = {r[0]: (r[1], r[2]) for r in self.g.s.q("SELECT id, text, episode_id FROM fragments")}
        lab = {r[0]: r[1] for r in self.g.s.q("SELECT id, label FROM concepts")}

        def name(n):
            return txt[n][0][:90] if n in txt else lab.get(n, n)

        top = sorted(tr.ppr.items(), key=lambda x: -x[1])[:25]
        sl = SL.sleeves_trace(self.g, step or q, q, self.count, 160, anchor_cos=0.76, scorer=scorer,
                              mode="hybrid")  # threshold = Config.min_score (provisional)
        for s_ in sl["sleeves"]:
            s_["nodes"] = [{"id": n, "label": name(n), "w": round(w, 3), "type": "fragment" if n.startswith("f:") else "concept"}
                           for n, w in s_["nodes"].items()]  # fmt: skip
            s_["seeds"] = [{"id": n, "label": name(n), "w": round(w, 3)} for n, w in s_["seeds"].items()]
            s_["stats"] = s_.get("stats", {})
        return {
            "query": q, "step": step or q, "seconds": round(t_ret, 2), "rerank": rerank,
            "concept_seeds": [{"id": k, "label": lab.get(k, k), "w": round(v, 3)} for k, v in tr.concept_seeds.items()],
            "fragment_seeds": [{"id": k, "label": name(k), "w": round(v, 4)} for k, v in list(tr.fragment_seeds.items())[:10]],
            "ppr_top": [{"id": k, "label": name(k), "w": round(v, 4), "type": "fragment" if k.startswith("f:") else "concept"}
                        for k, v in top],
            "memory_query_only": mem, "memory": sl["final"], "memory_tokens": self.count(sl["final"]) if sl["final"] else 0,
            "sleeves": sl["sleeves"], "intersections": {name(k): v for k, v in sl["intersections"].items()},
            "periphery": sl["periphery"],
            "memory_for_model": "\n\n".join(x for x in (sl["final"], sl["periphery"]) if x),
        }  # fmt: skip

    def answer(self, q: str, memory: str) -> dict:
        from ctxgraph.e1b.backends import NvidiaChat

        for ln in (ROOT / ".env").read_text("utf-8").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        # Mistral Large 4 on NVIDIA reasons by default and the reasoning counts toward max_tokens:
        # with 700 tokens the whole budget went to reasoning and the answer came back empty.
        llm = NvidiaChat(NV_MODEL, str(OUT / "answers_cache.jsonl"), wall=300, tries=2)
        msgs = [{"role": "system", "content": "You help a user with their software project. Answer in the user's language, briefly."},
                {"role": "user", "content": (memory + "\n\n---\n" if memory else "") + q}]  # fmt: skip
        t0 = time.time()
        r = llm.chat(msgs, max_tokens=4000)
        u = r.get("usage") or {}
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        text = r.get("content", "")
        if not text:
            text = (f"(пустой ответ: finish_reason={r.get('finish_reason')}, "
                    f"токенов на рассуждения {reasoning} из {u.get('completion_tokens')})")  # fmt: skip
        return {"model": NV_MODEL, "answer": text, "seconds": round(time.time() - t0, 1),
                "finish_reason": r.get("finish_reason"), "reasoning_tokens": reasoning}  # fmt: skip

    def privacy_check(self, text: str) -> dict:
        clean, red = X.scrub(text)
        return {"redactions": red, "clean_preview": clean[:400]}


def serve(demo: Demo, port: int) -> None:
    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    b = HTML.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(b)))
                    self.end_headers()
                    self.wfile.write(b)
                elif u.path == "/api/meta":
                    self._json(demo.meta())
                elif u.path == "/api/graph":
                    self._json(demo.graph(int(qs.get("upto", 10**6)), int(qs.get("top", 120))))
                elif u.path == "/api/node":
                    self._json(demo.node(qs["id"]))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)[:300]}, 500)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if self.path == "/api/query":
                    self._json(demo.query(body.get("q", ""), body.get("step", ""), bool(body.get("rerank"))))
                elif self.path == "/api/answer":
                    self._json(demo.answer(body.get("q", ""), body.get("memory", "")))
                elif self.path == "/api/privacy":
                    self._json(demo.privacy_check(body.get("text", "")))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)[:300]}, 500)

    print(f"ctxgraph v2 demo: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    demo = Demo(a.rebuild)
    print("graph:", demo.g.s.counts(), flush=True)
    if not a.build_only:
        serve(demo, a.port)


if __name__ == "__main__":
    main()
