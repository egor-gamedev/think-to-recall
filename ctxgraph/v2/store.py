"""Personal-layer storage (ARCHITECTURE_V2 §2, §3.5): one SQLite file per user, append-only.

Rules enforced here:
- no `DELETE` and no table rebuilds; rows are inserted, counters and validity fields are updated;
- edges are bi-temporal (Graphiti): `t_valid` / `t_invalid` are dialogue times (episode order),
  `t_created` / `t_expired` are wall-clock write times;
- every fragment points to its episode, every mention to its fragment, every edge to its episodes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL, session_id TEXT NOT NULL, role TEXT NOT NULL,
    source TEXT DEFAULT '', text TEXT NOT NULL, redactions TEXT DEFAULT '[]',
    t_created REAL NOT NULL, t_expired REAL
);
CREATE TABLE IF NOT EXISTS fragments (
    id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, idx INTEGER NOT NULL, seq INTEGER NOT NULL,
    text TEXT NOT NULL, lang TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'sentence', emb BLOB
);
CREATE INDEX IF NOT EXISTS fragments_episode ON fragments(episode_id);
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, label_en TEXT, lang TEXT NOT NULL, kind TEXT NOT NULL,
    qid TEXT, ili TEXT, emb BLOB, first_seq INTEGER NOT NULL, n_fragments INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS mentions (
    fragment_id TEXT NOT NULL, concept_id TEXT NOT NULL, surface TEXT, extractor TEXT NOT NULL,
    PRIMARY KEY (fragment_id, concept_id)
);
CREATE INDEX IF NOT EXISTS mentions_concept ON mentions(concept_id);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL, weight REAL NOT NULL,
    t_valid INTEGER NOT NULL, t_invalid INTEGER, t_created REAL NOT NULL, t_expired REAL,
    episodes TEXT NOT NULL DEFAULT '[]', extractor TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS supersedes (
    new_fragment TEXT NOT NULL, old_fragment TEXT NOT NULL, seq INTEGER NOT NULL,
    detector TEXT NOT NULL, evidence TEXT DEFAULT '{}', PRIMARY KEY (new_fragment, old_fragment)
);
CREATE TABLE IF NOT EXISTS usage (
    target TEXT NOT NULL, session_id TEXT NOT NULL, ts REAL NOT NULL, kind TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_fragments USING fts5(id UNINDEXED, text);
"""


def _vec(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32)


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.now = time.time  # replaced in tests

    # ---------- writes (append-only) ----------
    def next_seq(self) -> int:
        return (self.db.execute("SELECT COALESCE(MAX(seq), -1) FROM episodes").fetchone()[0]) + 1

    def add_episode(self, eid: str, seq: int, session_id: str, role: str, source: str, text: str,
                    redactions: list[dict]) -> bool:  # fmt: skip
        cur = self.db.execute(
            "INSERT OR IGNORE INTO episodes(id,seq,session_id,role,source,text,redactions,t_created) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (eid, seq, session_id, role, source, text, json.dumps(redactions), self.now()),
        )
        return cur.rowcount == 1

    def add_fragment(self, fid: str, episode_id: str, idx: int, seq: int, text: str, lang: str,
                     kind: str, emb: np.ndarray | None, fts_text: str = "") -> None:  # fmt: skip
        blob = None if emb is None else np.asarray(emb, dtype=np.float32).tobytes()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO fragments(id,episode_id,idx,seq,text,lang,kind,emb) VALUES(?,?,?,?,?,?,?,?)",
            (fid, episode_id, idx, seq, text, lang, kind, blob),
        )
        if cur.rowcount == 1 and kind == "sentence":
            # BM25 over lemmas: Russian inflection would defeat plain token matching
            self.db.execute("INSERT INTO fts_fragments(id,text) VALUES(?,?)", (fid, fts_text or text))

    def upsert_concept(self, cid: str, label: str, lang: str, kind: str, seq: int,
                       emb: np.ndarray | None) -> bool:  # fmt: skip
        blob = None if emb is None else np.asarray(emb, dtype=np.float32).tobytes()
        cur = self.db.execute(
            "INSERT OR IGNORE INTO concepts(id,label,lang,kind,emb,first_seq) VALUES(?,?,?,?,?,?)",
            (cid, label, lang, kind, blob, seq),
        )
        return cur.rowcount == 1

    def add_mention(self, fid: str, cid: str, surface: str, extractor: str) -> None:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO mentions(fragment_id,concept_id,surface,extractor) VALUES(?,?,?,?)",
            (fid, cid, surface, extractor),
        )
        if cur.rowcount == 1:
            self.db.execute("UPDATE concepts SET n_fragments=n_fragments+1 WHERE id=?", (cid,))

    def add_edge(self, src: str, dst: str, rel: str, weight: float, seq: int, episode: str,
                 extractor: str, *, accumulate: bool = True) -> None:  # fmt: skip
        """Insert an edge or merge into the existing one (LightRAG-style union): weight grows if
        `accumulate`, the episode is appended to provenance. Invalidated edges are not revived."""
        r = self.db.execute(
            "SELECT weight, episodes, t_invalid FROM edges WHERE src=? AND dst=? AND rel=?", (src, dst, rel)
        ).fetchone()
        if r is None:
            self.db.execute(
                "INSERT INTO edges(src,dst,rel,weight,t_valid,t_created,episodes,extractor) VALUES(?,?,?,?,?,?,?,?)",
                (src, dst, rel, weight, seq, self.now(), json.dumps([episode]), extractor),
            )
            return
        eps = json.loads(r[1])
        if episode in eps:
            return
        eps.append(episode)
        w = r[0] + weight if accumulate else max(r[0], weight)
        self.db.execute(
            "UPDATE edges SET weight=?, episodes=? WHERE src=? AND dst=? AND rel=?",
            (w, json.dumps(eps), src, dst, rel),
        )

    def invalidate_edge(self, src: str, dst: str, rel: str, seq: int) -> None:
        self.db.execute(
            "UPDATE edges SET t_invalid=? WHERE src=? AND dst=? AND rel=? AND t_invalid IS NULL",
            (seq, src, dst, rel),
        )

    def add_supersedes(self, new_f: str, old_f: str, seq: int, detector: str, evidence: dict) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO supersedes(new_fragment,old_fragment,seq,detector,evidence) VALUES(?,?,?,?,?)",
            (new_f, old_f, seq, detector, json.dumps(evidence, ensure_ascii=False)),
        )

    def record_usage(self, target: str, session_id: str, kind: str) -> None:
        """Consolidation interface (ARCHITECTURE_V2 §3.10): stored, not used by retrieval yet."""
        self.db.execute("INSERT INTO usage(target,session_id,ts,kind) VALUES(?,?,?,?)",
                        (target, session_id, self.now(), kind))  # fmt: skip

    def commit(self) -> None:
        self.db.commit()

    # ---------- reads ----------
    def q(self, sql: str, args: tuple = ()) -> list[tuple]:
        return self.db.execute(sql, args).fetchall()

    def counts(self) -> dict:
        tables = ("episodes", "fragments", "concepts", "mentions", "edges", "supersedes")
        out = {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
        out["edges_invalid"] = self.db.execute("SELECT COUNT(*) FROM edges WHERE t_invalid IS NOT NULL").fetchone()[0]
        return out

    def concept_vecs(self) -> tuple[list[str], np.ndarray]:
        rows = self.q("SELECT id, emb FROM concepts WHERE emb IS NOT NULL ORDER BY rowid")
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        return [r[0] for r in rows], np.stack([_vec(r[1]) for r in rows])

    def fragment_vecs(self, upto_seq: int | None = None) -> tuple[list[str], np.ndarray]:
        sql = "SELECT id, emb FROM fragments WHERE emb IS NOT NULL"
        args: tuple = ()
        if upto_seq is not None:
            sql += " AND seq<=?"
            args = (upto_seq,)
        rows = self.q(sql + " ORDER BY rowid", args)
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        return [r[0] for r in rows], np.stack([_vec(r[1]) for r in rows])

    def bm25(self, query_terms: list[str], limit: int = 50) -> list[tuple[str, float]]:
        terms = [t.replace('"', "") for t in query_terms if t.strip()]
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.q("SELECT id, bm25(fts_fragments) FROM fts_fragments WHERE fts_fragments MATCH ? "
                          "ORDER BY bm25(fts_fragments) LIMIT ?", (match, limit))  # fmt: skip
        except sqlite3.OperationalError:
            return []
        return [(r[0], -r[1]) for r in rows]
