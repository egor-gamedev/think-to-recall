"""Хранилище графа на SQLite. У каждого узла и ребра — происхождение.

Вес ребра = `base` (из источника, не меняется) + `bonus` (закрепление по пользе).
`bonus` медленно затухает с полураспадом `half_life_days` и не удаляется:
забытая связь может вернуться.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, source TEXT NOT NULL,
    lang TEXT DEFAULT '', meta TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL, source TEXT NOT NULL,
    slot TEXT NOT NULL, base REAL NOT NULL, bonus REAL DEFAULT 0, uses INTEGER DEFAULT 0,
    expansions INTEGER DEFAULT 0, last_used REAL, created REAL NOT NULL, provenance TEXT DEFAULT '',
    shown INTEGER DEFAULT 0,
    PRIMARY KEY (src, dst, rel)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS shortcuts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, anchor TEXT NOT NULL, target TEXT NOT NULL,
    via TEXT NOT NULL, sig TEXT NOT NULL, sig_vec BLOB NOT NULL, weight REAL NOT NULL,
    uses INTEGER DEFAULT 1, last_used REAL NOT NULL, created REAL NOT NULL, session TEXT
);
CREATE INDEX IF NOT EXISTS shortcuts_anchor ON shortcuts(anchor);
CREATE TABLE IF NOT EXISTS fragments (
    id TEXT PRIMARY KEY, kind TEXT, source TEXT, label TEXT, text TEXT,
    session TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS mentions (
    frag_id TEXT NOT NULL, entity TEXT NOT NULL, kind TEXT NOT NULL, session TEXT NOT NULL,
    surface TEXT, PRIMARY KEY (frag_id, entity)
);
CREATE INDEX IF NOT EXISTS mentions_entity ON mentions(entity);
CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL, kind TEXT NOT NULL, node TEXT, session TEXT, meta TEXT DEFAULT '{}'
);
"""

DAY = 86400.0


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    source: str
    lang: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    rel: str
    source: str
    slot: str
    base: float
    bonus: float = 0.0
    uses: int = 0
    expansions: int = 0
    last_used: float | None = None
    provenance: str = ""
    shown: int = 0  # сколько раз ребро попадало в меню


class Store:
    def __init__(self, path: str | Path = ":memory:", *, half_life_days: float = 30.0) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(mentions)")}
        if "surface" not in cols:  # миграция старых баз
            self.db.execute("ALTER TABLE mentions ADD COLUMN surface TEXT")
        if "shown" not in {r[1] for r in self.db.execute("PRAGMA table_info(edges)")}:
            self.db.execute("ALTER TABLE edges ADD COLUMN shown INTEGER DEFAULT 0")
        self.half_life = half_life_days * DAY
        self.now = time.time  # подменяется в тестах

    # ---------- узлы и рёбра ----------
    def put_node(self, n: Node) -> None:
        self.db.execute(
            "INSERT INTO nodes(id,label,source,lang,meta) VALUES(?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET label=excluded.label",
            (n.id, n.label, n.source, n.lang, json.dumps(n.meta, ensure_ascii=False)),
        )

    def node(self, node_id: str) -> Node | None:
        r = self.db.execute(
            "SELECT id,label,source,lang,meta FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        return Node(r[0], r[1], r[2], r[3], json.loads(r[4])) if r else None

    def put_edge(self, e: Edge) -> None:
        self.db.execute(
            "INSERT INTO edges(src,dst,rel,source,slot,base,created,provenance) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(src,dst,rel) DO UPDATE SET base=MAX(base, excluded.base)",
            (e.src, e.dst, e.rel, e.source, e.slot, e.base, self.now(), e.provenance),
        )

    def _row_edge(self, r: tuple) -> Edge:
        return Edge(*r)

    _EDGE_COLS = "src,dst,rel,source,slot,base,bonus,uses,expansions,last_used,provenance,shown"

    def edges_from(self, src: str) -> list[Edge]:
        rows = self.db.execute(
            f"SELECT {self._EDGE_COLS} FROM edges WHERE src=?", (src,)
        ).fetchall()
        return [self._row_edge(r) for r in rows]

    def edges_to(self, dst: str) -> list[Edge]:
        rows = self.db.execute(
            f"SELECT {self._EDGE_COLS} FROM edges WHERE dst=?", (dst,)
        ).fetchall()
        return [self._row_edge(r) for r in rows]

    def edge_between(self, src: str, dst: str) -> Edge | None:
        r = self.db.execute(
            f"SELECT {self._EDGE_COLS} FROM edges WHERE src=? AND dst=? ORDER BY base DESC",
            (src, dst),
        ).fetchone()
        return self._row_edge(r) if r else None

    def all_edges(self) -> list[Edge]:
        return [self._row_edge(r) for r in self.db.execute(f"SELECT {self._EDGE_COLS} FROM edges")]

    def decayed(self, bonus: float, last_used: float | None) -> float:
        if not bonus or last_used is None:
            return bonus
        return bonus * 0.5 ** ((self.now() - last_used) / self.half_life)

    def weight(self, e: Edge) -> float:
        return e.base + self.decayed(e.bonus, e.last_used)

    def reinforce(self, src: str, dst: str, amount: float = 0.5) -> bool:
        e = self.edge_between(src, dst)
        if e is None:
            return False
        bonus = self.decayed(e.bonus, e.last_used) + amount
        self.db.execute(
            "UPDATE edges SET bonus=?, uses=uses+1, last_used=? WHERE src=? AND dst=? AND rel=?",
            (bonus, self.now(), e.src, e.dst, e.rel),
        )
        return True

    def count_shown(self, src: str, dst: str) -> None:
        self.db.execute("UPDATE edges SET shown=shown+1 WHERE src=? AND dst=?", (src, dst))

    def count_expansion(self, src: str, dst: str) -> None:
        self.db.execute(
            "UPDATE edges SET expansions=expansions+1 WHERE src=? AND dst=?", (src, dst)
        )

    # ---------- сокращения ----------
    def add_shortcut(
        self,
        anchor: str,
        target: str,
        via: str,
        sig: list[str],
        sig_vec: np.ndarray,
        weight: float,
        session: str,
    ) -> None:
        now = self.now()
        row = self.db.execute(
            "SELECT id, weight, last_used, sig_vec FROM shortcuts WHERE anchor=? AND target=?",
            (anchor, target),
        ).fetchall()
        for sid, w, lu, vec in row:
            if float(np.frombuffer(vec, dtype=np.float32) @ sig_vec.astype(np.float32)) > 0.9:
                w = self.decayed(w, lu) + weight
                self.db.execute(
                    "UPDATE shortcuts SET weight=?, uses=uses+1, last_used=? WHERE id=?",
                    (w, now, sid),
                )
                return
        self.db.execute(
            "INSERT INTO shortcuts(anchor,target,via,sig,sig_vec,weight,last_used,created,session) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                anchor,
                target,
                via,
                json.dumps(sig, ensure_ascii=False),
                sig_vec.astype(np.float32).tobytes(),
                weight,
                now,
                now,
                session,
            ),
        )

    def shortcuts_for(self, anchor: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT anchor,target,via,sig,sig_vec,weight,last_used,uses FROM shortcuts WHERE anchor=?",
            (anchor,),
        ).fetchall()
        return [
            {
                "anchor": a,
                "target": t,
                "via": v,
                "sig": json.loads(s),
                "sig_vec": np.frombuffer(vec, dtype=np.float32),
                "weight": self.decayed(w, lu),
                "uses": u,
            }
            for a, t, v, s, vec, w, lu, u in rows
        ]

    # ---------- опыт ----------
    def put_fragment(self, f, session: str) -> None:  # f: ingest.Fragment
        self.db.execute(
            "INSERT OR IGNORE INTO fragments(id,kind,source,label,text,session,ts) VALUES(?,?,?,?,?,?,?)",
            (f.id, f.kind, f.source, f.label, f.text, session, self.now()),
        )
        self.db.executemany(
            "INSERT OR IGNORE INTO mentions(frag_id,entity,kind,session,surface) VALUES(?,?,?,?,?)",
            # URL в опыт не пишем: это адреса, а не понятия, и могут быть приватными.
            [
                (f.id, e.canonical, e.kind, session, e.surface)
                for e in f.entities
                if e.kind != "url"
            ],
        )

    def cooccurring(
        self, entity: str, *, exclude_session: str | None, limit: int = 30
    ) -> list[tuple]:
        """Сущности, встречавшиеся с `entity` в прошлых фрагментах: (entity, kind, n_together, surface)."""
        return self.db.execute(
            "SELECT m2.entity, m2.kind, COUNT(DISTINCT m2.frag_id) c, MAX(m2.surface) FROM mentions m1 "
            "JOIN mentions m2 ON m1.frag_id=m2.frag_id AND m2.entity<>m1.entity "
            "WHERE m1.entity=? AND (? IS NULL OR m1.session<>?) "
            "GROUP BY m2.entity ORDER BY c DESC LIMIT ?",
            (entity, exclude_session, exclude_session, limit),
        ).fetchall()

    def cooccurring_sessions(
        self, entity: str, *, exclude_session: str | None, limit: int = 60
    ) -> list[tuple]:
        """Как `cooccurring`, но считает **сессии**, в которых пара встретилась в одном фрагменте:
        два файла одного проекта — одно наблюдение, а не два."""
        return self.db.execute(
            "SELECT m2.entity, m2.kind, COUNT(DISTINCT m1.session) c, MAX(m2.surface) FROM mentions m1 "
            "JOIN mentions m2 ON m1.frag_id=m2.frag_id AND m2.entity<>m1.entity "
            "WHERE m1.entity=? AND (? IS NULL OR m1.session<>?) "
            "GROUP BY m2.entity ORDER BY c DESC, m2.entity LIMIT ?",
            (entity, exclude_session, exclude_session, limit),
        ).fetchall()

    def session_freq(self, entity: str, exclude_session: str | None = None) -> int:
        return self.db.execute(
            "SELECT COUNT(DISTINCT session) FROM mentions WHERE entity=? AND (? IS NULL OR session<>?)",
            (entity, exclude_session, exclude_session),
        ).fetchone()[0]

    def n_sessions(self, exclude_session: str | None = None) -> int:
        return self.db.execute(
            "SELECT COUNT(DISTINCT session) FROM fragments WHERE (? IS NULL OR session<>?)",
            (exclude_session, exclude_session),
        ).fetchone()[0]

    def doc_freq(self, entity: str, exclude_session: str | None = None) -> int:
        return self.db.execute(
            "SELECT COUNT(DISTINCT frag_id) FROM mentions WHERE entity=? AND (? IS NULL OR session<>?)",
            (entity, exclude_session, exclude_session),
        ).fetchone()[0]

    def n_fragments(self, exclude_session: str | None = None) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM fragments WHERE (? IS NULL OR session<>?)",
            (exclude_session, exclude_session),
        ).fetchone()[0]

    def fragments_with(
        self, entity: str, exclude_session: str | None, limit: int = 3
    ) -> list[tuple]:
        return self.db.execute(
            "SELECT f.id, f.label, f.session FROM mentions m JOIN fragments f ON f.id=m.frag_id "
            "WHERE m.entity=? AND (? IS NULL OR m.session<>?) ORDER BY f.ts DESC LIMIT ?",
            (entity, exclude_session, exclude_session, limit),
        ).fetchall()

    # ---------- кэш источников ----------
    def cache_get(self, key: str):
        r = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def cache_put(self, key: str, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO cache(key,value,ts) VALUES(?,?,?)",
            (key, json.dumps(value, ensure_ascii=False), self.now()),
        )

    # ---------- события и метрики ----------
    def log(self, kind: str, node: str | None, session: str, **meta) -> None:
        self.db.execute(
            "INSERT INTO events(ts,kind,node,session,meta) VALUES(?,?,?,?,?)",
            (self.now(), kind, node, session, json.dumps(meta, ensure_ascii=False)),
        )

    def events(self, kind: str | None = None) -> list[tuple]:
        if kind:
            return self.db.execute(
                "SELECT ts,kind,node,session,meta FROM events WHERE kind=? ORDER BY ts", (kind,)
            ).fetchall()
        return self.db.execute(
            "SELECT ts,kind,node,session,meta FROM events ORDER BY ts"
        ).fetchall()

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def diversity(nodes: Iterable[str]) -> dict:
    """Разнообразие раскрытых узлов: доля уникальных и нормированная энтропия (0..1)."""
    xs = list(nodes)
    if not xs:
        return {"n": 0, "unique_ratio": 0.0, "entropy": 0.0}
    _, counts = np.unique(xs, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    hmax = np.log(len(xs)) if len(xs) > 1 else 1.0
    return {
        "n": len(xs),
        "unique_ratio": len(counts) / len(xs),
        "entropy": h / hmax if hmax else 0.0,
    }
