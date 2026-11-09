"""ConceptNet 5: ассоциации (RelatedTo, UsedFor, PartOf, ...). Публичный API, кэш в хранилище.

На 2026-10-18 api.conceptnet.io отвечает 502; источник отключается сам после серии ошибок,
меню собирается без него. Дамп (~475 МБ) не скачивается без решения Егора.
"""

from __future__ import annotations

import logging

import requests

from ctxgraph.sources.base import Link, Source
from ctxgraph.store import Node, Store

log = logging.getLogger(__name__)

API = "https://api.conceptnet.io"
REL_RU = {
    "RelatedTo": "связано",
    "UsedFor": "используется для",
    "PartOf": "часть",
    "HasA": "имеет",
    "CapableOf": "может",
    "AtLocation": "находится в",
    "Causes": "вызывает",
    "HasProperty": "свойство",
    "IsA": "является",
    "HasSubevent": "включает",
    "MotivatedByGoal": "ради",
    "Desires": "хочет",
    "CreatedBy": "создаётся",
    "MadeOf": "сделано из",
    "ReceivesAction": "с ним делают",
}
SLOT = {"IsA": "side", "PartOf": "assoc", "HasA": "deeper"}


def _norm(term: str) -> str:
    return term.strip().lower().replace(" ", "_")


class ConceptNetSource(Source):
    name = "conceptnet"
    prefix = "cn:"

    def __init__(self, store: Store, *, timeout: float = 10.0, offline: bool = False) -> None:
        self.store = store
        self.timeout = timeout
        self.offline = offline
        self._failures = 0

    @property
    def available(self) -> bool:
        return not self.offline and self._failures < 2

    def link(self, term: str, lang: str) -> list[Node]:
        t = _norm(term)
        return [Node(f"cn:/c/{lang}/{t}", term, self.name, lang)] if t else []

    def neighbors(self, node: Node) -> list[Link]:
        uri = node.id[3:]
        key = f"cn:nb:{uri}"
        edges = self.store.cache_get(key)
        if edges is None:
            if not self.available:
                return []
            try:
                r = requests.get(f"{API}{uri}", params={"limit": 50}, timeout=self.timeout)
                r.raise_for_status()
                edges = r.json().get("edges", [])
            except (requests.RequestException, ValueError) as e:
                self._failures += 1
                log.warning("conceptnet unavailable: %s", e)
                return []
            self.store.cache_put(key, edges)
        lang = uri.split("/")[2] if uri.count("/") >= 3 else ""
        out: list[Link] = []
        for e in edges:
            rel = e.get("rel", {}).get("label", "")
            if rel not in REL_RU:
                continue
            other = e["end"] if e["start"]["@id"].startswith(uri) else e["start"]
            if other.get("language") not in (lang, "en", "ru"):
                continue
            w = min(1.0, float(e.get("weight", 1.0)) / 4.0)
            n = Node(
                f"cn:{other['@id']}",
                other.get("label", other["@id"]),
                self.name,
                other.get("language", ""),
            )
            out.append(Link(n, REL_RU[rel], SLOT.get(rel, "assoc"), w, self.name))
        return out
