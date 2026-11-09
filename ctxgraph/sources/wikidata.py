"""Викиданные: подклассы, таксоны, части (глубже), родители и братья (в сторону),
«использует / применяется для / аспект» и т.п. (ассоциации). SPARQL + поиск сущностей, всё кэшируется.
"""

from __future__ import annotations

import logging
import time

import requests

from ctxgraph.sources.base import Link, Source
from ctxgraph.store import Node, Store

log = logging.getLogger(__name__)

SPARQL = "https://query.wikidata.org/sparql"
API = "https://www.wikidata.org/w/api.php"
MIN_INTERVAL = 0.3  # с между живыми запросами
COOLDOWN = 60  # с паузы после 5 сбоев подряд
UA = "ctxgraph/0.1 (research prototype; contextual graph experiment)"

DEEPER = {"P279": "подкласс", "P171": "дочерний таксон", "P361": "часть", "P31": "экземпляр"}
UP = {"P279": "обобщение", "P171": "родительский таксон", "P361": "входит в"}
ASSOC_OUT = {
    "P2283": "использует",
    "P366": "применяется для",
    "P1269": "аспект чего",
    "P527": "состоит из",
    "P1552": "свойство",
    "P2578": "изучает",
    "P2579": "изучается в",
    "P737": "под влиянием",
    "P1542": "приводит к",
    "P828": "причина",
    "P461": "противоположность",
    "P1535": "используется в",
}
ASSOC_IN = {
    "P2283": "используется в",
    "P366": "служит для этого",
    "P1269": "аспект",
    "P2579": "наука о",
}

#: Типы, к которым термин из контекста привязывать нельзя без признаков в самом контексте:
#: человек, организация, научная статья, статья, патент, литературное произведение, книга,
#: фильм, альбом, музыкальное произведение, телесериал, телепрограмма.
BANNED_TYPES = (
    "Q5",
    "Q43229",
    "Q13442814",
    "Q191067",
    "Q253623",
    "Q7725634",
    "Q571",
    "Q11424",
    "Q482994",
    "Q2188189",
    "Q5398426",
    "Q15416",
)

#: Описания, по которым кандидат при привязке отбрасывается.
JUNK = (
    "страница значений",
    "disambiguation",
    "научная статья",
    "scholarly article",
    "фамилия",
    "family name",
    "given name",
    "мужское имя",
    "женское имя",
    "категория",
    "category",
    "шаблон",
    "template",
    "статья в",
    "article",
    "album",
    "альбом",
    "song",
    "песня",
    "film",
    "фильм",
)


def _q(node_id: str) -> str:
    return node_id.split(":", 1)[1]


class WikidataSource(Source):
    name = "wikidata"
    prefix = "wd:"

    def __init__(self, store: Store, *, timeout: float = 25.0, offline: bool = False) -> None:
        self.store = store
        self.timeout = timeout
        self.offline = offline
        self._session = requests.Session()
        self._session.headers["User-Agent"] = UA
        self._failures = 0
        self._cooldown_until = 0.0
        self._last = 0.0
        self.live_requests = 0

    @property
    def available(self) -> bool:
        if self.offline:
            return False
        if self._failures >= 5 and time.time() >= self._cooldown_until:
            self._failures = 0  # после паузы пробуем снова, а не до конца прогона
        return self._failures < 5

    def _get(self, url: str, params: dict, *, post: bool = False) -> dict | None:
        if not self.available:
            return None
        for attempt in range(3):
            wait = self._last + MIN_INTERVAL - time.time()
            if wait > 0:
                time.sleep(wait)  # вежливая пауза между запросами к публичному сервису
            try:
                self.live_requests += 1
                self._last = time.time()
                if post:
                    r = self._session.post(url, data=params, timeout=self.timeout)
                else:
                    r = self._session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    self._failures = 0
                    return r.json()
                if r.status_code in (429, 500, 502, 503, 504):
                    retry = r.headers.get("Retry-After", "")
                    delay = float(retry) if retry.isdigit() else 2.0 * (attempt + 1) ** 2
                    log.warning("wikidata %s -> %s, повтор через %.0f с", url, r.status_code, delay)
                    time.sleep(min(delay, 30.0))
                    continue
                log.warning("wikidata %s -> %s", url, r.status_code)
                break
            except (requests.RequestException, ValueError) as e:
                log.warning("wikidata error: %s", e)
                time.sleep(2.0)
        self._failures += 1
        if self._failures >= 5:
            self._cooldown_until = time.time() + COOLDOWN
            log.warning("wikidata: 5 сбоев подряд, источник на паузе %d с", COOLDOWN)
        return None

    # ---------- привязка ----------
    def link(self, term: str, lang: str) -> list[Node]:
        key = f"wd:search:{lang}:{term}"
        cached = self.store.cache_get(key)
        if cached is None:
            data = self._get(
                API,
                {
                    "action": "wbsearchentities",
                    "search": term,
                    "language": lang,
                    "uselang": lang,
                    "type": "item",
                    "limit": 7,
                    "format": "json",
                },
            )
            if data is None:
                return []
            cached = [
                {"id": d["id"], "label": d.get("label", term), "desc": d.get("description", "")}
                for d in data.get("search", [])
            ]
            self.store.cache_put(key, cached)
        out = []
        kept = [d for d in cached if not any(j in d["desc"].lower() for j in JUNK)]
        banned = self.banned([d["id"] for d in kept])
        for rank, d in enumerate(cached):
            if d not in kept:
                continue
            out.append(
                Node(
                    f"wd:{d['id']}",
                    d["label"],
                    self.name,
                    lang,
                    {"desc": d["desc"], "rank": rank, "banned": d["id"] in banned},
                )
            )
        return out

    def banned(self, qids: list[str]) -> set[str]:
        """Какие из элементов — экземпляры запрещённых типов (P31/P279*). Кэш по элементу."""
        out, todo = set(), []
        for q in qids:
            v = self.store.cache_get(f"wd:ban:{q}")
            if v is None:
                todo.append(q)
            elif v:
                out.add(q)
        if todo:
            values = " ".join(f"wd:{q}" for q in todo)
            bans = " ".join(f"wd:{b}" for b in BANNED_TYPES)
            rows = self._sparql(
                f"SELECT DISTINCT ?item WHERE {{ VALUES ?item {{ {values} }} "
                f"VALUES ?ban {{ {bans} }} ?item wdt:P31/wdt:P279* ?ban }}"
            )
            if rows is None:  # сеть упала: без проверки не привязываем ничего сомнительного
                return set(todo) | out
            hit = {r["item"].rsplit("/", 1)[-1] for r in rows}
            for q in todo:
                self.store.cache_put(f"wd:ban:{q}", q in hit)
            out |= hit
        return out

    # ---------- соседи ----------
    def _sparql(self, query: str) -> list[dict] | None:
        data = self._get(SPARQL, {"query": query, "format": "json"}, post=True)
        if data is None:
            return None
        rows = []
        for b in data["results"]["bindings"]:
            rows.append({k: v["value"] for k, v in b.items()})
        return rows

    _LABEL = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }'

    def _ranked(self, pattern: str, extra: str, limit: int, inner: int = 400) -> str:
        """Кандидаты по шаблону, отсортированные по числу статей в Википедиях (известность).

        Сортировка вынесена на два уровня вложенности: иначе сервис упирается в таймаут.
        """
        return (
            f"SELECT ?x ?xLabel ?xDescription {extra} ?sl WHERE {{ {{ SELECT ?x {extra} ?sl WHERE {{ "
            f"{{ SELECT DISTINCT ?x {extra} WHERE {{ {pattern} }} LIMIT {inner} }} "
            f"OPTIONAL {{ ?x wikibase:sitelinks ?sl }} }} ORDER BY DESC(?sl) LIMIT {limit} }} "
            f"{self._LABEL} }}"
        )

    def _plain(self, pattern: str, extra: str, limit: int) -> str:
        return f"SELECT ?x ?xLabel ?xDescription {extra} WHERE {{ {pattern} {self._LABEL} }} LIMIT {limit}"

    def _queries(self, q: str) -> list[tuple[str, str]]:
        """(слот, запрос). Переменная ?p — свойство, ?via — общий родитель."""
        deeper = " ".join(f"wdt:{p}" for p in DEEPER)
        up = " ".join(f"wdt:{p}" for p in UP)
        aout = " ".join(f"wdt:{p}" for p in ASSOC_OUT)
        ain = " ".join(f"wdt:{p}" for p in ASSOC_IN)
        return [
            ("deeper", self._ranked(f"VALUES ?p {{ {deeper} }} ?x ?p wd:{q} .", "?p", 14)),
            ("side", self._plain(f"VALUES ?p {{ {up} }} wd:{q} ?p ?x .", "?p", 6)),
            (
                "side",
                self._ranked(
                    f"wd:{q} wdt:P279|wdt:P171 ?via . ?x wdt:P279|wdt:P171 ?via . FILTER(?x != wd:{q})",
                    "?via",
                    8,
                    inner=300,
                ),
            ),
            ("assoc", self._plain(f"VALUES ?p {{ {aout} }} wd:{q} ?p ?x .", "?p", 24)),
            ("assoc_in", self._plain(f"VALUES ?p {{ {ain} }} ?x ?p wd:{q} .", "?p", 12)),
        ]

    def _fetch(self, q: str) -> list[dict] | None:
        rows: list[dict] = []
        for slot, query in self._queries(q):
            res = self._sparql(query)
            if res is None:
                return None
            for r in res:
                prop = r.get("p", "").rsplit("/", 1)[-1]
                if prop == "P31" and int(r.get("sl", 0)) == 0:
                    continue  # экземпляр без единой статьи в Википедии — как правило, мусор
                if slot == "deeper":
                    rel = DEEPER.get(prop, "подкласс")
                elif slot == "side":
                    rel = UP.get(prop, "обобщение") if prop else "брат (общий родитель)"
                elif slot == "assoc":
                    rel = ASSOC_OUT.get(prop, "связано")
                else:
                    rel = ASSOC_IN.get(prop, "связано")
                r["slot"] = "assoc" if slot == "assoc_in" else slot
                r["rel"] = rel
                rows.append(r)
        return rows

    def neighbors(self, node: Node) -> list[Link]:
        q = _q(node.id)
        key = f"wd:nb:{q}"
        rows = self.store.cache_get(key)
        if rows is None:
            rows = self._fetch(q)
            if rows is None:
                return []
            self.store.cache_put(key, rows)
        out: list[Link] = []
        seen: set[tuple[str, str]] = set()
        for r in rows:
            xid = r["x"].rsplit("/", 1)[-1]
            label = r.get("xLabel", xid)
            if label == xid or label.startswith("Q") and label[1:].isdigit():
                continue  # нет подписи ни на русском, ни на английском
            if (xid, r["slot"]) in seen:
                continue
            seen.add((xid, r["slot"]))
            via = r.get("via", "")
            via = f"wd:{via.rsplit('/', 1)[-1]}" if via.startswith("http") else ""
            w = {"deeper": 0.85, "side": 0.75, "assoc": 0.7}[r["slot"]]
            n = Node(f"wd:{xid}", label, self.name, "", {"desc": r.get("xDescription", "")[:160]})
            out.append(Link(n, r["rel"], r["slot"], w, self.name, via=via))
        return out
