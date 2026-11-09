"""Приём контекста: фрагменты по естественным границам + сущности. Без генеративной модели.

Вход — список сообщений чата (`{"role", "content", ...}`), элементы контекста в формате
widethink (`{"id", "kind", "title", "content"}`) или сырой текст. Перед любой обработкой
текст проходит фильтр приватности. Текст файлов, кода и результатов инструментов не
сохраняется — только сущности и ярлык фрагмента.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from ctxgraph import text as T
from ctxgraph.privacy import redact

#: Виды фрагментов, текст которых можно хранить. Остальные — только сущности.
TEXT_KINDS = {"message", "paragraph", "statement", "tool_call"}

_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_URL = re.compile(r"\bhttps?://[^\s<>\"')\]]+")
_PATH = re.compile(
    r"(?<![\w/])(?:[A-Za-z]:\\[\w\\. -]+|(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+|[\w-]+\.(?:py|js|ts|tsx|go|rs|"
    r"java|kt|rb|php|sql|md|yaml|yml|json|toml|cfg|ini|txt|csv|html|css|sh|ps1|c|h|cpp|hpp))(?![\w/])"
)
_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{1,2}\s+(?:января|февраля|марта|апреля|мая|"
    r"июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+\d{4})?|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|"
    r"Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.I,
)
_UNITS = (
    r"ms|мс|s|sec|seconds?|с|сек|секунд[аы]?|min|minutes?|мин(?:ут[аы]?)?|h|hours?|ч|час(?:а|ов)?|"
    r"days?|дн(?:я|ей)?|день|недел[ьиюя]|weeks?|months?|месяц(?:а|ев)?|years?|лет|год(?:а)?|"
    r"KB|MB|GB|TB|КБ|МБ|ГБ|ТБ|%|₽|руб(?:\.|лей|ля)?|\$|USD|EUR|€|rps|qps|req/s|rpm|"
    r"km|км|m|м|kg|кг|mg|мг|ml|мл|°C|шт"
)
_QUANTITY = re.compile(rf"(?<![\w.])(\d+(?:[.,]\d+)?)\s?({_UNITS})(?![\wА-Яа-я])")
_CODE = re.compile(
    r"\b(?:HTTP\s?[1-5]\d\d|[1-5]\d\d\s(?:Not Found|Bad Request|Unauthorized|Forbidden|Internal Server "
    r"Error|Too Many Requests)|E\d{3,5}|ORA-\d{4,5}|[A-Z]{2,6}-\d{2,6}|SQLSTATE\s?\w{5}|errno\s?\d+)\b"
)
_CODE_DEF = re.compile(r"\b(?:def|class|function|func|fn|interface|struct|type)\s+([A-Za-z_]\w*)")
_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|require\(['\"]([\w@/.-]+))", re.M
)
_COLUMN_LIKE = re.compile(r"\b([a-z]+(?:_[a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]+)\b")


@dataclass(frozen=True)
class Entity:
    kind: str  # url | path | date | quantity | code | identifier | keyphrase | lemma
    canonical: str
    surface: str


@dataclass
class Fragment:
    id: str
    kind: str
    source: str  # происхождение: "msg:3:user", "item:README.md", "text:0" ...
    label: str  # короткий ярлык без генерации: заголовок или первые слова
    text: str | None  # None, если текст хранить нельзя
    entities: list[Entity] = field(default_factory=list)
    redactions: int = 0

    @property
    def signature(self) -> list[str]:
        """Ключевые леммы фрагмента — для сигнатуры контекста."""
        return sorted({e.canonical for e in self.entities if e.kind in ("keyphrase", "lemma")})


def _fid(source: str, body: str) -> str:
    return "frag:" + hashlib.sha1(f"{source}\n{body}".encode()).hexdigest()[:12]


def _label(body: str, title: str | None = None, n: int = 8) -> str:
    if title:
        return title[:80]
    out = ""
    for w in body.split()[: n * 2]:
        if len(out) + len(w) + 1 > 60:
            break
        out = f"{out} {w}".strip()
    return out.strip(" ,.;:") or "(пусто)"


@lru_cache(maxsize=4)
def _yake(lang: str):
    import yake

    return yake.KeywordExtractor(lan=lang, n=3, top=8, dedupLim=0.7)


def keyphrases(body: str, top: int = 8) -> list[str]:
    if len(T.words(body)) < 3:
        return []
    # Перевод строки — граница фразы: иначе склеиваются заголовок и первая строка.
    body = re.sub(r"\s*\n\s*", ". ", body)
    # Ничьи по оценке разбиваем по самой фразе: порядок YAKE зависит от PYTHONHASHSEED.
    kws = sorted(_yake(T.detect_lang(body)).extract_keywords(body), key=lambda kv: (kv[1], kv[0]))
    out = []
    for kw, _score in kws[:top]:
        kw = kw.strip()
        toks = T.canonical(kw).split()
        if len(kw) < 3 or not toks or toks[0] in T.STOP_ALL or toks[-1] in T.STOP_ALL:
            continue
        if not T.is_noun_phrase(kw):
            continue
        out.append(kw)
    return out


def extract_entities(body: str, *, code: bool = False) -> list[Entity]:
    ents: list[Entity] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, canon: str, surface: str) -> None:
        canon = canon.strip()
        if not canon or (kind, canon) in seen or "REDACTED" in canon:
            return
        seen.add((kind, canon))
        ents.append(Entity(kind, canon, surface.strip()))

    for m in _URL.finditer(body):
        add("url", m.group(0).rstrip(".,;:").lower(), m.group(0))
    for m in _PATH.finditer(body):
        if not _URL.search(m.group(0)):
            add("path", m.group(0).replace("\\", "/").lower(), m.group(0))
    for m in _DATE.finditer(body):
        add("date", m.group(0).lower(), m.group(0))
    for m in _QUANTITY.finditer(body):
        num, unit = m.group(1).replace(",", "."), m.group(2)
        add(
            "quantity",
            f"{num} {T.lemma(unit) if T.is_cyrillic(unit) else unit.lower()}",
            m.group(0),
        )
    for m in _CODE.finditer(body):
        add("code", m.group(0).upper().replace(" ", ""), m.group(0))
    if code:
        for m in _CODE_DEF.finditer(body):
            add("identifier", T.canonical(m.group(1)), m.group(1))
        for m in _IMPORT.finditer(body):
            mod = next(g for g in m.groups() if g)
            add("identifier", mod.lower(), mod)
    for m in _COLUMN_LIKE.finditer(body):
        canon = T.canonical(m.group(1))
        if any(t not in T.STOP_ALL for t in canon.split()):
            add("identifier", canon, m.group(1))
    if (
        not code
    ):  # в коде ключевые фразы — обрывки синтаксиса («r.product»); там хватает идентификаторов
        for kp in keyphrases(body):
            add("keyphrase", T.canonical(kp), kp)
    for lem in T.content_lemmas(body):
        add("lemma", lem, lem)
    return ents


CODE_EXT = (
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".php",
    ".sql",
    ".sh",
    ".ps1",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
)
_REDACTED = re.compile(r"\[REDACTED:\w+\]")


def _make(kind: str, source: str, body: str, title: str | None = None) -> Fragment:
    red = redact(body)
    clean = red.text
    # Метки вырезанного не должны становиться сущностями.
    plain = _REDACTED.sub(" ", clean)
    is_code = (
        kind == "code"
        or (kind == "file" and bool(title) and title.lower().endswith(CODE_EXT))
        or bool(_CODE_DEF.search(plain))
    )
    return Fragment(
        id=_fid(source, clean),
        kind=kind,
        source=source,
        label=_label(clean, title),
        text=clean if kind in TEXT_KINDS else None,
        entities=extract_entities(plain, code=is_code),
        redactions=red.total,
    )


def split_text(body: str, source: str, base_kind: str = "message") -> list[Fragment]:
    """Сообщение → блоки кода и абзацы. Одно-абзацное сообщение остаётся «message»."""
    frags: list[Fragment] = []
    pos = 0
    pieces: list[tuple[str, str]] = []
    for m in _CODE_BLOCK.finditer(body):
        pieces.append(("text", body[pos : m.start()]))
        pieces.append(("code", m.group(1)))
        pos = m.end()
    pieces.append(("text", body[pos:]))
    paras: list[tuple[str, str]] = []
    for kind, chunk in pieces:
        if kind == "code":
            if chunk.strip():
                paras.append(("code", chunk))
            continue
        for p in re.split(r"\n\s*\n", chunk):
            if p.strip():
                paras.append(("paragraph", p.strip()))
    single = len(paras) == 1
    for i, (kind, chunk) in enumerate(paras):
        k = base_kind if single and kind == "paragraph" else kind
        frags.append(_make(k, f"{source}:{i}", chunk))
    return frags


def ingest(items: Iterable[dict] | str) -> list[Fragment]:
    """Главная точка входа: превратить контекст в фрагменты."""
    if isinstance(items, str):
        return split_text(items, "text")
    frags: list[Fragment] = []
    for i, it in enumerate(items):
        if "role" in it:
            role = it["role"]
            src = f"msg:{i}:{role}"
            if role == "tool":
                frags.append(_make("tool_result", src, str(it.get("content", "")), it.get("name")))
                continue
            for call in it.get("tool_calls") or []:
                fn = call.get("function", call)
                body = f"{fn.get('name', '')}({fn.get('arguments', '')})"
                frags.append(_make("tool_call", src, body, fn.get("name")))
            if it.get("content"):
                frags.extend(split_text(str(it["content"]), src))
        else:
            kind = it.get("kind", "doc")
            kind = {"doc": "file", "file": "file", "statement": "statement"}.get(kind, kind)
            src = f"item:{it.get('id', i)}"
            frags.append(
                _make(kind, src, str(it.get("content", "")), it.get("title") or it.get("id"))
            )
    return frags
