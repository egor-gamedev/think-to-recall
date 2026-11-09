"""Фильтр приватности: вырезает секреты и персональные данные до записи в граф.

Правило: в хранилище не попадает ничего, что совпало с шаблоном ниже. Совпадение
заменяется меткой `[REDACTED:<вид>]`, чтобы было видно, что фильтр сработал.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Порядок важен: сначала длинные и специфичные шаблоны, потом общие.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    ),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    (
        "api_key",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}"
            r"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|nvapi-[A-Za-z0-9_-]{20,})"
        ),
    ),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    (
        "secret_assignment",
        re.compile(
            r"(?i)\b(password|passwd|pwd|пароль|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret)"
            r"(\s*[:=]\s*|\s+)(\"[^\"]*\"|'[^']*'|\S+)"
        ),
    ),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@\S+", re.I)),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # 10-значный номер (3-3-2-2) с необязательным кодом страны; даты 4-2-2 сюда не попадают.
    (
        "phone",
        re.compile(
            r"(?<![\w.])(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?![\w.])"
        ),
    ),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("hex", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("base64", re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])")),
]


@dataclass(frozen=True)
class Redaction:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    counts: dict[str, int] = {}
    for kind, pattern in _PATTERNS:

        def _sub(m: re.Match[str], kind: str = kind) -> str:
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "secret_assignment":
                # Имя переменной оставляем: оно само по себе не секрет и полезно как сущность.
                return f"{m.group(1)}{m.group(2)}[REDACTED:{kind}]"
            return f"[REDACTED:{kind}]"

        text = pattern.sub(_sub, text)
    return Redaction(text, counts)


def is_clean(text: str) -> bool:
    return redact(text).total == 0
