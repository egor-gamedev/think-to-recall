"""Интерфейс источника подсказок. Подсказки извлекаются, а не генерируются."""

from __future__ import annotations

from dataclasses import dataclass

from ctxgraph.store import Node

#: Слоты меню. `far` собирается в menu.py из кандидатов остальных слотов.
SLOTS = ("deeper", "side", "assoc", "experience")


@dataclass(frozen=True)
class Link:
    node: Node
    rel: str  # как связано, словами источника: "подкласс", "часть", "использует" ...
    slot: str  # deeper | side | assoc | experience
    weight: float  # 0..1, доверие источника к связи
    # тип источника для отчёта: wordnet | ruwordnet | wikidata | conceptnet | experience
    source: str
    via: str = ""  # для 2-шаговых связей: через какой узел


class Source:
    name = "base"
    prefix = ""

    def owns(self, node_id: str) -> bool:
        return node_id.startswith(self.prefix)

    def link(self, term: str, lang: str) -> list[Node]:
        """Кандидаты-узлы для термина (точное совпадение / лемма). Порядок — по уверенности."""
        return []

    def neighbors(self, node: Node) -> list[Link]:
        return []

    @property
    def available(self) -> bool:
        return True
