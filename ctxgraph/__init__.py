"""ctxgraph — контекстный граф: меню извлечённых подсказок для LLM."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ctxgraph.embed import Encoder, HashingEncoder, Model2VecEncoder
from ctxgraph.graph import Engine
from ctxgraph.menu import Menu, MenuConfig
from ctxgraph.store import Store

__all__ = ["E1_CONFIG", "Engine", "Menu", "MenuConfig", "Store", "build_engine"]

#: Конфигурация меню для E1 (reports/prereg_e1.md): слоты как в задании, без «опыта».
E1_CONFIG = MenuConfig(
    slot_limits=(("deeper", 3), ("side", 2), ("assoc", 2), ("experience", 0)),
    auto_depth=0,
)

DEFAULT_DB = Path(__file__).resolve().parent.parent / ".cache" / "graph.sqlite"


def build_engine(
    db: str | Path = DEFAULT_DB,
    *,
    encoder: str | Encoder = "model2vec",
    offline: bool = False,
    conceptnet: bool = False,  # api.conceptnet.io отвечает 502; без дампа источник выключен
    experience: bool = True,  # слот «опыт»; в E1 выключен (reports/01b_edge_fixes.md §8)
    config: MenuConfig | None = None,
    session: str | None = None,
    seed: int = 0,
) -> Engine:
    from ctxgraph.sources import (
        ConceptNetSource,
        ExperienceSource,
        RuWordNetSource,
        WikidataSource,
        WordNetSource,
    )

    store = Store(db)
    if isinstance(encoder, str):
        encoder = HashingEncoder() if encoder == "hashing" else Model2VecEncoder()
    sources = [
        WordNetSource(),
        RuWordNetSource(),
        WikidataSource(store, offline=offline),
    ]
    if experience:
        sources += [ExperienceSource(store)]
    else:
        # Без опыта нет и сокращений в слоте «опыт»: лимит слота 0.
        base = config or MenuConfig()
        limits = tuple((k, 0 if k == "experience" else v) for k, v in base.slot_limits)
        config = replace(base, slot_limits=limits)
    if conceptnet:
        sources.insert(3, ConceptNetSource(store, offline=offline))
    return Engine(store, sources, encoder, config=config, session=session, seed=seed)
