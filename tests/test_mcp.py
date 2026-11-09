import ctxgraph.mcp_server as srv
from ctxgraph.menu import HEADER


def test_mcp_tools_roundtrip(make_engine, monkeypatch):
    eng = make_engine()
    monkeypatch.setattr(srv, "_engine", eng)
    text = srv.menu("Какие рыбы подойдут для аквариума?")
    assert text.startswith(HEADER) and "[f:" in text
    out = srv.expand("f:salmonids")
    assert "сёмга" in out or "форель" in out
    assert "неизвестен" in srv.expand("f:nope")
    assert "Отмечено: f:salmon" in srv.mark_used("f:salmon")


def test_mcp_tools_registered():
    import asyncio

    names = {t.name for t in asyncio.run(srv.server.list_tools())}
    assert {"menu", "expand", "mark_used"} <= names
