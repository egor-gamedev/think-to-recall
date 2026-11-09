from ctxgraph import text as T
from ctxgraph.ingest import extract_entities, ingest


def test_canonical_identifiers_merge():
    assert T.canonical("parse_config") == T.canonical("parseConfig") == T.canonical("parse config")


def test_russian_lemmas():
    assert T.lemma("рыбы") == "рыба"
    assert T.lemma("платежей") == "платеж"
    assert "аквариум" in T.content_lemmas("Для аквариумов и с рыбами")


def test_message_split_into_paragraphs_and_code():
    msg = "Первый абзац про оплаты.\n\nВторой абзац про вебхуки.\n\n```python\ndef save_payment(x):\n    pass\n```"
    frags = ingest([{"role": "user", "content": msg}])
    assert [f.kind for f in frags] == ["paragraph", "paragraph", "code"]
    assert frags[2].text is None  # код не хранится
    assert any(e.canonical == "save payment" for e in frags[2].entities)


def test_single_paragraph_message_keeps_kind():
    frags = ingest([{"role": "assistant", "content": "Коротко: используй idempotency_key."}])
    assert len(frags) == 1 and frags[0].kind == "message"
    assert frags[0].source == "msg:0:assistant:0"


def test_file_items_store_only_entities():
    items = [
        {
            "id": "README.md",
            "kind": "doc",
            "title": "README.md",
            "content": "Technicians stay offline for 4 days in basements.",
        }
    ]
    f = ingest(items)[0]
    assert f.kind == "file" and f.text is None and f.label == "README.md"
    assert any(e.kind == "quantity" and e.canonical == "4 days" for e in f.entities)


def test_tool_call_and_result():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "app/models.py"}'}}
            ],
        },
        {"role": "tool", "name": "read_file", "content": "class Tablet(Base): pass"},
    ]
    frags = ingest(msgs)
    assert [f.kind for f in frags] == ["tool_call", "tool_result"]
    assert frags[1].text is None
    assert any(e.kind == "path" and e.canonical == "app/models.py" for e in frags[0].entities)


def test_regex_entities():
    ents = extract_entities(
        "See https://example.org/docs, file src/app/main.py, deadline 18 октября 2026, "
        "latency 250 ms, error ORA-00942 and HTTP 503."
    )
    kinds = {(e.kind, e.canonical) for e in ents}
    assert ("url", "https://example.org/docs") in kinds
    assert ("path", "src/app/main.py") in kinds
    assert ("date", "18 октября 2026") in kinds
    assert ("quantity", "250 ms") in kinds
    assert ("code", "ORA-00942") in kinds and ("code", "HTTP503") in kinds


def test_raw_text_ingest_and_signature():
    frags = ingest("Рыбы в аквариуме.\n\nФильтр для воды.")
    assert len(frags) == 2
    assert "рыба" in frags[0].signature
