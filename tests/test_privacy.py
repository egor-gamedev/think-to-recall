from ctxgraph.ingest import ingest
from ctxgraph.privacy import is_clean, redact


def test_redacts_keys_and_personal_data():
    text = (
        "key sk-abcdefghijklmnopqrstuvwxyz012345, github ghp_abcdefghijklmnopqrstuvwxyz0123, "
        "mail ivan.petrov@example.com, phone +7 916 123-45-67, "
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U, "
        "hash 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    )
    r = redact(text)
    for secret in ("sk-abc", "ghp_", "ivan.petrov", "916 123", "eyJhbGci", "9f86d0818"):
        assert secret not in r.text
    assert {"api_key", "email", "phone", "jwt", "hex"} <= set(r.counts)


def test_secret_assignment_keeps_variable_name():
    r = redact('password = "hunter2"  и  API_KEY: abc123xyz')
    assert "hunter2" not in r.text and "abc123xyz" not in r.text
    assert "password" in r.text and "API_KEY" in r.text


def test_dates_and_ordinary_numbers_survive():
    text = "Релиз 2026-10-18, таймаут 15 минут, 100 литров, HTTP 404"
    assert is_clean(text)
    assert redact(text).text == text


def test_credentials_in_url_and_bearer():
    r = redact("postgres://admin:s3cret@db.local/app; Authorization: Bearer abcdef1234567890xyz")
    assert "s3cret" not in r.text and "abcdef1234567890xyz" not in r.text


def test_nothing_secret_reaches_fragments():
    frags = ingest("Мой ключ sk-abcdefghijklmnopqrstuvwxyz и почта a.b@mail.ru")
    f = frags[0]
    assert f.redactions == 2
    assert "sk-abc" not in (f.text or "") and "mail.ru" not in (f.text or "")
    canon = {e.canonical for e in f.entities}
    assert not any("redacted" in c or c == "mail" for c in canon)
