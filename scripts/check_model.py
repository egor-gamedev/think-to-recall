"""Проверка модели на timeweb: список моделей, точное имя, вызов инструментов.

Запуск: python scripts/check_model.py  (ключ берётся из .env, в вывод не попадает)
Делает не больше 2 запросов к чату.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "deepseek/deepseek-flash"  # DeepSeek V4.1 Flash


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def call(base: str, key: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "body": e.read().decode(errors="replace")[:500]}


def main() -> int:
    load_env(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("TIMEWEB_API_KEY")
    base = os.environ.get("TIMEWEB_BASE_URL", "https://api.timeweb.ai/v1")
    if not key:
        print("TIMEWEB_API_KEY не задан в .env")
        return 1

    models = call(base, key, "/models")
    ids = [m.get("id") for m in models.get("data", [])] if "data" in models else []
    print("models:", ids or models)
    # deepseek-flash — официальное имя V4.1 Flash; deepseek-v4-flash на timeweb отдаёт
    # тот же system_fingerprint (проверено 2026-10-18).
    model = os.environ.get("TIMEWEB_MODEL") or DEFAULT_MODEL
    if model not in ids:
        print("модель", model, "не найдена на timeweb; задай TIMEWEB_MODEL в .env")
        return 1
    for m in models.get("data", []):
        if m.get("id") == model:
            print("model meta:", json.dumps(m, ensure_ascii=False))
    print("using:", model)

    plain = call(
        base,
        key,
        "/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Назови свою модель и версию одной строкой."}],
        },
    )
    print(
        "plain:",
        json.dumps({k: plain.get(k) for k in ("model", "usage")}, ensure_ascii=False),
        "|",
        (plain.get("choices") or [{}])[0].get("message", {}).get("content") or plain,
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "expand",
                "description": "Раскрыть узел графа",
                "parameters": {
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": ["node_id"],
                },
            },
        }
    ]
    tc = call(
        base,
        key,
        "/chat/completions",
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 200,
            "tools": tools,
            "messages": [{"role": "user", "content": "Раскрой узел n_fish инструментом expand."}],
        },
    )
    msg = (tc.get("choices") or [{}])[0].get("message", {})
    print("tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False) if msg else tc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
