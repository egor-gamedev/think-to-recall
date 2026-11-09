"""OpenAI-совместимый клиент timeweb с повторами и записью расхода."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ctxgraph.e1.budget import Ledger

log = logging.getLogger(__name__)
BASE_URL = "https://api.timeweb.ai/v1"


@dataclass
class Reply:
    content: str
    tool_calls: list[dict]
    reasoning: str | None
    usage: dict
    fingerprint: str | None
    model: str
    ms: float
    raw_message: dict = field(default_factory=dict)


class ApiError(RuntimeError):
    pass


class Client:
    def __init__(
        self,
        model: str,
        ledger: Ledger,
        *,
        phase: str,
        base_url: str = BASE_URL,
        key: str | None = None,
        params: dict | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.ledger = ledger
        self.phase = phase
        self.base_url = base_url.rstrip("/")
        self.key = key or os.environ["TIMEWEB_API_KEY"]
        self.params = params or {}
        self.timeout = timeout

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str,
        run_id: str,
        tools: list[dict] | None = None,
        extra: dict | None = None,
    ) -> Reply:
        body = {"model": self.model, "messages": messages, **self.params, **(extra or {})}
        if tools:
            body["tools"] = tools
        last: Exception | None = None
        for attempt in range(4):
            self.ledger.check()  # до запроса: при исчерпанном бюджете не отправляем
            t0 = time.perf_counter()
            try:
                data = self._post(body)
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:300]
                last = ApiError(f"HTTP {e.code}: {text}")
                if e.code in (429, 500, 502, 503, 504):
                    log.warning("%s: HTTP %s, повтор %d", self.model, e.code, attempt + 1)
                    time.sleep(3 * (attempt + 1) ** 2)
                    continue
                raise last from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = ApiError(str(e))
                log.warning("%s: %s, повтор %d", self.model, e, attempt + 1)
                time.sleep(3 * (attempt + 1) ** 2)
                continue
            ms = (time.perf_counter() - t0) * 1000
            usage = data.get("usage") or {}
            self.ledger.add(self.model, usage, phase=self.phase, purpose=purpose, run_id=run_id)
            msg = data["choices"][0]["message"]
            return Reply(
                content=msg.get("content") or "",
                tool_calls=msg.get("tool_calls") or [],
                reasoning=msg.get("reasoning_content"),
                usage=usage,
                fingerprint=data.get("system_fingerprint"),
                model=data.get("model", self.model),
                ms=ms,
                raw_message=msg,
            )
        raise last or ApiError("unknown error")
