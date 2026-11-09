"""Калибровка оценки токенов меню по настоящему токенизатору модели (usage.prompt_tokens).

Берёт меню из reports/data/stage1_menus.json, шлёт каждое одним сообщением с max_tokens=1
и вычитает накладные расходы пустого сообщения. Это калибровка инструмента, не эксперимент.
Запросов: 1 + число меню (по умолчанию 11).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph.text import approx_tokens  # noqa: E402
from scripts.check_model import DEFAULT_MODEL, load_env  # noqa: E402


def prompt_tokens(base: str, key: str, model: str, content: str) -> int:
    body = {
        "model": model,
        "max_tokens": 1,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(json.loads(r.read())["usage"]["prompt_tokens"])


def main() -> int:
    load_env(ROOT / ".env")
    key = os.environ["TIMEWEB_API_KEY"]
    base = os.environ.get("TIMEWEB_BASE_URL", "https://api.timeweb.ai/v1")
    model = os.environ.get("TIMEWEB_MODEL") or DEFAULT_MODEL
    data = json.loads((ROOT / "reports/data/stage1_menus.json").read_text(encoding="utf-8"))
    menus = [(t["topic"], d, t[f"depth{d}"]["menu"]) for t in data["topics"] for d in (0, 1)]
    overhead = prompt_tokens(base, key, model, ".")
    rows = []
    for topic, depth, menu in menus:
        real = prompt_tokens(base, key, model, menu) - overhead + 1  # «.» — один токен
        rows.append(
            {
                "topic": topic,
                "auto_depth": depth,
                "real": real,
                "est": approx_tokens(menu),
                "chars": len(menu),
            }
        )
    ratio = sum(r["real"] for r in rows) / sum(r["est"] for r in rows)
    out = {"model": model, "overhead": overhead, "rows": rows, "real_over_est": round(ratio, 3)}
    (ROOT / "reports/data/token_calibration.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for r in rows:
        print(r)
    print("real/est =", round(ratio, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
