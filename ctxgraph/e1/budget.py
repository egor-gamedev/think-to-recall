"""Учёт расхода в рублях и жёсткий ограничитель (пре-регистрация E1, §10).

Каждый ответ API пишется строкой в `spend.jsonl`; накопленная сумма читается из файла,
поэтому ограничитель действует на весь E1 (пилот + основной) и переживает перезапуски.
Перед каждым запросом вызывается `check()`: при сумме ≥ лимита запрос не отправляется.
Перерасход ограничен стоимостью одного запроса (проверка идёт до запроса, цена — после).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

#: ₽ за 1 млн токенов (вход, выход), timeweb, 2026-10-19 (страница от Егора).
PRICES_RUB = {
    "deepseek/deepseek-flash": (59.0, 178.0),
    "openai/gpt-5.4-nano": (27.0, 169.0),
    "openai/gpt-5.4-mini": (101.0, 608.0),
}
# v1.5: лимит поднят Егором с 310 до 330 ₽ по учёту (≈ 287 ₽ реального списания timeweb).
LIMIT_RUB = 330.0


class BudgetExceeded(RuntimeError):
    pass


def cost_rub(model: str, usage: dict) -> float:
    pin, pout = PRICES_RUB[model]
    # Кэш-попадания считаем по полной цене входа: скидку timeweb не проверяли (оценка сверху).
    return (
        usage.get("prompt_tokens", 0) / 1e6 * pin + usage.get("completion_tokens", 0) / 1e6 * pout
    )


class Ledger:
    def __init__(self, path: Path, limit_rub: float = LIMIT_RUB) -> None:
        self.path = Path(path)
        self.limit = limit_rub
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._spent = 0.0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._spent += json.loads(line)["rub"]

    @property
    def spent(self) -> float:
        return self._spent

    def check(self) -> None:
        if self._spent >= self.limit:
            raise BudgetExceeded(f"потрачено {self._spent:.2f} ₽ ≥ лимита {self.limit:.0f} ₽")

    def add(self, model: str, usage: dict, *, phase: str, purpose: str, run_id: str) -> float:
        rub = cost_rub(model, usage)
        self._spent += rub
        rec = {
            "ts": time.time(),
            "phase": phase,
            "purpose": purpose,
            "run_id": run_id,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "rub": rub,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rub
