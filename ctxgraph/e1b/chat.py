"""История чата с лимитом контекста и сжатием в резюме — одинаково для всех вариантов.

Модель видит: системный промпт, резюме (если было сжатие), блок памяти варианта, окно
последних реплик и вопрос. Когда окно не помещается в бюджет, старшие реплики уходят в
резюме (одна модель, один промпт, температура 0) и в архив — архив доступен памяти вариантов.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

SUMMARY_SYSTEM = (
    "You maintain a running summary of a long working conversation between a user and an assistant "
    "about the user's project. Merge the previous summary with the new messages. Keep: the project's "
    "facts, constraints, current decisions, rejected options and decisions that were changed (say what "
    "replaced what), and open questions. Drop small talk and code details. Write in the conversation's "
    "language, as a compact list, at most {max_tokens} tokens."
)


@dataclass
class Turn:
    id: str
    role: str  # user | assistant
    text: str
    origin: str  # load | model | followup | question
    tokens: int


@dataclass
class ChatState:
    summary: str = ""
    window: list[Turn] = field(default_factory=list)
    archive: list[Turn] = field(default_factory=list)  # всё, что ушло в резюме
    compressions: list[dict] = field(default_factory=list)

    def copy(self) -> ChatState:
        return ChatState(
            self.summary, list(self.window), list(self.archive), list(self.compressions)
        )


@dataclass(frozen=True)
class Limits:
    context: int = 6600  # общий лимит модели в эксперименте (окно истории 3750)
    answer: int = 1500  # резерв под ответ (после пилота 2: 900 обрывал половину ответов)
    memory: int = 600  # бюджет блока памяти (одинаковый для вариантов 2–6)
    summary: int = 500  # максимум резюме
    system: int = 250  # системный промпт + обвязка

    @property
    def window(self) -> int:
        """Сколько токенов истории помещается рядом со всем остальным (вопрос — внутри окна)."""
        return self.context - self.answer - self.memory - self.summary - self.system


def window_tokens(state: ChatState) -> int:
    return sum(t.tokens for t in state.window)


def compress_if_needed(
    state: ChatState,
    limits: Limits,
    summarize: Callable[[str, list[Turn]], str],
    count: Callable[[str], int],
    *,
    reserve: int = 0,
) -> None:
    """Сжимать, пока окно (+ reserve под следующую реплику) не влезет. Уходит старшая половина окна."""
    while window_tokens(state) + reserve > limits.window and len(state.window) > 1:
        target = limits.window // 2
        out: list[Turn] = []
        while state.window and window_tokens(state) > target and len(state.window) > 1:
            out.append(state.window.pop(0))
        new_summary = summarize(state.summary, out)
        state.compressions.append(
            {
                "after_turn": out[-1].id,
                "turns": len(out),
                "tokens_out": sum(t.tokens for t in out),
                "summary_tokens": count(new_summary),
            }
        )
        state.summary = new_summary
        state.archive.extend(out)


def summary_messages(prev: str, turns: list[Turn], max_tokens: int) -> list[dict]:
    body = "\n\n".join(f"[{t.role}] {t.text}" for t in turns)
    user = f"PREVIOUS SUMMARY:\n{prev or '(none)'}\n\nNEW MESSAGES:\n{body}\n\nWrite the updated summary."
    return [
        {"role": "system", "content": SUMMARY_SYSTEM.format(max_tokens=max_tokens)},
        {"role": "user", "content": user},
    ]
