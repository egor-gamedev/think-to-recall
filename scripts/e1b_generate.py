"""E1b: генерация загрузочного диалога проекта по скелету (spec.yaml).

Генератор — НЕ тестируемая модель (по умолчанию DeepSeek V4 Flash через timeweb; тестируется Qwen2.5).
Каждый сегмент — один вызов: модель пишет реплики пользователя и ассистента, помечая, где
прозвучал факт. Валидатор проверяет, что каждый факт сегмента есть в помеченной реплике
(доля содержательных лемм факта ≥ 0,5); при провале — один повтор сегмента.
Расход — отдельный учёт reports/e1b/spend.jsonl с лимитом --limit-rub.

Запуск: python scripts/e1b_generate.py crypto-wallet [--segments S1] [--limit-rub 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctxgraph import text as T  # noqa: E402
from ctxgraph.e1.budget import Ledger  # noqa: E402
from ctxgraph.e1.llm import Client  # noqa: E402
from scripts.check_model import load_env  # noqa: E402

GENERATOR = "deepseek/deepseek-flash"
PROJECTS = ROOT / "benchmark_e1b" / "projects"
TARGET_TOKENS = 6500

SYSTEM = """\
You write realistic chat transcripts between a software team lead (the user) and an AI assistant, \
for a research benchmark about long-term project memory. Write in the project's language. The user \
writes briefly and casually, sometimes with typos; the assistant answers helpfully, with code where \
natural. Every listed FACT must be stated clearly at least once, in the user's words (decisions are \
announced by the user) or confirmed by the user after the assistant proposes it. State each fact as a \
present decision; do not contradict facts from earlier segments, and treat facts marked as replaced as \
outdated (mention the change explicitly when a replacing fact is introduced). Fill the rest of the \
segment with ordinary development talk on the background topics. Do NOT recap or summarize the \nproject's decisions (no 'remember: ...' lists, no closing summaries): each fact is stated at most twice in \nthe whole segment, the way it would come up naturally. Return json only."""

USER_TMPL = """\
Project: {title}
User persona: {persona}

Facts already established earlier (current versions):
{earlier}

FACTS TO INTRODUCE IN THIS SEGMENT (id: text):
{facts}

Background topics for the rest of this segment: {background}

Length: about {target} tokens in total, 14-20 exchanges (user + assistant), assistant replies of \
varying length (some with code). Spread the facts over the segment, not all at the start.

Return json in this format:
{{"turns": [{{"role": "user", "text": "...", "facts": ["F01"]}}, {{"role": "assistant", "text": "...", "facts": []}}]}}
"facts" lists the ids of facts stated or confirmed in that turn (empty list if none)."""


def current_facts(spec: dict, upto: int) -> dict[str, str]:
    """Факты из сегментов до `upto` с учётом замен."""
    facts: dict[str, str] = {}
    for seg in spec["segments"][:upto]:
        for fid in seg["facts"]:
            f = spec["facts"][fid]
            if "replaces" in f:
                facts.pop(f["replaces"], None)
            facts[fid] = f["text"]
    return facts


def check_segment(spec: dict, seg: dict, turns: list[dict]) -> list[str]:
    """Факты сегмента, которые не прозвучали (по помеченным репликам и леммам)."""
    missing = []
    for fid in seg["facts"]:
        need = set(T.content_lemmas(spec["facts"][fid]["text"]))
        tagged = [t for t in turns if fid in (t.get("facts") or [])]
        ok = any(len(need & set(T.content_lemmas(t["text"]))) >= 0.5 * len(need) for t in tagged)
        if not ok:
            missing.append(fid)
    return missing


def generate(project: str, segment_ids: list[str] | None, limit_rub: float) -> None:
    load_env(ROOT / ".env")
    pdir = PROJECTS / project
    spec = yaml.safe_load((pdir / "spec.yaml").read_text(encoding="utf-8"))
    out_path = pdir / "load.json"
    done = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"segments": {}}
    )
    ledger = Ledger(ROOT / "reports" / "e1b" / "spend.jsonl", limit_rub=limit_rub)
    client = Client(
        GENERATOR,
        ledger,
        phase="e1b-generate",
        params={"temperature": 0.7, "max_tokens": 12000, "reasoning_effort": "none"},
    )
    for i, seg in enumerate(spec["segments"]):
        if segment_ids and seg["id"] not in segment_ids:
            continue
        if seg["id"] in done["segments"] and not done["segments"][seg["id"]].get("missing"):
            continue
        earlier = current_facts(spec, i)
        prompt = USER_TMPL.format(
            title=spec["title"],
            persona=spec["user_persona"].strip(),
            earlier="\n".join(f"- {k}: {v}" for k, v in earlier.items()) or "(none yet)",
            facts="\n".join(
                f"- {fid}: {spec['facts'][fid]['text']}"
                + (
                    f" (this REPLACES {spec['facts'][fid]['replaces']})"
                    if "replaces" in spec["facts"][fid]
                    else ""
                )
                for fid in seg["facts"]
            ),
            background=seg["background"],
            target=TARGET_TOKENS,
        )
        for attempt in range(2):
            r = client.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                purpose=f"generate {project}/{seg['id']}",
                run_id=f"{project}|{seg['id']}|{attempt}",
                extra={"response_format": {"type": "json_object"}},
            )
            try:
                turns = json.loads(r.content)["turns"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raw = pdir / f"raw_{seg['id']}_{attempt}.txt"
                raw.write_text(r.content or "", encoding="utf-8")
                print(
                    f"{seg['id']}: невалидный JSON ({type(e).__name__}; finish/usage {r.usage}), "
                    f"сырой ответ → {raw.name}, повтор"
                )
                continue
            missing = check_segment(spec, seg, turns)
            tokens = sum(T.approx_tokens(t["text"]) for t in turns)
            done["segments"][seg["id"]] = {
                "turns": turns,
                "missing": missing,
                "approx_tokens": tokens,
                "generator": r.model,
                "usage": r.usage,
            }
            print(
                f"{seg['id']}: реплик {len(turns)}, ~{tokens} ток., не прозвучали: {missing or 'нет'}, "
                f"потрачено всего {ledger.spent:.2f} ₽"
            )
            if not missing:
                break
        out_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--segments", default="")
    ap.add_argument("--limit-rub", type=float, default=15.0)
    a = ap.parse_args()
    generate(a.project, a.segments.split(",") if a.segments else None, a.limit_rub)
    return 0


if __name__ == "__main__":
    sys.exit(main())
