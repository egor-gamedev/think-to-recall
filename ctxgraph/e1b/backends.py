"""Модель и эмбеддер для прогона E1b на Kaggle (T4×2).

`QwenVLLM` — Qwen2.5-7B-Instruct-AWQ (4 бита) в vLLM, температура 0. Вызов функций — через
шаблон чата Qwen (`tools=`) и разбор `<tool_call>{...}</tool_call>` из ответа, без серверного парсера.
`E5Encoder` — multilingual-e5-base, общий для V2 и V6 (префиксы «query:» / «passage:»).
"""

from __future__ import annotations

import json
import re

import numpy as np

from ctxgraph.embed import Encoder

MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
_TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class QwenVLLM:
    def __init__(self, model: str = MODEL, max_model_len: int = 8192, tp: int = 2) -> None:
        from vllm import LLM

        self.model = model
        self.llm = LLM(
            model=model,
            quantization="awq",
            dtype="half",
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.88,
            enforce_eager=True,
            seed=0,
        )
        self.tok = self.llm.get_tokenizer()

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("tool_calls"):  # шаблон Qwen ждёт arguments словарём
                m["tool_calls"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.loads(tc["function"].get("arguments") or "{}"),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            msgs.append(m)
        return self.tok.apply_chat_template(
            msgs, tools=tools, tokenize=False, add_generation_prompt=True
        )

    def chat(self, messages: list[dict], max_tokens: int, tools: list[dict] | None = None) -> dict:
        from vllm import SamplingParams

        prompt = self.prompt(messages, tools)
        out = self.llm.generate([prompt], SamplingParams(temperature=0, max_tokens=max_tokens))[0]
        o = out.outputs[0]
        return parse(o.text, tools, len(out.prompt_token_ids), len(o.token_ids), o.finish_reason)


def parse(text: str, tools, prompt_tokens: int, completion_tokens: int, finish: str) -> dict:
    """Ответ Qwen → content + tool_calls (формат OpenAI) + usage."""
    calls = []
    if tools:
        for k, raw in enumerate(_TOOL.findall(text)):
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            calls.append(
                {
                    "id": f"call_{k}",
                    "type": "function",
                    "function": {
                        "name": d.get("name", ""),
                        "arguments": json.dumps(d.get("arguments", {}), ensure_ascii=False),
                    },
                }
            )
        text = _TOOL.sub("", text).strip()
    return {
        "content": text,
        "tool_calls": calls,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "finish_reason": finish,
    }


class QwenServer(QwenVLLM):
    """Тот же шаблон и разбор, но через `vllm serve` (/v1/completions): параллельные запросы
    вариантов сервер собирает в батч сам."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", model: str = MODEL) -> None:
        import requests
        from transformers import AutoTokenizer

        self.model, self.base_url, self.http = model, base_url, requests.Session()
        self.tok = AutoTokenizer.from_pretrained(model)

    def chat(self, messages: list[dict], max_tokens: int, tools: list[dict] | None = None) -> dict:
        r = self.http.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": self.prompt(messages, tools),
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 0,
                "skip_special_tokens": True,
            },
            timeout=900,
        )
        r.raise_for_status()
        d = r.json()
        c, u = d["choices"][0], d["usage"]
        return parse(
            c["text"], tools, u["prompt_tokens"], u["completion_tokens"], c["finish_reason"]
        )


class QwenHF(QwenVLLM):
    """Запасной путь, если vLLM не встанет: transformers + bitsandbytes NF4, жадное декодирование."""

    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model = f"{model} (bnb-nf4, transformers)"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.hf = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            ),
        )
        self.torch = torch

    def chat(self, messages: list[dict], max_tokens: int, tools: list[dict] | None = None) -> dict:
        ids = self.tok(self.prompt(messages, tools), return_tensors="pt").to(self.hf.device)
        with self.torch.no_grad():
            gen = self.hf.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        new = gen[0, ids["input_ids"].shape[1] :]
        finish = "length" if len(new) >= max_tokens else "stop"
        text = self.tok.decode(new, skip_special_tokens=True)
        return parse(text, tools, ids["input_ids"].shape[1], len(new), finish)


class E5Encoder(Encoder):
    """multilingual-e5: реплики архива — «passage: », вопрос — «query: »."""

    def __init__(self, model: str = "intfloat/multilingual-e5-base", device: str = "cpu", revision: str | None = None) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model, device=device, revision=revision)
        self.name = f"e5:{model}"
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._m.encode(texts, batch_size=32, normalize_embeddings=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        return super().encode([f"passage: {t}" for t in texts])

    def one(self, text: str) -> np.ndarray:
        return super().encode([f"query: {text}"])[0]


class NvidiaChat:
    """Тестируемая модель через API NVIDIA (OpenAI-совместимый), для E1c.

    - Кэш на диске: каждый вызов пишется в jsonl по ключу sha256(модель, сообщения, лимит).
      Повторный запуск берёт готовые ответы из кэша и продолжает с места остановки, без запросов.
    - Жёсткий предел на вызов (`wall`, поток с join): зависшее соединение не держит прогон.
    - Повторы при таймауте, 429 и 5xx — с паузой; после `tries` неудач — исключение (прогон
      останавливается, кэш остаётся).
    - Токены для лимита контекста считаются токенизатором Qwen2.5 как приближением
      (токенизатор Nemotron не нужен для решений о сжатии; одинаков для всех вариантов).
    """

    def __init__(self, model: str, cache: str, *, thinking: bool = False, wall: float = 240,
                 tries: int = 10, base_url: str = "https://integrate.api.nvidia.com/v1",
                 key_env: str = "NVIDIA_API_KEY", max_pause: float | None = None) -> None:  # fmt: skip
        import os
        import threading
        from pathlib import Path

        from tokenizers import Tokenizer

        self.model, self.thinking, self.wall, self.tries = model, thinking, wall, tries
        self.max_pause = max_pause  # None: old caps (60 s, 180 s on 429); else exponential up to this many seconds
        self.failed_attempts = 0
        self.url, self.key = base_url.rstrip("/") + "/chat/completions", os.environ[key_env]
        self.cache_path = Path(cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for ln in self.cache_path.read_text("utf-8").splitlines():
                if ln.strip():
                    d = json.loads(ln)
                    self.cache[d["k"]] = d["r"]
        self.lock = threading.Lock()
        self.api_calls = 0
        from huggingface_hub import hf_hub_download

        self.tok = Tokenizer.from_file(
            hf_hub_download("Qwen/Qwen2.5-7B-Instruct", "tokenizer.json")
        )

    def count(self, text: str) -> int:
        return len(self.tok.encode(text).ids)

    def _post(self, body: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.wall) as r:
            return json.loads(r.read())

    def _go(self, body: dict, box: dict) -> None:
        try:
            box["d"] = self._post(body)
        except Exception as e:  # noqa: BLE001 — разбирается в chat()
            box["e"] = e

    def chat(self, messages: list[dict], max_tokens: int, tools=None) -> dict:
        import hashlib
        import threading
        import time
        import urllib.error

        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}  # fmt: skip
        k = hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if k in self.cache:
                return self.cache[k]
        last = None
        for attempt in range(self.tries):
            box: dict = {}
            t = threading.Thread(target=self._go, args=(body, box), daemon=True)
            t.start()
            t.join(self.wall + 5)
            if "d" in box:
                d = box["d"]
                c = d["choices"][0]
                res = {"content": c["message"].get("content") or "", "tool_calls": [],
                       "usage": d.get("usage") or {}, "finish_reason": c.get("finish_reason")}  # fmt: skip
                with self.lock:
                    self.api_calls += 1
                    self.cache[k] = res
                    with self.cache_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"k": k, "r": res}, ensure_ascii=False) + "\n")
                return res
            e = box.get("e", TimeoutError(f"нет ответа за {self.wall} с"))
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}") from e
            last = e
            # 429 — лимит бесплатного API: ждать дольше
            rate = isinstance(e, urllib.error.HTTPError) and e.code == 429
            with self.lock:
                self.failed_attempts += 1
            cap = self.max_pause if self.max_pause is not None else (180 if rate else 60)
            time.sleep(min(cap, (15 if rate else 5) * 2**attempt))
        raise RuntimeError(f"{self.model}: {self.tries} неудачных попыток: {last}")
