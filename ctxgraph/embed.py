"""Маленький кодировщик на CPU для привязки и ранжирования.

Основной: model2vec `minishlab/potion-multilingual-128M` (статические эмбеддинги,
многоязычный, без torch). Запасной `HashingEncoder` — детерминированный, без загрузок,
для тестов и офлайна; видит только поверхность слов.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"


class Encoder:
    name = "base"
    dim = 0

    def _encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def __init__(self) -> None:
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = np.asarray(self._encode(missing), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1, norms)
            for t, v in zip(missing, vecs, strict=True):
                self._cache[t] = v
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])

    def one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def sim(self, a: str, b: str) -> float:
        return float(self.one(a) @ self.one(b))


class Model2VecEncoder(Encoder):
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        super().__init__()
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from model2vec import StaticModel

        self._model = StaticModel.from_pretrained(model)
        self.name = f"model2vec:{model}"
        self.dim = int(self._model.dim)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts)


class HashingEncoder(Encoder):
    """Символьные 3-граммы + слова, хешированные в фиксированный вектор."""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = f" {t.lower()} "
            feats = [t[j : j + 3] for j in range(len(t) - 2)] + t.split()
            for f in feats:
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0 if h & 1 << 31 else -1.0
        return out
