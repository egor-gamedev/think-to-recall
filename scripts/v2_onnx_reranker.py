"""Export BAAI/bge-reranker-v2-m3 (pinned revision) to ONNX and quantize to int8 — no remote code.

    python scripts/v2_onnx_reranker.py   # -> .cache/onnx/bge-reranker-v2-m3/{model.onnx, model.int8.onnx}

Dynamic int8 quantization of linear layers (onnxruntime.quantization.quantize_dynamic). Inference
class `OnnxReranker` mirrors `selection.Reranker.score` (sigmoid of the logit, as sentence-transformers
does for this model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL, REV = "BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
OUT = ROOT / ".cache" / "onnx" / "bge-reranker-v2-m3"


def export() -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV).eval()
    enc = tok([["query", "passage text"]], padding=True, truncation=True, max_length=512, return_tensors="pt")
    fp32 = OUT / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(fp32),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "b", 1: "t"}, "attention_mask": {0: "b", 1: "t"}, "logits": {0: "b"}},
                          opset_version=17, dynamo=False)  # fmt: skip
    tok.save_pretrained(OUT)
    quantize_dynamic(str(fp32), str(OUT / "model.int8.onnx"), weight_type=QuantType.QInt8)
    print("ok", fp32.stat().st_size // 2**20, "MB ->", (OUT / "model.int8.onnx").stat().st_size // 2**20, "MB")


class OnnxReranker:
    def __init__(self, path: Path = OUT / "model.int8.onnx", threads: int = 0) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(OUT)
        self.name = f"{MODEL}@{REV[:10]} onnx:{path.name}"
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, query: str, texts: list[str]) -> list[float]:
        new = [t for t in dict.fromkeys(texts) if (query, t) not in self.cache]
        for i in range(0, len(new), 16):
            batch = new[i : i + 16]
            enc = self.tok([[query, t] for t in batch], padding=True, truncation=True, max_length=512, return_tensors="np")
            logits = self.sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                                "attention_mask": enc["attention_mask"].astype(np.int64)})[0]  # fmt: skip
            for t, z in zip(batch, logits.reshape(-1), strict=True):
                self.cache[(query, t)] = float(1 / (1 + np.exp(-z)))
        return [self.cache[(query, t)] for t in texts]


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    export()
