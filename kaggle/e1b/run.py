"""Kaggle-скрипт E1b: ставит зависимости, копирует код из датасета ctxgraph-e1b и гоняет прогон.

Что гонять — в RUN_ARGS (основной прогон E1b по пре-регистрации v1.0: 3 проекта × 6 вариантов × 18 вопросов). Результаты — /kaggle/working/out.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
import urllib.request

RUN_ARGS = [
    "--backend", "server",
    "--projects", "crypto-wallet,django-shop,rnaseq-thesis",
    "--encoder", "e5",
    "--out", "/kaggle/working/out",
]  # fmt: skip


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


src = os.path.dirname(glob.glob("/kaggle/input/**/ctxgraph/e1b/runner.py", recursive=True)[0])
src = os.path.dirname(os.path.dirname(src))
repo = "/kaggle/working/repo"
shutil.copytree(src, repo, dirs_exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(
    f"{sys.executable} -m pip install -q pymorphy3 yake nltk pyyaml sentence-transformers bitsandbytes vllm"
)
# vLLM ставит свой torch; torchaudio из образа собраны под другую CUDA и ломают импорт
sh(f"{sys.executable} -m pip uninstall -y -q torchaudio torchcodec")
sh(
    f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(vllm|torch|transformers|sentence)' || true"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# сервер vLLM: параллельные запросы шести вариантов идут одним батчем
server = subprocess.Popen(
    "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype half "
    "--tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.88 "
    "--seed 0 --port 8000 > /kaggle/working/vllm.log 2>&1",
    shell=True,
)
for _ in range(180):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5)
        break
    except OSError:
        if server.poll() is not None:
            sh("tail -50 /kaggle/working/vllm.log")
            raise SystemExit("vLLM server died") from None
        time.sleep(10)
print("vLLM server up", flush=True)
sh(f"cd {repo} && {sys.executable} scripts/e1b_run.py " + " ".join(RUN_ARGS))
server.terminate()
shutil.rmtree(repo)  # в выход — только out/ и vllm.log
