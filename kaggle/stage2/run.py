"""Kaggle (GPU, private): stage 2 dev debug — planner sleeves (plans precomputed locally, no LLM in the kernel).

Code + IMLogic data from the private dataset `ctxgraph-bench1`; ImplexConv and models from Hugging Face at pinned
revisions. Stages per dataset: graph (A/B/C), embed (Qwen3-Embedding-4B), stack (Qwen3-Reranker-4B: D0/D), summary.
Output: /kaggle/working/implicit/<dataset>_<split>/{A,B,C,D0,D}.jsonl, timing.jsonl, summary.json.
"""

import glob
import os
import shutil
import subprocess
import sys

SPLIT = "dev"
DATASETS = ["imlogic", "implexconv"]


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


assert sys.version_info >= (3, 12), f"everalgo needs Python >= 3.12, kernel has {sys.version}"
hits = glob.glob("/kaggle/input/**/scripts/implicit_run.py", recursive=True)
if not hits:
    sh("find /kaggle/input -maxdepth 4 | head -50")
    raise SystemExit("implicit_run.py not in the attached dataset version")
src = os.path.dirname(os.path.dirname(hits[0]))
repo = "/kaggle/working/repo"
shutil.copytree(src, repo, dirs_exist_ok=True)
os.makedirs(f"{repo}/.cache/bench", exist_ok=True)
shutil.move(f"{repo}/nltk_data", f"{repo}/.cache/nltk")
for d in glob.glob(f"{repo}/.cache/nltk/*/*/"):  # Kaggle unzips into nested folders: flatten
    inner = os.path.join(d, os.path.basename(os.path.dirname(d)))
    if os.path.isdir(inner):
        for f in os.listdir(inner):
            shutil.move(os.path.join(inner, f), os.path.join(d, f))
        os.rmdir(inner)
shutil.move(f"{repo}/imlogic_data", f"{repo}/.cache/bench/imlogic")
for q in glob.glob(f"{repo}/.cache/bench/imlogic/qa/qa/*.json"):  # nested unzip of qa/
    shutil.move(q, f"{repo}/.cache/bench/imlogic/qa/")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip install -q pymorphy3 yake nltk pyyaml 'spacy>=3.7,<4' sentence-transformers huggingface_hub "
   "pyarrow rank-bm25 everalgo-core==0.2.0 everalgo-rank==0.3.0 everalgo-clustering==0.2.1")
sh(f"{sys.executable} -m spacy download en_core_web_sm")
sh(f"{sys.executable} -c \"from huggingface_hub import hf_hub_download as h; "
   "[h('Kaylee0501/ImplexConv', f, repo_type='dataset', revision='baf436948f4b6beb7150c75339ac575b1cab55f9', "
   f"local_dir='{repo}/.cache/bench/implexconv') for f in ['data/opposed-00000-of-00002.parquet', "
   "'data/opposed-00001-of-00002.parquet', 'data/supportive-00000-of-00001.parquet']]\"")
os.environ.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NLTK_DATA": f"{repo}/.cache/nltk", "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "0"})
try:
    for ds in DATASETS:
        for stage in ("graph", "embed", "sleeves", "summary"):
            sh(f"cd {repo} && {sys.executable} scripts/implicit_run.py --dataset {ds} --split {SPLIT} --stage {stage} "
               f"--device cuda --out /kaggle/working/implicit --plans {repo}/reports/implicit/plans_{ds}_{SPLIT}.jsonl")
finally:  # also on failure: per-corpus graphs / caches would make the output several GB
    for f in glob.glob("/kaggle/working/implicit/*/*/"):  # keep outputs small: drop per-corpus graphs and caches
        shutil.rmtree(f)
    shutil.rmtree(repo)
