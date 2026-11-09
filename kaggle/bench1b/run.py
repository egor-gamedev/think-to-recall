"""Kaggle (GPU, private): benchmark 1b (second attempt) per prereg `reports/prereg_bench1b.md` — no LLM calls.

Same setup as kaggle/bench1/run.py (code from the private dataset `ctxgraph-bench1`, data from Hugging Face at
pinned revisions); runs scripts/bench1b_run.py: graph build, variants d0 / K / G / KG, gate config
scripts/bench1b_config.json. MODE "debug" = dev-100 only (check before freezing); "test" = MuSiQue then 2Wiki.
Output: /kaggle/working/out1b/<dataset>/{recall.jsonl,summary.json}.
"""

import glob
import os
import shutil
import subprocess
import sys

MODE = "test"
DATASETS = ["musique_dev100"] if MODE == "debug" else ["musique", "2wiki"]


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


src = os.path.dirname(glob.glob("/kaggle/input/**/scripts/bench1b_run.py", recursive=True)[0])
src = os.path.dirname(src)
repo = "/kaggle/working/repo"
shutil.copytree(src, repo, dirs_exist_ok=True)
os.makedirs(f"{repo}/.cache", exist_ok=True)
shutil.move(f"{repo}/nltk_data", f"{repo}/.cache/nltk")  # WordNet data (hidden folders are not uploaded)
# Kaggle unzips uploaded zips into a nested folder (corpora/wordnet/wordnet/...): flatten it
for d in glob.glob(f"{repo}/.cache/nltk/corpora/*/"):
    name = os.path.basename(os.path.dirname(d))
    inner = os.path.join(d, name)
    if os.path.isdir(inner):
        for f in os.listdir(inner):
            shutil.move(os.path.join(inner, f), os.path.join(d, f))
        os.rmdir(inner)
assert os.path.exists(f"{repo}/.cache/nltk/corpora/wordnet/lexnames"), "WordNet data missing"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip install -q pymorphy3 yake nltk pyyaml 'spacy>=3.7,<4' sentence-transformers huggingface_hub")
sh(f"{sys.executable} -m spacy download en_core_web_sm")
sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE '^(torch|transformers|sentence|spacy)' || true")
os.environ["NLTK_DATA"] = f"{repo}/.cache/nltk"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "0"
for ds in DATASETS:
    sh(f"cd {repo} && {sys.executable} scripts/bench1b_run.py --dataset {ds} --device cuda --reranker fp32 "
       f"--out /kaggle/working/out1b --data /kaggle/working/data")
shutil.rmtree(repo)  # output: only out/
