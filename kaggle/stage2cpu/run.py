"""Kaggle CPU (no GPU, internet on, private): stage 2 main per `reports/prereg_stage2.md` amendment 1.

1. Planner on NVIDIA (same model), key from Kaggle Secrets (NVIDIA_API_KEY); ImplexConv first (seed-0 question
   order), then the rest of IMLogic; 2 workers, back-off up to 5 min. Answers are appended to
   /kaggle/working/planner_cache/ as they arrive; previous caches (dataset `ctxgraph-stage2-data` and any earlier
   kernel output attached via kernel_sources) are merged first, so a rerun continues where the last one stopped.
2. When the analysis minimum is reached (ImplexConv answered prefix >= 300; IMLogic whole users) and before the
   12-hour limit: search A / SC / SR (+ SG on IMLogic) on CPU and the report -> /kaggle/working/stage2/.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
PLANNER_DEADLINE = T0 + 9.5 * 3600  # no new planner request after this; the rest is search + report
W = "/kaggle/working"
TMP = "/tmp/stage2"  # heavy inputs, not part of the output


def sh(cmd: str, check: bool = True) -> int:
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


hits = glob.glob("/kaggle/input/**/scripts/implicit_run.py", recursive=True)
src = os.path.dirname(os.path.dirname(hits[0]))
repo = f"{TMP}/repo"
shutil.copytree(src, repo, dirs_exist_ok=True)
os.makedirs(f"{repo}/.cache/bench", exist_ok=True)
shutil.move(f"{repo}/nltk_data", f"{repo}/.cache/nltk")
shutil.move(f"{repo}/imlogic_data", f"{repo}/.cache/bench/imlogic")
for q in glob.glob(f"{repo}/.cache/bench/imlogic/qa/qa/*.json"):  # nested unzip of qa/
    shutil.move(q, f"{repo}/.cache/bench/imlogic/qa/")
sh(f"{sys.executable} -m pip install -q pymorphy3 yake nltk pyyaml python-dotenv 'spacy>=3.7,<4' sentence-transformers "
   "huggingface_hub pyarrow rank-bm25 everalgo-core==0.2.0 everalgo-rank==0.3.0 everalgo-clustering==0.2.1")
sh(f"{sys.executable} -m spacy download en_core_web_sm")
sh(f"{sys.executable} -c \"from huggingface_hub import hf_hub_download as h; "
   "[h('Kaylee0501/ImplexConv', f, repo_type='dataset', revision='baf436948f4b6beb7150c75339ac575b1cab55f9', "
   f"local_dir='{repo}/.cache/bench/implexconv') for f in ['data/opposed-00000-of-00002.parquet', "
   "'data/opposed-00001-of-00002.parquet', 'data/supportive-00000-of-00001.parquet']]\"")
# ── restore: planner caches (merged, de-duplicated by request hash), e5 fragments, graphs ──────────────────────────
os.makedirs(f"{W}/planner_cache", exist_ok=True)
for ds in ("implexconv", "imlogic"):
    seen, lines = set(), []
    for f in sorted(glob.glob(f"/kaggle/input/**/cache__{ds}_main.jsonl", recursive=True)
                    + glob.glob(f"/kaggle/input/**/planner_cache/{ds}_main.jsonl", recursive=True)):  # fmt: skip
        for x in open(f, encoding="utf-8"):
            if x.strip() and json.loads(x)["k"] not in seen:
                seen.add(json.loads(x)["k"])
                lines.append(x if x.endswith("\n") else x + "\n")
    open(f"{W}/planner_cache/{ds}_main.jsonl", "w", encoding="utf-8").writelines(lines)
    print(f"restored {len(lines)} cached answers for {ds}", flush=True)
out = f"{TMP}/implicit"
for f in glob.glob("/kaggle/input/**/e5__*", recursive=True):
    _, corpus, name = os.path.basename(f).split("__")
    os.makedirs(f"{out}/implexconv_main/{corpus}", exist_ok=True)
    shutil.copy(f, f"{out}/implexconv_main/{corpus}/{name}")
for f in glob.glob("/kaggle/input/**/graph__*.sqlite", recursive=True):
    user = os.path.basename(f)[len("graph__") : -len(".sqlite")]
    os.makedirs(f"{out}/imlogic_main/{user}", exist_ok=True)
    shutil.copy(f, f"{out}/imlogic_main/{user}/graph.sqlite")

from kaggle_secrets import UserSecretsClient  # noqa: E402

os.environ.update({"NVIDIA_API_KEY": UserSecretsClient().get_secret("NVIDIA_API_KEY"), "NLTK_DATA": f"{repo}/.cache/nltk",
                   "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": "0", "PYTHONUTF8": "1"})  # fmt: skip

# ── 1. planner ──────────────────────────────────────────────────────────────────────────────────────────────────────
os.makedirs(f"{W}/plans", exist_ok=True)
def planner(ds: str, limit: int | None = None, deadline: float = PLANNER_DEADLINE) -> None:
    """deadline=1: cache only (no request), just writes the plans of the questions answered so far."""
    sh(f"cd {repo} && {sys.executable} scripts/planner.py --dataset {ds} --split main --workers 2 --max-pause 300 "
       f"--passes 1000 --deadline {deadline} --data {repo}/.cache/bench --cache {W}/planner_cache/{ds}_main.jsonl "
       f"--out {W}/plans/plans_{ds}_main.jsonl" + (f" --limit {limit}" if limit else ""), check=False)  # fmt: skip


def analysis() -> bool:
    """Search A / SC / SR (+ SG on IMLogic) and the report, on the amendment-1 selection. Runs at most ONCE over the
    whole chain: a kernel that finds a previous report in its inputs only carries it over."""
    n_e5 = len(glob.glob(f"{out}/implexconv_main/*/e5_fids.json"))
    print(f"ImplexConv e5 corpora: {n_e5} / 195", flush=True)
    if n_e5 != 195:
        return False
    planner("imlogic", deadline=1)  # IMLogic plans from the cache (whole users), no new request
    for ds in ("implexconv", "imlogic"):
        if sh(f"cd {repo} && {sys.executable} scripts/implicit_run.py --dataset {ds} --split main --stage sleeves_a "
              f"--device cpu --out {out} --data {repo}/.cache/bench --plans {W}/plans/plans_{ds}_main.jsonl", check=False):  # fmt: skip
            return False  # ImplexConv: answered seed-0 prefix still < 300
        for f in glob.glob(f"{out}/{ds}_main/*.jsonl"):
            os.makedirs(f"{W}/stage2/{ds}_main", exist_ok=True)
            shutil.copy(f, f"{W}/stage2/{ds}_main/")
    for ds in ("implexconv", "imlogic"):  # the plans the analysis used
        shutil.copy(f"{W}/plans/plans_{ds}_main.jsonl", f"{W}/stage2/plans_{ds}_main.jsonl")
    sh(f"cd {repo} && {sys.executable} scripts/stage2_report.py {W}/stage2", check=False)
    return os.path.exists(f"{W}/stage2/stage2_report.md")


def full_analysis() -> bool:
    """Egor 2026-11-09: once ImplexConv has all 855 plans, ONE extra analysis over the full sample — descriptive;
    the main result stays the 300-question analysis (commit 1ec39ea). IMLogic is not completed (not included)."""
    plans = f"{W}/plans/plans_implexconv_main.jsonl"
    if sum(1 for x in open(plans, encoding="utf-8") if x.strip()) < 855:
        return False
    d = f"{W}/stage2_full"
    if sh(f"cd {repo} && {sys.executable} scripts/implicit_run.py --dataset implexconv --split main --stage sleeves_a "
          f"--device cpu --out {TMP}/full --data {repo}/.cache/bench --plans {plans}", check=False):  # fmt: skip
        return False
    os.makedirs(f"{d}/implexconv_main", exist_ok=True)
    for f in glob.glob(f"{TMP}/full/implexconv_main/*.jsonl"):
        shutil.copy(f, f"{d}/implexconv_main/")
    shutil.copy(plans, f"{d}/plans_implexconv_main.jsonl")
    sh(f"cd {repo} && {sys.executable} scripts/stage2_report.py {d}", check=False)
    md = f"{d}/stage2_report.md"
    if os.path.exists(md):
        body = open(md, encoding="utf-8").read()
        note = ("> **Описательно, полная выборка ImplexConv (855 вопросов). Главный результат — анализ 300 вопросов, "
                "коммит 1ec39ea; этот анализ его не меняет.** Вывод ниже (H1/H2) — формальный пересчёт, не новый тест гипотезы.\n\n")
        open(md, "w", encoding="utf-8").write(note + body)
    return os.path.exists(md)


# ── chain (Egor 2026-11-09): main analysis done (1ec39ea) -> only ImplexConv planning, one descriptive full analysis
for name in ("stage2", "stage2_full"):  # carry earlier results over; never re-analyse
    prev = glob.glob(f"/kaggle/input/**/{name}/stage2_report.md", recursive=True)
    if prev:
        shutil.copytree(os.path.dirname(prev[0]), f"{W}/{name}", dirs_exist_ok=True)
full_done = os.path.exists(f"{W}/stage2_full/stage2_report.md")
if not full_done:
    planner("implexconv")
    full_done = full_analysis()
open(f"{W}/status.json", "w").write(json.dumps({"full_analysis_done": full_done, "hours": (time.time() - T0) / 3600}))
shutil.rmtree(TMP, ignore_errors=True)
