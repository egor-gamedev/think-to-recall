"""Benchmark 1 runner: data mapping, tuning-set isolation, passage ranking and Recall (seconds)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench1_run as B  # noqa: E402

DATA = ROOT / ".cache" / "bench"
need = pytest.mark.skipif(not (DATA / "musique.json").exists(), reason="benchmark data not downloaded")


def test_rank_passages_max_and_fallback():
    r = B.rank_passages({"f:p3:0": 0.2, "f:p3:1": 0.9, "f:p1:0": 0.5}, fallback=[7, 1, 3, 2])
    assert r == [3, 1, 7, 2]


def test_recall():
    assert B.recall([4, 2, 9, 1, 0, 5], [2, 5], 5) == 0.5
    assert B.recall([5, 2], [2, 5], 2) == 1.0


@need
@pytest.mark.parametrize("ds", ["musique", "2wiki"])
def test_same_1000_questions_and_gold_in_corpus(ds, capsys):
    qs, corpus = B.load(ds, DATA, None)
    assert len(qs) == 1000 and "WARNING" not in capsys.readouterr().out
    assert all(q["gold"] and max(q["gold"]) < len(corpus) for q in qs)
    assert len(corpus) == {"musique": 11656, "2wiki": 6119}[ds]


@need
def test_dev100_disjoint_from_test():
    if not (DATA / B.DEV_FILE).exists():
        pytest.skip("MuSiQue dev not downloaded")
    test = {q["id"] for q in B.load("musique", DATA, None)[0]}
    dev, corpus = B.load("musique_dev100", DATA, None)
    assert len(dev) == 100 and not test & {q["id"] for q in dev}
    assert all(q["gold"] for q in dev) and corpus
    dev2, _ = B.load("musique_dev200", DATA, None)
    ids2 = {q["id"] for q in dev2}
    assert len(ids2) == 200 and not ids2 & test and not ids2 & {q["id"] for q in dev}
    ids3 = {q["id"] for q in B.load("musique_dev200b", DATA, None)[0]}
    assert len(ids3) == 200 and not ids3 & (test | ids2 | {q["id"] for q in dev})
