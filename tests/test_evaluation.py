import pytest

from src.evaluation import mean_metrics, per_query_metrics, ranking, read_run, write_run


def test_write_read_round_trip(tmp_path):
    run = {"q1": {"d3": 5.0, "d2": 4.25, "d1": 3.0}, "q2": {"d7": 0.5}}
    path = tmp_path / "run.trec"
    write_run(run, str(path), "test")
    assert read_run(str(path)) == run


def test_ranking_breaks_ties_by_doc_id_descending():
    assert ranking({"a": 1.0, "c": 2.0, "b": 2.0}) == ["c", "b", "a"]


def test_missing_query_scores_zero_and_counts_in_mean():
    run = {"q1": {"d3": 5.0, "d2": 4.0, "d1": 3.0, "d4": 2.0, "d6": 1.0}}
    qrels = {"q1": {"d1": 2, "d3": 1, "d6": 1}, "q3": {"d2": 1}}
    per_q = per_query_metrics(run, qrels)
    assert set(per_q) == {"q1", "q3"}
    assert all(v == 0 for v in per_q["q3"].values())
    assert mean_metrics(per_q)["MAP"] == pytest.approx(0.756 / 2, abs=1e-3)
