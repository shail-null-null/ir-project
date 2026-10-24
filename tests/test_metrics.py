import pytest
from src.metrics import precision_at_k, average_precision, ndcg_at_k, evaluate

# Toy collection (values hand-computed in Microtask 6)
RANKED_Q1 = ["d3", "d2", "d1", "d4", "d6"]
RELS_Q1 = {"d1": 2, "d3": 1, "d6": 1}
RANKED_Q2 = ["d7", "d5", "d9"]
RELS_Q2 = {"d5": 1, "d8": 1}


def approx(x):
    return pytest.approx(x, abs=1e-3)


# --- Hand-computed table ---
def test_q1_p_at_3():
    assert precision_at_k(RANKED_Q1, RELS_Q1, 3) == approx(0.667)

def test_q1_ap():
    assert average_precision(RANKED_Q1, RELS_Q1) == approx(0.756)

def test_q1_ndcg_at_3():
    assert ndcg_at_k(RANKED_Q1, RELS_Q1, 3) == approx(0.639)

def test_q1_ndcg_at_5():
    assert ndcg_at_k(RANKED_Q1, RELS_Q1, 5) == approx(0.762)

def test_q2_p_at_3():
    assert precision_at_k(RANKED_Q2, RELS_Q2, 3) == approx(0.333)

def test_q2_ap_counts_unretrieved_relevant():
    assert average_precision(RANKED_Q2, RELS_Q2) == approx(0.25)

def test_q2_ndcg_at_3():
    assert ndcg_at_k(RANKED_Q2, RELS_Q2, 3) == approx(0.387)


# --- Adversarial cases: one per bug ---
def test_ndcg_independent_of_qrels_order():          # bug 5
    reordered = {"d3": 1, "d1": 2, "d6": 1}
    assert ndcg_at_k(RANKED_Q1, reordered, 3) == approx(0.639)

def test_k_larger_than_ranked_list():                # bug 3
    assert precision_at_k(RANKED_Q2, RELS_Q2, 5) == approx(0.2)
    assert ndcg_at_k(RANKED_Q2, RELS_Q2, 5) == approx(0.387)

def test_grade_zero_is_not_relevant():               # bugs 2 and 4
    with_zero = {**RELS_Q1, "d2": 0}
    assert precision_at_k(RANKED_Q1, with_zero, 3) == approx(0.667)
    assert average_precision(RANKED_Q1, with_zero) == approx(0.756)

def test_ndcg_never_exceeds_one():                   # bug 6
    five = ["a", "b", "c", "d", "e"]
    assert ndcg_at_k(five, {d: 1 for d in five}, 5) == approx(1.0)

def test_no_relevant_docs_returns_zero():            # bugs 4 and 7
    assert average_precision(["a", "b"], {"a": 0}) == 0.0
    assert ndcg_at_k(["a", "b"], {"a": 0}, 2) == 0.0


# --- evaluate(): missing query and MAP ---
def test_evaluate_missing_query_scores_zero():
    run = {"q1": {"d3": 5.0, "d2": 4.0, "d1": 3.0, "d4": 2.0, "d6": 1.0},
           "q2": {"d7": 3.0, "d5": 2.0, "d9": 1.0}}
    qrels = {"q1": RELS_Q1, "q2": RELS_Q2, "q3": {"d2": 1}}
    results = evaluate(run, qrels, 3)

    assert set(results) == {"q1", "q2", "q3"}
    assert results["q3"] == {"P@3": 0, "AP": 0, "NDCG@3": 0}
    assert results["q1"]["NDCG@3"] == approx(0.639)
    mean_ap = sum(r["AP"] for r in results.values()) / len(results)
    assert mean_ap == approx(0.335)