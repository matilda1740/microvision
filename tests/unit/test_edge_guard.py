from scripts.run_full_pipeline import adjust_top_k


def test_adjust_top_k_no_adjust():
    # small N should not adjust
    N = 10
    requested = 5
    top_k = adjust_top_k(N, requested, max_total_edges=1000, per_source_top_k_cap=100)
    assert top_k == requested


def test_adjust_top_k_adjusts():
    N = 1000
    requested = 500
    top_k = adjust_top_k(N, requested, max_total_edges=100000, per_source_top_k_cap=50)
    # N*requested = 500k > 100k so should adjust to floor(100k / 1000) = 100, but capped to 50
    assert top_k == 50
