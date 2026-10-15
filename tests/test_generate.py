"""
tests/test_generate.py
Unit tests for overlap/generate.py:
Tests planted overlap generator, crisp control (p_ov=0), and seed reproducibility.
"""

import pytest
from overlap.generate import generate
from overlap.stats import compute_overlap_statistics


def test_generator_crisp_control():
    """When p_ov = 0, no nodes should belong to multiple communities."""
    n, c = 200, 4
    hyperedges, truth = generate(n=n, c=c, p_ov=0.0, xi=0.1, seed=42)
    
    stats = compute_overlap_statistics(truth, n=n)
    assert stats["num_overlapping_nodes"] == 0
    assert stats["percent_overlapping"] == 0.0
    assert stats["avg_memberships"] == 1.0


def test_generator_planted_overlap():
    """When p_ov = 0.25, approximately 25% of nodes belong to >= 2 communities."""
    n, c = 200, 4
    hyperedges, truth = generate(n=n, c=c, p_ov=0.25, xi=0.1, seed=42)
    
    stats = compute_overlap_statistics(truth, n=n)
    assert stats["num_overlapping_nodes"] == int(0.25 * n)
    assert stats["percent_overlapping"] == 25.0
    assert stats["avg_memberships"] > 1.0
    
    # Check hyperedge validity
    for he in hyperedges:
        assert len(he) >= 2
        for node in he:
            assert 0 <= node < n


def test_generator_seed_reproducibility():
    """Identical seeds must generate identical outputs."""
    h1, t1 = generate(n=100, c=3, p_ov=0.2, xi=0.2, seed=123)
    h2, t2 = generate(n=100, c=3, p_ov=0.2, xi=0.2, seed=123)
    
    assert h1 == h2
    assert t1 == t2
