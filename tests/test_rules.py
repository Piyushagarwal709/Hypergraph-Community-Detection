"""
tests/test_rules.py
Unit tests for overlap/rules.py and overlap/stats.py:
Tests R0, R1, R2, R3, and R4 on hand-computed synthetic matrices.
"""

import pytest
import numpy as np
from overlap.rules import assign
from overlap.stats import compute_overlap_statistics


@pytest.fixture
def sample_membership_matrix():
    """
    Toy 4-node, 3-community matrix:
    Node 0: [0.60, 0.40, 0.00] (Bridge between 0 and 1)
    Node 1: [1.00, 0.00, 0.00] (Pure community 0)
    Node 2: [0.10, 0.80, 0.10] (Predominantly community 1, slight mass on 0 and 2)
    Node 3: [0.00, 0.00, 1.00] (Pure community 2)
    """
    return np.array([
        [0.60, 0.40, 0.00],
        [1.00, 0.00, 0.00],
        [0.10, 0.80, 0.10],
        [0.00, 0.00, 1.00]
    ], dtype=float)


def test_rule_R0_argmax(sample_membership_matrix):
    """R0 must produce a crisp, disjoint partition (no overlapping nodes)."""
    U = sample_membership_matrix
    cov = assign(U, rule="R0")
    
    # Community 0: nodes 0, 1
    # Community 1: node 2
    # Community 2: node 3
    assert cov[0] == {0, 1}
    assert cov[1] == {2}
    assert cov[2] == {3}
    
    stats = compute_overlap_statistics(cov, n=4)
    assert stats["num_overlapping_nodes"] == 0
    assert stats["percent_overlapping"] == 0.0
    assert stats["avg_memberships"] == 1.0


def test_rule_R1_fixed_threshold(sample_membership_matrix):
    """
    R1 joins all communities with u >= tau (always including argmax).
    Node 0: [0.6, 0.4, 0.0]
    With tau = 0.35: joins {0, 1}
    With tau = 0.50: joins {0}
    """
    U = sample_membership_matrix
    
    # tau = 0.35
    cov_35 = assign(U, rule="R1", param=0.35)
    assert 0 in cov_35[0] and 0 in cov_35[1]  # Node 0 overlaps in comm 0 & 1
    assert 1 in cov_35[0] and 1 not in cov_35[1]
    assert 2 in cov_35[1] and 2 not in cov_35[0]
    
    stats_35 = compute_overlap_statistics(cov_35, n=4)
    assert stats_35["num_overlapping_nodes"] == 1  # Node 0
    assert stats_35["overlapping_node_ids"] == [0]
    assert stats_35["percent_overlapping"] == 25.0
    
    # tau = 0.50 -> Node 0 no longer qualifies for comm 1 (0.40 < 0.50)
    cov_50 = assign(U, rule="R1", param=0.50)
    assert cov_50[0] == {0, 1}
    assert cov_50[1] == {2}
    assert cov_50[2] == {3}


def test_rule_R2_relative_threshold(sample_membership_matrix):
    """
    R2 joins all communities with u >= rho * u_max.
    Node 0: u_max = 0.60
    rho = 0.60 -> threshold 0.36 -> 0.40 >= 0.36 -> joins {0, 1}
    rho = 0.80 -> threshold 0.48 -> 0.40 < 0.48 -> joins only {0}
    """
    U = sample_membership_matrix
    
    # rho = 0.60
    cov_60 = assign(U, rule="R2", param=0.60)
    assert 0 in cov_60[0] and 0 in cov_60[1]
    
    # rho = 0.80
    cov_80 = assign(U, rule="R2", param=0.80)
    assert 0 in cov_80[0] and 0 not in cov_80[1]


def test_rule_R3_top_k(sample_membership_matrix):
    """
    R3 joins the top-k communities with u > 0.
    Node 0: [0.6, 0.4, 0.0] -> k=2 joins {0, 1}
    Node 1: [1.0, 0.0, 0.0] -> k=2 joins {0} (since other entries are 0)
    Node 2: [0.1, 0.8, 0.1] -> k=2 joins {1, 0} or {1, 2}
    """
    U = sample_membership_matrix
    
    cov_k1 = assign(U, rule="R3", param=1)
    assert cov_k1[0] == {0, 1}
    assert cov_k1[1] == {2}
    assert cov_k1[2] == {3}
    
    cov_k2 = assign(U, rule="R3", param=2)
    assert 0 in cov_k2[0] and 0 in cov_k2[1]
    # Node 1 only has 1 non-zero entry, should NOT join a 2nd community with u=0
    assert cov_k2[0] == {0, 1} or 1 in cov_k2[0]
    assert 1 not in cov_k2[1] and 1 not in cov_k2[2]
