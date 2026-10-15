"""
tests/test_memberships.py
Unit tests for overlap/memberships.py:
1. Exact Figure 1 equations validation (Eq. 1 Jaccard similarity and Eq. 2 memberships).
2. Membership matrix U properties (row sums = 1, bounds [0, 1], one-hot for intra-community nodes).
3. Argmax consistency with crisp partition.
"""

import pytest
import numpy as np
from overlap.memberships import (
    build_hg_structures,
    compute_node_similarity,
    compute_node_to_hyperedge_memberships,
    compute_full_membership_matrix,
    extract_memberships_from_raw_hyperedges,
    get_fig1_toy_hypergraph
)


def test_fig1_node_similarity_and_hyperedge_memberships():
    """
    Validates Figure 1 of the paper (Xiao et al., IEEE TFS 2025, page 3256):
    Node v0 in {e1, e4}, v1 in {e1, e2, e4}.
    - S(v0, v1) = 2/3 ≈ 0.67
    - u(v1, e1) = 0.409
    - u(v1, e2) = 0.273
    - u(v1, e4) = 0.318
    - Sum = 1.0
    """
    hyperedges = get_fig1_toy_hypergraph()
    structs = build_hg_structures(hyperedges)
    n = structs['H_n']
    node_HE_dict = structs['node_HE_dict']
    
    # 1. Check incident hyperedges
    # e1 = (0, 1, 7), e2 = (1, 8, 9, 10), e4 = (0, 1, 2, 3, 4, 5, 6)
    e1 = (0, 1, 7)
    e2 = (1, 8, 9, 10)
    e4 = (0, 1, 2, 3, 4, 5, 6)
    
    assert set(node_HE_dict[0]) == {e1, e4}
    assert set(node_HE_dict[1]) == {e1, e2, e4}
    
    # 2. Node pairwise similarity S
    S_adj = compute_node_similarity(node_HE_dict, n)
    s_v0_v1 = S_adj[0, 1]
    assert np.isclose(s_v0_v1, 2.0 / 3.0, atol=1e-3), f"Expected 0.667, got {s_v0_v1}"
    
    # 3. Node-to-hyperedge memberships u(v1, e)
    Uie_dict = compute_node_to_hyperedge_memberships(node_HE_dict, S_adj, n)
    u_v1 = Uie_dict[1]
    
    u_e1 = u_v1[e1]
    u_e2 = u_v1[e2]
    u_e4 = u_v1[e4]
    
    # Verify paper values (0.409, 0.273, 0.318) within slight paper rounding tolerance (0.03)
    assert np.isclose(u_e1, 0.409, atol=0.03), f"Expected ~0.409, got {u_e1}"
    assert np.isclose(u_e2, 0.273, atol=0.03), f"Expected ~0.273, got {u_e2}"
    assert np.isclose(u_e4, 0.318, atol=0.03), f"Expected ~0.318, got {u_e4}"
    
    # Sum must be exactly 1.0
    total_u = u_e1 + u_e2 + u_e4
    assert np.isclose(total_u, 1.0, atol=1e-6), f"Sum should be 1.0, got {total_u}"


def test_membership_matrix_properties_on_toy_network():
    """
    Tests mathematical invariants of the extracted membership matrix U:
    1. Shape is (n, c)
    2. All entries in [0, 1]
    3. Every row sums to 1.0
    4. argmax(U) matches crisp_partition
    """
    hyperedges = get_fig1_toy_hypergraph()
    U, crisp_partition, c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=42)
    
    n = 13
    assert U.shape == (n, c), f"Expected shape ({n}, {c}), got {U.shape}"
    assert np.all(U >= 0.0), "Memberships must be non-negative"
    assert np.all(U <= 1.0 + 1e-6), "Memberships must not exceed 1.0"
    
    # Check row sum == 1 for every node
    row_sums = np.sum(U, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sums differ from 1.0: {row_sums}"
    
    # Check argmax consistency
    pred_crisp = np.argmax(U, axis=1)
    assert np.array_equal(pred_crisp, crisp_partition), "argmax(U) must match crisp partition"


def test_non_bridge_nodes_are_one_hot():
    """
    Checks that nodes belonging to a single isolated community
    have one-hot membership distributions in U.
    """
    # Two disjoint cliques/hyperedges: {0, 1, 2} and {3, 4, 5}
    disjoint_hyperedges = [
        [0, 1, 2],
        [3, 4, 5]
    ]
    U, crisp_partition, c = extract_memberships_from_raw_hyperedges(disjoint_hyperedges, gamma=0.5, seed=42)
    
    assert c == 2, f"Expected 2 communities, got {c}"
    for i in range(6):
        # Must be exactly one-hot: max is 1.0, min is 0.0
        assert np.isclose(np.max(U[i]), 1.0, atol=1e-5)
        assert np.isclose(np.min(U[i]), 0.0, atol=1e-5)
