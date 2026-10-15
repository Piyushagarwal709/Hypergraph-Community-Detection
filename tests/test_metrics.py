"""
tests/test_metrics.py
Unit tests for overlap/metrics.py:
Tests ONMI, Omega, overlapping F1, and overlap-node PRF.
"""

import pytest
import numpy as np
from overlap.metrics import onmi, omega, overlap_f1, overlap_node_prf


def test_identical_covers():
    """Identical community covers must yield 1.0 on all metrics."""
    cover = [{0, 1, 2}, {2, 3, 4}, {4, 5, 6}]
    n = 7
    
    assert np.isclose(onmi(cover, cover, n), 1.0, atol=1e-5)
    assert np.isclose(omega(cover, cover, n), 1.0, atol=1e-5)
    assert np.isclose(overlap_f1(cover, cover), 1.0, atol=1e-5)
    
    prf = overlap_node_prf(cover, cover, n)
    assert prf["precision"] == 1.0
    assert prf["recall"] == 1.0
    assert prf["f1"] == 1.0


def test_hand_computed_overlap_prf():
    """
    Ground truth has node 2 and node 4 overlapping:
      Truth: C0 = {0, 1, 2}, C1 = {2, 3, 4}, C2 = {4, 5} -> overlap nodes = {2, 4}
    Prediction:
      Pred:  C0 = {0, 1, 2}, C1 = {2, 3}, C2 = {4, 5} -> overlap nodes = {2}
    TP = {2} (1), FP = 0, FN = {4} (1)
    Precision = 1/1 = 1.0, Recall = 1/2 = 0.5, F1 = 2/3 ≈ 0.6667
    """
    truth = [{0, 1, 2}, {2, 3, 4}, {4, 5}]
    pred = [{0, 1, 2}, {2, 3}, {4, 5}]
    n = 6
    
    prf = overlap_node_prf(pred, truth, n)
    assert prf["precision"] == 1.0
    assert np.isclose(prf["recall"], 0.5, atol=1e-4)
    assert np.isclose(prf["f1"], 2.0 / 3.0, atol=1e-4)


def test_disjoint_crisp_omega_matches_ari_concept():
    """Omega index on completely disjoint non-overlapping covers."""
    truth = [{0, 1, 2}, {3, 4, 5}]
    # Independent random partition
    pred = [{0, 3}, {1, 4}, {2, 5}]
    n = 6
    
    om = omega(pred, truth, n)
    # Completely misaligned should be negative or near zero
    assert om < 0.2
