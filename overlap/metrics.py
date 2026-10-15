"""
overlap/metrics.py
Implements overlap-aware evaluation metrics:
1. Overlapping NMI (ONMI) - LFK / McDaid formulation
2. Omega Index - Overlapping pair agreement (generalization of Rand Index)
3. Overlapping F1 - Best-match F1 averaged symmetrically
4. Overlap-node Precision, Recall, F1 - Binary overlap identification accuracy
"""

import math
from collections import defaultdict
import numpy as np


def _binary_entropy(p):
    """Computes binary entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _conditional_entropy_binary(x, y, n):
    """
    Computes conditional entropy H(X | Y) between two binary indicator variables
    representing community memberships over n nodes.
    """
    # x and y are sets of node IDs
    n11 = len(x & y)
    n10 = len(x - y)
    n01 = len(y - x)
    n00 = n - (n11 + n10 + n01)
    
    p11, p10 = n11 / n, n10 / n
    p01, p00 = n01 / n, n00 / n
    
    p_y1 = (n11 + n01) / n
    p_y0 = (n10 + n00) / n
    
    # H(X, Y)
    h_xy = 0.0
    for p in [p11, p10, p01, p00]:
        if p > 0.0:
            h_xy -= p * math.log2(p)
            
    # H(Y)
    h_y = 0.0
    for p in [p_y1, p_y0]:
        if p > 0.0:
            h_y -= p * math.log2(p)
            
    # H(X | Y) = H(X, Y) - H(Y)
    h_x_given_y = max(0.0, h_xy - h_y)
    
    # Check if x is independent or negatively correlated
    hx = _binary_entropy(len(x) / n)
    if h_x_given_y > hx:
        h_x_given_y = hx
        
    return h_x_given_y


def onmi(pred, truth, n):
    """
    Overlapping Normalized Mutual Information (LFK / McDaid formulation).
    
    Args:
        pred: list[set[int]], predicted community cover
        truth: list[set[int]], ground truth community cover
        n: int, total number of nodes
        
    Returns:
        float in [0, 1], where 1 indicates identical covers.
    """
    # Filter empty sets
    pred = [s for s in pred if len(s) > 0]
    truth = [s for s in truth if len(s) > 0]
    
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
        
    # Check identical covers
    if len(pred) == len(truth):
        sorted_pred = sorted([sorted(list(s)) for s in pred])
        sorted_truth = sorted([sorted(list(s)) for s in truth])
        if sorted_pred == sorted_truth:
            return 1.0
            
    # Compute H(X) and H(Y)
    h_pred = sum([_binary_entropy(len(c) / n) for c in pred])
    h_truth = sum([_binary_entropy(len(c) / n) for c in truth])
    
    if h_pred == 0.0 and h_truth == 0.0:
        return 1.0
    if h_pred == 0.0 or h_truth == 0.0:
        return 0.0
        
    # Compute H(X | Y) = sum_k min_l H(X_k | Y_l)
    h_pred_given_truth = 0.0
    for cp in pred:
        min_cond = min([_conditional_entropy_binary(cp, ct, n) for ct in truth])
        h_pred_given_truth += min_cond
        
    # Compute H(Y | X) = sum_l min_k H(Y_l | X_k)
    h_truth_given_pred = 0.0
    for ct in truth:
        min_cond = min([_conditional_entropy_binary(ct, cp, n) for cp in pred])
        h_truth_given_pred += min_cond
        
    # Standard normalized mutual information formula
    # NMI = 1 - 0.5 * (H(X|Y)/H(X) + H(Y|X)/H(Y))
    term1 = h_pred_given_truth / h_pred if h_pred > 0 else 0.0
    term2 = h_truth_given_pred / h_truth if h_truth > 0 else 0.0
    
    score = 1.0 - 0.5 * (term1 + term2)
    return float(np.clip(score, 0.0, 1.0))


def omega(pred, truth, n):
    """
    Omega Index (standard generalization of the Adjusted Rand Index for overlapping covers).
    Evaluates whether pairs of nodes share the exact same number of communities.
    
    Args:
        pred: list[set[int]], predicted community cover
        truth: list[set[int]], ground truth community cover
        n: int, total number of nodes
        
    Returns:
        float in [-1, 1], where 1 indicates perfect agreement.
    """
    if n <= 1:
        return 1.0
        
    # Filter empty sets
    pred = [s for s in pred if len(s) > 0]
    truth = [s for s in truth if len(s) > 0]
    
    # Check identical covers
    if len(pred) == len(truth):
        sorted_pred = sorted([sorted(list(s)) for s in pred])
        sorted_truth = sorted([sorted(list(s)) for s in truth])
        if sorted_pred == sorted_truth:
            return 1.0
            
    # Count pairwise co-memberships in pred
    pair_pred = defaultdict(int)
    for c in pred:
        c_list = sorted(list(c))
        for i in range(len(c_list)):
            u = c_list[i]
            for j in range(i + 1, len(c_list)):
                v = c_list[j]
                pair_pred[(u, v)] += 1
                
    # Count pairwise co-memberships in truth
    pair_truth = defaultdict(int)
    for c in truth:
        c_list = sorted(list(c))
        for i in range(len(c_list)):
            u = c_list[i]
            for j in range(i + 1, len(c_list)):
                v = c_list[j]
                pair_truth[(u, v)] += 1
                
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 1.0
        
    # Find all pairs that have co-membership > 0 in either pred or truth
    all_pairs = set(pair_pred.keys()) | set(pair_truth.keys())
    
    obs_agree = 0
    pred_counts = defaultdict(int)
    truth_counts = defaultdict(int)
    
    for pair in all_pairs:
        t1 = pair_pred[pair]
        t2 = pair_truth[pair]
        if t1 == t2:
            obs_agree += 1
        pred_counts[t1] += 1
        truth_counts[t2] += 1
        
    # Pairs with 0 in both
    zero_zero_pairs = total_pairs - len(all_pairs)
    obs_agree += zero_zero_pairs
    pred_counts[0] += (total_pairs - len(pair_pred))
    truth_counts[0] += (total_pairs - len(pair_truth))
    
    # Expected agreement
    all_k = set(pred_counts.keys()) | set(truth_counts.keys())
    exp_agree = sum([pred_counts[k] * truth_counts[k] for k in all_k]) / total_pairs
    
    if total_pairs - exp_agree == 0:
        return 1.0
        
    omega_val = (obs_agree - exp_agree) / (total_pairs - exp_agree)
    return float(np.clip(omega_val, -1.0, 1.0))


def overlap_f1(pred, truth):
    """
    Symmetric best-match F1 score for overlapping covers (Shen et al. / Yang & Leskovec).
    
    Args:
        pred: list[set[int]]
        truth: list[set[int]]
        
    Returns:
        float in [0, 1], where 1 indicates perfect match.
    """
    pred = [s for s in pred if len(s) > 0]
    truth = [s for s in truth if len(s) > 0]
    
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
        
    # 1. Match each predicted community to best ground-truth community
    f1_pred_to_truth = []
    for cp in pred:
        best_f1 = 0.0
        for ct in truth:
            intersection = len(cp & ct)
            if intersection > 0:
                f1 = (2.0 * intersection) / (len(cp) + len(ct))
                if f1 > best_f1:
                    best_f1 = f1
        f1_pred_to_truth.append(best_f1)
        
    # 2. Match each ground-truth community to best predicted community
    f1_truth_to_pred = []
    for ct in truth:
        best_f1 = 0.0
        for cp in pred:
            intersection = len(ct & cp)
            if intersection > 0:
                f1 = (2.0 * intersection) / (len(ct) + len(cp))
                if f1 > best_f1:
                    best_f1 = f1
        f1_truth_to_pred.append(best_f1)
        
    score = 0.5 * (np.mean(f1_pred_to_truth) + np.mean(f1_truth_to_pred))
    return float(score)


def overlap_node_prf(pred, truth, n):
    """
    Overlap-Node Precision, Recall, and F1.
    Evaluates how accurately the method identifies which nodes belong to multiple communities.
    
    Args:
        pred: list[set[int]]
        truth: list[set[int]]
        n: int
        
    Returns:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    # Count communities per node
    pred_counts = [0] * n
    for c in pred:
        for node in c:
            if 0 <= node < n:
                pred_counts[node] += 1
                
    truth_counts = [0] * n
    for c in truth:
        for node in c:
            if 0 <= node < n:
                truth_counts[node] += 1
                
    pred_overlap = set([i for i in range(n) if pred_counts[i] > 1])
    truth_overlap = set([i for i in range(n) if truth_counts[i] > 1])
    
    tp = len(pred_overlap & truth_overlap)
    fp = len(pred_overlap - truth_overlap)
    fn = len(truth_overlap - pred_overlap)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if len(truth_overlap) == 0 and len(pred_overlap) == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if len(truth_overlap) == 0 and len(pred_overlap) == 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
