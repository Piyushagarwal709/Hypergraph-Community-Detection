"""
overlap/rules.py
Implements assignment rules (R0, R1, R2, R3, R4) that turn continuous
fuzzy membership matrix U (n x c) into overlapping community covers.

Rules defined in Section 5.2 of PROJECT_BRIEF.md / final.md:
  R0: argmax (crisp baseline from FMHMO)
  R1: fixed threshold tau (tau in {0.1, 0.2, 0.3, 0.4}, always includes argmax)
  R2: relative threshold rho * u_max (rho in {0.5, 0.6, 0.7, 0.8, 0.9})
  R3: top-k communities (k in {1, 2, 3}, only where u > 0)
  R4: data-driven modularity maximization across candidate thresholds
"""

import numpy as np


def assign(U, rule="R0", param=None, **kwargs):
    """
    Transforms membership matrix U (n x c) into overlapping communities.
    
    Args:
        U: np.ndarray of shape (n, c), where U[i, j] is membership of node i in community j.
        rule: str, one of {"R0", "R1", "R2", "R3", "R4"}.
        param: rule-specific parameter:
            - R0: None
            - R1: float tau (e.g. 0.2)
            - R2: float rho (e.g. 0.7)
            - R3: int k (e.g. 2)
            - R4: candidate param list (optional, default [0.1, 0.2, 0.3, 0.4])
            
    Returns:
        communities: list[set[int]] of length c, where communities[j] contains
                     all node IDs that belong to community j.
                     
    Caveat:
        Because u(v, .) sums to 1 over neighbor communities only, a fixed tau (R1)
        behaves differently for nodes with 2 neighbor communities (where random chance is 0.5)
        versus 6 neighbor communities (where random chance is ~0.17). R2 (relative threshold)
        and R3 (top-k) address this variation.
    """
    U = np.asarray(U, dtype=float)
    n, c = U.shape
    
    # Initialize c community sets
    communities = [set() for _ in range(c)]
    
    if rule == "R0":
        # Crisp argmax assignment
        best_comm = np.argmax(U, axis=1)
        for i in range(n):
            communities[best_comm[i]].add(i)
            
    elif rule == "R1":
        # Fixed threshold tau: join every community with u(v, C) >= tau,
        # always including argmax to guarantee no node is left orphaned.
        tau = 0.2 if param is None else float(param)
        for i in range(n):
            best_c = int(np.argmax(U[i]))
            communities[best_c].add(i)
            for j in range(c):
                if j != best_c and U[i, j] >= tau:
                    communities[j].add(i)
                    
    elif rule == "R2":
        # Relative threshold rho: join all C where u(v, C) >= rho * u_max
        rho = 0.7 if param is None else float(param)
        for i in range(n):
            u_max = np.max(U[i])
            threshold = rho * u_max
            joined = False
            for j in range(c):
                if U[i, j] >= threshold and U[i, j] > 0.0:
                    communities[j].add(i)
                    joined = True
            if not joined:
                # Guarantee at least argmax is included
                communities[int(np.argmax(U[i]))].add(i)
                
    elif rule == "R3":
        # Top-k: join the best k communities, only those with u > 0
        k = 2 if param is None else int(param)
        for i in range(n):
            row = U[i]
            # Get sorted indices descending
            sorted_indices = np.argsort(-row)
            count = 0
            for idx in sorted_indices:
                if row[idx] > 0.0 and count < k:
                    communities[idx].add(i)
                    count += 1
                elif count >= k:
                    break
            if count == 0:
                communities[int(np.argmax(row))].add(i)
                
    elif rule == "R4":
        # Data-driven selection: sweeps tau in {0.1, 0.2, 0.3, 0.4} and selects
        # the partition that maximizes internal edge density / coverage balance
        hyperedges = kwargs.get("hyperedges", None)
        candidate_taus = [0.1, 0.2, 0.3, 0.4] if param is None else param
        best_cover = None
        best_score = -1e9
        
        for candidate_tau in candidate_taus:
            candidate_cover = assign(U, rule="R1", param=candidate_tau)
            score = evaluate_overlapping_modularity(candidate_cover, hyperedges, n)
            if score > best_score:
                best_score = score
                best_cover = candidate_cover
                
        return best_cover if best_cover is not None else assign(U, rule="R0")
        
    else:
        raise ValueError(f"Unknown rule '{rule}'. Must be one of 'R0', 'R1', 'R2', 'R3', 'R4'.")
        
    return communities


def evaluate_overlapping_modularity(communities, hyperedges, n):
    """
    Evaluates quality of an overlapping hypergraph cover without ground truth.
    Rewards hyperedges whose member nodes are covered together by at least one community,
    penalizing excessive community sizes (Shen/EQ style modularity adaptation).
    """
    if hyperedges is None or len(hyperedges) == 0:
        return 0.0
        
    # Count hyperedges completely covered by at least one community
    covered = 0
    for he in hyperedges:
        he_set = set(he)
        for comm in communities:
            if he_set.issubset(comm):
                covered += 1
                break
                
    coverage_ratio = covered / len(hyperedges)
    
    # Redundancy penalty: penalty if nodes are assigned to too many communities
    memberships_per_node = [sum([1 for comm in communities if i in comm]) for i in range(n)]
    avg_memberships = np.mean(memberships_per_node)
    
    return coverage_ratio - 0.1 * max(0.0, avg_memberships - 1.0)
