"""
overlap/memberships.py
Exposes the full node-to-community fuzzy membership matrix U (n x c)
from FMHMO (Xiao et al., IEEE TFS 2025).

Unlike original FMHMO which collapses fuzzy memberships into a crisp partition via argmax,
this module preserves the full continuous membership distribution for every node.
"""

import os
import sys
import copy
from collections import Counter
import numpy as np
import networkx as nx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMHMO_PATH = os.path.join(PROJECT_ROOT, "FMHMO", "FMHMO")
if FMHMO_PATH not in sys.path:
    sys.path.insert(0, FMHMO_PATH)

import FMHMO_utils as HG_util
import FMHMO_function as HG_func
import community as community_louvain


def build_hg_structures(hyperedges, n=None):
    """
    Builds hypergraph data structures from a list of hyperedges.
    Each hyperedge is an iterable of node IDs (0-indexed).
    """
    edges_set = [tuple(sorted(list(set(he)))) for he in hyperedges if len(he) > 0]
    all_nodes = set()
    for e in edges_set:
        all_nodes.update(e)
    
    if n is None:
        n = max(all_nodes) + 1 if all_nodes else 0
        
    node_HE_dict = {i: [] for i in range(n)}
    for e in edges_set:
        for node in e:
            if node < n:
                node_HE_dict[node].append(e)
                
    # Group by size for FMHMO format H
    max_size = max([len(e) for e in edges_set]) if edges_set else 0
    H_by_size = [[] for _ in range(max_size + 1)]
    for e in edges_set:
        H_by_size[len(e)].append(set(e))
        
    return {
        "H": H_by_size,
        "H_edges": set(edges_set),
        "H_n": n,
        "node_HE_dict": node_HE_dict
    }


def compute_node_similarity(node_HE_dict, n):
    """
    Eq. 1 in Paper:
    S(v, v') = |Gamma_e(v) cap Gamma_e(v')| / |Gamma_e(v) cup Gamma_e(v')|
    Jaccard similarity of incident hyperedge sets.
    """
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        HEi = set(node_HE_dict[i])
        for j in range(i, n):
            HEj = set(node_HE_dict[j])
            intersection = HEi & HEj
            if len(intersection) > 0:
                union = HEi | HEj
                s_val = len(intersection) / len(union)
                S[i, j] = s_val
                S[j, i] = s_val
    return S


def compute_node_to_hyperedge_memberships(node_HE_dict, S_adj, n):
    """
    Def. 1 / Eq. 2 in Paper:
    u(v, e) = S(v, e) / sum_{e_k in Gamma_e(v)} S(v, e_k)
    where S(v, e) = sum_{v' in e, v' != v} S(v, v') / (|e| - 1).
    Sums to 1 over node's incident hyperedges.
    """
    Uie_dict = {}
    for i in range(n):
        hei_list = node_HE_dict[i]
        Uie = {}
        if len(hei_list) == 0:
            Uie_dict[i] = Uie
            continue
        for he in hei_list:
            other_nodes = [j for j in he if j != i]
            if len(other_nodes) == 0:
                Uie[he] = 1.0
            else:
                Uie[he] = sum([S_adj[i, j] for j in other_nodes]) / len(other_nodes)
        total_s = sum(Uie.values())
        if total_s > 0:
            for he in hei_list:
                Uie[he] /= total_s
        else:
            # Uniform fallback
            for he in hei_list:
                Uie[he] = 1.0 / len(hei_list)
        Uie_dict[i] = Uie
    return Uie_dict


def compute_full_membership_matrix(HG_info, crisp_partition, c=None):
    """
    Extracts the full (n x c) node-to-community membership matrix U
    according to Def. 3 / Eqs. 4-5 and NCM procedure.
    
    Args:
        HG_info: dict with keys 'H_n', 'node_HE_dict', 'Uie_dict', 'Sij_adj'
        crisp_partition: list[int] of length n containing community labels (0 .. c-1)
        c: number of communities (optional, inferred from crisp_partition)
        
    Returns:
        U: numpy array of shape (n, c), where each row sums to 1.0.
           For non-bridge nodes, row is one-hot on their assigned community.
           For bridge nodes, row contains the normalized fuzzy memberships across neighbor communities.
    """
    n = HG_info['H_n']
    X = list(crisp_partition)
    
    if c is None:
        c = len(set(X))
        
    U = np.zeros((n, c), dtype=float)
    node_HE_dict = HG_info['node_HE_dict']
    Uie_dict = HG_info['Uie_dict']
    Sij_adj = HG_info['Sij_adj']
    
    # Precompute incident communities for each node
    for i in range(n):
        i_hes = node_HE_dict[i]
        if not i_hes:
            # Isolated node: assign to its crisp community
            U[i, X[i]] = 1.0
            continue
            
        # Check all communities touched by hyperedges containing node i
        neighbor_communities = set()
        for he in i_hes:
            for node_in_he in he:
                neighbor_communities.add(X[node_in_he])
                
        # Non-bridge node: all incident hyperedges belong to exactly 1 community
        if len(neighbor_communities) <= 1:
            comm = list(neighbor_communities)[0] if neighbor_communities else X[i]
            U[i, comm] = 1.0
            continue
            
        # Bridge node: compute fuzzy memberships across neighbor communities (Eqs. 4-5)
        # 1. Hyperedge-level community affiliation u(e, C)
        i_jcno_mem = {jcno: 0.0 for jcno in neighbor_communities}
        for he in i_hes:
            he_nodes = [node_in_he for node_in_he in he if node_in_he != i]
            if not he_nodes:
                he_nodes = list(he)
            he_cs = [X[hei] for hei in he_nodes]
            heic_counts = Counter(he_cs)
            for cno, count in heic_counts.items():
                if cno in i_jcno_mem:
                    # Def. 2 / Eq. 3: rv = |e cap C| / d(e)
                    rv = count / len(he)
                    # Def. 3 / Eq. 4: attr(v, C) += u(v, e) * rv
                    i_jcno_mem[cno] += (Uie_dict[i][he] * rv)
                    
        # Normalize over neighbor communities
        sum_v = sum(i_jcno_mem.values())
        if sum_v > 0:
            for jcno in i_jcno_mem:
                i_jcno_mem[jcno] /= sum_v
        else:
            for jcno in i_jcno_mem:
                i_jcno_mem[jcno] = 1.0 / len(i_jcno_mem)
                
        # 2. Refine with node-neighbor topological similarity
        i_jcno_mem_1 = {jcno: 0.0 for jcno in neighbor_communities}
        # Find all direct neighbor nodes connected via any hyperedge
        neighbor_nodes = set()
        for he in i_hes:
            neighbor_nodes.update(he)
        neighbor_nodes.discard(i)
        
        for nb in neighbor_nodes:
            nb_comm = X[nb]
            if nb_comm in i_jcno_mem_1:
                i_jcno_mem_1[nb_comm] += Sij_adj[i, nb]
                
        sum_v1 = sum(i_jcno_mem_1.values())
        if sum_v1 > 0:
            for jcno in i_jcno_mem:
                i_jcno_mem[jcno] *= (i_jcno_mem_1[jcno] / sum_v1)
            sum_v2 = sum(i_jcno_mem.values())
            if sum_v2 > 0:
                for jcno in i_jcno_mem:
                    i_jcno_mem[jcno] /= sum_v2
                    
        # Populate row in U
        for jcno, grade in i_jcno_mem.items():
            if jcno < c:
                U[i, jcno] = grade
                
        # Ensure row sums to 1.0 (numerical precision)
        row_sum = np.sum(U[i, :])
        if row_sum > 0:
            U[i, :] /= row_sum
        else:
            U[i, X[i]] = 1.0

    return U


def extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=42):
    """
    Full pipeline to run FMHMO and extract (U, crisp_partition).
    
    Args:
        hyperedges: list of lists of int (0-indexed node IDs)
        gamma: similarity weight parameter
        seed: random seed for Louvain
        
    Returns:
        U: numpy array of shape (n, c), rows sum to 1.
        crisp_partition: list[int] of length n (argmax of U).
        c: number of communities.
    """
    structs = build_hg_structures(hyperedges)
    n = structs['H_n']
    node_HE_dict = structs['node_HE_dict']
    
    # 1. Node similarity S
    Sij_adj = compute_node_similarity(node_HE_dict, n)
    
    # 2. Node to hyperedge memberships
    Uie_dict = compute_node_to_hyperedge_memberships(node_HE_dict, Sij_adj, n)
    
    # 3. Build HG_info dict compatible with FMHMO
    node_js_dict = {}
    for i in range(n):
        jnodes = []
        for jhe in node_HE_dict[i]:
            jnodes.extend(jhe)
        node_js_dict[i] = list(set(jnodes) - {i})
        
    HG_info = {
        'H': structs['H'],
        'H_edges': structs['H_edges'],
        'H_n': n,
        'node_HE_dict': node_HE_dict,
        'Sij_adj': Sij_adj,
        'Uie_dict': Uie_dict,
        'node_js_dict': node_js_dict
    }
    
    # 4. Hyperedge reduction to weighted graph
    WG, WG_nx, G_info = HG_util.HG_to_WG(HG_info, gamma)
    WG_info, Q_info = HG_util.WG_info_obtain(WG, G_info)
    FMHMO_params = HG_util.FMHMO_params_set()
    FMHMO_params['Independent_Runs'] = 1
    
    # 5. Louvain on weighted graph
    if nx.number_connected_components(WG_nx) == 1:
        ml = community_louvain.best_partition(graph=WG_nx, weight='weight', random_state=seed)
        membership_c = [ml[i] for i in range(len(ml))]
    else:
        ml = WG.community_multilevel(weights="weight", resolution=1.0)
        membership_c = [-1] * Q_info['n']
        for cno, cnodes in enumerate(ml):
            for node in cnodes:
                membership_c[node] = cno
                
    # 6. Run FMHMO's recovery
    X_res, X_part, X_NCM_res, X_NCM_part, X_NCR_res, X_NCR_part = HG_util.WGCD_to_HGCD(
        HG_info, WG_info, membership_c, FMHMO_params
    )
    
    # 7. Extract the complete membership matrix U
    c = len(set(X_res))
    U = compute_full_membership_matrix(HG_info, X_res, c=c)
    
    return U, X_res, c


def get_fig1_toy_hypergraph():
    """
    Returns the exact toy hypergraph from Figure 1 of Xiao et al., IEEE TFS 2025.
    13 nodes (0..12), 5 hyperedges:
      e0 = {9, 11}
      e1 = {0, 1, 7}
      e2 = {1, 8, 9, 10}
      e3 = {3, 4, 5, 12}
      e4 = {0, 1, 2, 3, 4, 5, 6}
    """
    e0 = [9, 11]
    e1 = [0, 1, 7]
    e2 = [1, 8, 9, 10]
    e3 = [3, 4, 5, 12]
    e4 = [0, 1, 2, 3, 4, 5, 6]
    return [e0, e1, e2, e3, e4]
