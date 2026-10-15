"""
overlap/baselines.py
Adapters and baseline implementations for comparative overlapping community detection.

Includes:
1. Clique-Expansion + Soft-Assignment Baseline (Expansion-Overlap):
   Standard clique projection of the hypergraph followed by spectral/fuzzy modularity.
2. Adapter stubs and wrappers for external baselines (Hy-MMSBM and Hypergraph-MT).

Note on baseline advantage:
Methods like Hy-MMSBM are probabilistic generative models explicitly designed from first
principles for mixed membership hypergraphs, giving them an inherent architectural advantage
over heuristic post-processing methods.
"""

import numpy as np
import networkx as nx
from overlap.rules import assign


def expansion_overlap_baseline(hyperedges, n, c=None, tau=0.25, seed=42):
    """
    Standard hypergraph community detection baseline:
    1. Projects hypergraph onto 2-section (clique expansion graph).
    2. Computes node degree and weighted transition matrix.
    3. Derives node-to-community soft membership via diffusion/projection.
    4. Applies soft thresholding to obtain overlapping communities.
    """
    import community as community_louvain
    
    # 1. Build clique expansion graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for he in hyperedges:
        he_nodes = [node for node in he if 0 <= node < n]
        weight = 1.0 / (len(he_nodes) - 1) if len(he_nodes) > 1 else 1.0
        for i in range(len(he_nodes)):
            u = he_nodes[i]
            for j in range(i + 1, len(he_nodes)):
                v = he_nodes[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += weight
                else:
                    G.add_edge(u, v, weight=weight)
                    
    # 2. Partition using Louvain
    if G.number_of_edges() == 0:
        return [set(range(n))]
        
    part = community_louvain.best_partition(G, weight='weight', random_state=seed)
    detected_c = max(part.values()) + 1 if part else 1
    if c is None:
        c = detected_c
        
    # 3. Derive soft membership matrix U_base (n x c)
    # Node membership in community C is proportional to weighted edges connecting to C
    U_base = np.zeros((n, c), dtype=float)
    for u in range(n):
        degree_c = np.zeros(c, dtype=float)
        total_deg = 0.0
        for v in G.neighbors(u):
            w = G[u][v].get('weight', 1.0)
            target_c = part.get(v, 0)
            if target_c < c:
                degree_c[target_c] += w
                total_deg += w
        if total_deg > 0:
            U_base[u, :] = degree_c / total_deg
        else:
            U_base[u, part.get(u, 0) % c] = 1.0
            
    # 4. Apply thresholding rule
    return assign(U_base, rule="R1", param=tau)
