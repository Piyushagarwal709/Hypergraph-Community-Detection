"""
overlap/generate.py
Synthetic hypergraph generator with planted overlapping community structure.

Follows the design specified in PROJECT_BRIEF.md / final.md:
- n nodes, c communities
- A fraction p_ov of nodes belongs to multiple (2 or 3) communities (planted overlap)
- When p_ov = 0, every node belongs to exactly 1 community (crisp control)
- Hyperedges of varied sizes (2..6) drawn mostly from within a single community (prob 1 - xi)
  and mixed across communities with noise prob xi
"""

import random
from collections import defaultdict
import numpy as np


def generate_planted_overlap_hypergraph(
    n=500,
    c=5,
    p_ov=0.2,
    xi=0.2,
    m=None,
    min_edge_size=2,
    max_edge_size=5,
    seed=42
):
    """
    Generates a synthetic hypergraph with planted overlapping community ground truth.
    
    Args:
        n: int, total number of nodes (0 .. n-1)
        c: int, number of communities
        p_ov: float in [0.0, 1.0], fraction of nodes with multi-community membership
        xi: float in [0.0, 1.0], noise/mixing parameter (prob of drawing random non-community hyperedge)
        m: int, number of hyperedges to generate (default: 3 * n)
        min_edge_size: int, minimum hyperedge size
        max_edge_size: int, maximum hyperedge size
        seed: int, random seed
        
    Returns:
        hyperedges: list[list[int]], list of hyperedges (node IDs 0 .. n-1)
        truth_communities: list[set[int]], ground-truth communities
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    if m is None:
        m = int(3.0 * n)
        
    truth_communities = [set() for _ in range(c)]
    node_to_comms = defaultdict(set)
    
    # 1. Base assignment: partition n nodes evenly into c communities
    nodes = list(range(n))
    rng.shuffle(nodes)
    
    for i, node in enumerate(nodes):
        primary_c = i % c
        truth_communities[primary_c].add(node)
        node_to_comms[node].add(primary_c)
        
    # 2. Plant overlap: choose p_ov fraction of nodes to join 1 or 2 extra communities
    num_overlap = int(round(p_ov * n))
    if num_overlap > 0 and c > 1:
        overlap_candidates = rng.sample(nodes, num_overlap)
        for node in overlap_candidates:
            primary_c = list(node_to_comms[node])[0]
            other_comms = [other for other in range(c) if other != primary_c]
            # Choose 1 or 2 additional communities
            extra_count = 1 if (len(other_comms) < 2 or rng.random() < 0.7) else 2
            extra_comms = rng.sample(other_comms, extra_count)
            for ec in extra_comms:
                truth_communities[ec].add(node)
                node_to_comms[node].add(ec)
                
    # 3. Generate hyperedges
    hyperedges = []
    seen_edges = set()
    
    # Pre-build list of nodes in each community for fast sampling
    comm_node_lists = [list(truth_communities[i]) for i in range(c)]
    
    # Ensure every node is included in at least one hyperedge
    for node in range(n):
        comms = list(node_to_comms[node])
        chosen_c = rng.choice(comms)
        c_nodes = comm_node_lists[chosen_c]
        size = rng.randint(min_edge_size, min(max_edge_size, len(c_nodes)))
        if len(c_nodes) >= size:
            other_nodes = rng.sample([x for x in c_nodes if x != node], size - 1)
            he = tuple(sorted([node] + other_nodes))
            if he not in seen_edges and len(he) >= 2:
                seen_edges.add(he)
                hyperedges.append(list(he))
                
    # Sample remaining hyperedges
    attempts = 0
    while len(hyperedges) < m and attempts < m * 10:
        attempts += 1
        size = rng.randint(min_edge_size, max_edge_size)
        
        if rng.random() < xi:
            # Noise hyperedge: sample completely at random from all nodes
            he = tuple(sorted(rng.sample(range(n), size)))
        else:
            # Community hyperedge: sample from community nodes
            chosen_c = rng.randint(0, c - 1)
            c_nodes = comm_node_lists[chosen_c]
            if len(c_nodes) < size:
                continue
            he = tuple(sorted(rng.sample(c_nodes, size)))
            
        if he not in seen_edges and len(he) >= 2:
            seen_edges.add(he)
            hyperedges.append(list(he))
            
    return hyperedges, truth_communities


# Standard API alias
def generate(n, c, p_ov, xi, seed=42, **kwargs):
    """Interface contract compliant function."""
    return generate_planted_overlap_hypergraph(
        n=n, c=c, p_ov=p_ov, xi=xi, seed=seed, **kwargs
    )
