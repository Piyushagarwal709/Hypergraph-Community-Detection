"""
overlap/stats.py
Calculates overlap statistics on overlapping community partitions:
- Number and percentage of overlapping nodes
- Average and maximum memberships per node
- Community size distributions
"""

import numpy as np


def compute_overlap_statistics(communities, n):
    """
    Computes key structural statistics on an overlapping community cover.
    
    Args:
        communities: list[set[int]], where each set contains node IDs in that community.
        n: total number of nodes in the hypergraph.
        
    Returns:
        dict containing:
            - num_nodes: total node count n
            - num_communities: number of communities c
            - num_overlapping_nodes: count of nodes in >= 2 communities
            - percent_overlapping: percentage of nodes with multi-membership
            - avg_memberships: average memberships per node
            - max_memberships: maximum memberships of any single node
            - per_node_counts: list of length n, communities joined by each node
            - community_sizes: list of sizes of each community
    """
    c = len(communities)
    per_node_counts = [0] * n
    
    for comm in communities:
        for node in comm:
            if 0 <= node < n:
                per_node_counts[node] += 1
                
    overlapping_nodes = [i for i in range(n) if per_node_counts[i] > 1]
    num_overlapping = len(overlapping_nodes)
    percent_overlapping = (num_overlapping / n * 100.0) if n > 0 else 0.0
    avg_memberships = float(np.mean(per_node_counts)) if n > 0 else 0.0
    max_memberships = int(np.max(per_node_counts)) if n > 0 else 0
    community_sizes = [len(comm) for comm in communities]
    
    return {
        "num_nodes": n,
        "num_communities": c,
        "num_overlapping_nodes": num_overlapping,
        "percent_overlapping": round(percent_overlapping, 2),
        "avg_memberships": round(avg_memberships, 3),
        "max_memberships": max_memberships,
        "overlapping_node_ids": overlapping_nodes,
        "per_node_counts": per_node_counts,
        "community_sizes": community_sizes
    }
