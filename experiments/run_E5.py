"""
experiments/run_E5.py
Experiment E5: Real-World Hypergraph Overlap Analysis.
Applies FMHMO-Overlap to real-world datasets:
- Evaluates % overlapping nodes and membership distributions across rules.
- Extracts concrete, interpretable bridge-node case studies (e.g. in hospital, senate_committees).
- Transparently adheres to guideline: Real datasets have crisp ground-truth,
  so we report structural overlap statistics rather than claiming overlap accuracy.
Saves results CSV and case study JSON to results/.
"""

import os
import sys
import json
import csv
import contextlib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMHMO_PATH = os.path.join(PROJECT_ROOT, "FMHMO", "FMHMO")
if FMHMO_PATH not in sys.path:
    sys.path.insert(0, FMHMO_PATH)

import FMHMO_utils as HG_util
import FMHMO_function as HG_func
import community as community_louvain
import networkx as nx

from overlap.memberships import compute_full_membership_matrix
from overlap.rules import assign
from overlap.stats import compute_overlap_statistics


@contextlib.contextmanager
def cd_context(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def analyze_real_dataset(net_name, gamma=0.5, seed=42):
    """Executes FMHMO-Overlap on a real-world hypergraph."""
    with cd_context(FMHMO_PATH):
        network, network_name, groundtruth_path = HG_util.network_set(net=net_name, network_type="real")
        H = HG_func.createHyperGraph(network)
        
        import logging
        dummy_log = logging.getLogger("e5_logger")
        dummy_log.setLevel(logging.CRITICAL)
        
        HG_info = HG_util.HG_info_obtain(dummy_log, H)
        WG, WG_nx, G_info = HG_util.HG_to_WG(HG_info, gamma)
        WG_info, Q_info = HG_util.WG_info_obtain(WG, G_info)
        FMHMO_params = HG_util.FMHMO_params_set()
        FMHMO_params['Independent_Runs'] = 1
        
        # Louvain partition on reduced graph
        if nx.number_connected_components(WG_nx) == 1:
            ml = community_louvain.best_partition(graph=WG_nx, weight='weight', random_state=seed)
            best_membership_c = [ml[i] for i in range(len(ml))]
        else:
            ml = WG.community_multilevel(weights="weight", resolution=1.0)
            best_membership_c = [-1] * Q_info['n']
            for cno, cnodes in enumerate(ml):
                for node in cnodes:
                    best_membership_c[node] = cno
                    
        X_res, X_part, X_NCM_res, X_NCM_part, X_NCR_res, X_NCR_part = HG_util.WGCD_to_HGCD(
            HG_info, WG_info, best_membership_c, FMHMO_params
        )
        
        c = len(set(X_res))
        U = compute_full_membership_matrix(HG_info, X_res, c=c)
        
        return {
            "name": net_name,
            "n": HG_info["H_n"],
            "m": len(HG_info["H_edges"]),
            "c": c,
            "U": U,
            "crisp": X_res,
            "node_HE_dict": HG_info["node_HE_dict"]
        }


def run_experiment_E5(datasets=None, out_dir=None):
    if datasets is None:
        datasets = ["hospital", "workspace", "senate_committees", "primary_school", "high_school"]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
    os.makedirs(out_dir, exist_ok=True)
    
    records = []
    case_studies = []
    
    print("=" * 70)
    print("  EXPERIMENT E5: REAL-WORLD HYPERGRAPH OVERLAPPING ANALYSIS")
    print("=" * 70)
    
    for net in datasets:
        print(f"\n[+] Analyzing dataset: '{net}'...")
        res = analyze_real_dataset(net, gamma=0.5, seed=42)
        n, c, U = res["n"], res["c"], res["U"]
        
        # Apply rules
        cov_r0 = assign(U, rule="R0")
        cov_r1 = assign(U, rule="R1", param=0.25)
        cov_r2 = assign(U, rule="R2", param=0.7)
        cov_r3 = assign(U, rule="R3", param=2)
        
        stats_r0 = compute_overlap_statistics(cov_r0, n)
        stats_r1 = compute_overlap_statistics(cov_r1, n)
        stats_r2 = compute_overlap_statistics(cov_r2, n)
        stats_r3 = compute_overlap_statistics(cov_r3, n)
        
        records.append({
            "Dataset": net,
            "Nodes_n": n,
            "Hyperedges_m": res["m"],
            "Communities_c": c,
            "R0_pct_overlap": stats_r0["percent_overlapping"],
            "R1_tau0.25_pct_overlap": stats_r1["percent_overlapping"],
            "R1_avg_memberships": stats_r1["avg_memberships"],
            "R2_rho0.7_pct_overlap": stats_r2["percent_overlapping"],
            "R2_avg_memberships": stats_r2["avg_memberships"],
            "R3_top2_pct_overlap": stats_r3["percent_overlapping"]
        })
        
        print(f"    Nodes={n}, Hyperedges={res['m']}, Communities={c}")
        print(f"    R1 (tau=0.25): {stats_r1['percent_overlapping']}% overlap, avg memberships={stats_r1['avg_memberships']}")
        print(f"    R2 (rho=0.70): {stats_r2['percent_overlapping']}% overlap, avg memberships={stats_r2['avg_memberships']}")
        
        # Find the most prominent bridge node (highest non-max membership)
        node_HE = res["node_HE_dict"]
        bridge_candidates = []
        for i in range(n):
            row = U[i]
            sorted_m = sorted(row, reverse=True)
            if len(sorted_m) > 1 and sorted_m[1] > 0.05:
                bridge_candidates.append((sorted_m[1], i, row, len(node_HE[i])))
                
        bridge_candidates.sort(key=lambda x: x[0], reverse=True)
        if bridge_candidates:
            second_m, best_node, best_row, deg = bridge_candidates[0]
            top_comms = [
                {"community": int(c_idx), "membership": round(float(best_row[c_idx]), 4)}
                for c_idx in np.argsort(-best_row)[:3] if best_row[c_idx] > 0.01
            ]
            case_studies.append({
                "dataset": net,
                "node_id": int(best_node),
                "incident_hyperedges_count": int(deg),
                "memberships": top_comms
            })
            print(f"    [Case Study] Node {best_node}: Degree={deg}, Memberships: {[(x['community'], x['membership']) for x in top_comms]}")
            
    # Save CSV
    csv_path = os.path.join(out_dir, "E5_real_world_overlap.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved E5 summary table to: {csv_path}")
    
    # Save JSON Case Studies
    json_path = os.path.join(out_dir, "E5_case_studies.json")
    with open(json_path, "w") as f:
        json.dump(case_studies, f, indent=2)
    print(f"[OK] Saved E5 case studies to: {json_path}")
    
    return records


if __name__ == "__main__":
    run_experiment_E5()
