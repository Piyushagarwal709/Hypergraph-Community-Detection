"""
experiments/run_E0.py
Reproduction of FMHMO baseline results on real datasets (hospital, workspace, high_school).
Saves results to results/E0_reproduction/ and compares against paper Tables III and IV.
"""

import sys
import os
import json
import csv
import copy
import logging
import numpy as np

# Ensure root and FMHMO paths are importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMHMO_PATH = os.path.join(PROJECT_ROOT, "FMHMO", "FMHMO")
sys.path.insert(0, FMHMO_PATH)

import networkx as nx
import community as community_louvain
import FMHMO_utils as HG_util
import FMHMO_function as HG_func
import measure as ms
import strictModularity as sm


# Paper ground truth reference values from Table III and Table IV (Xiao et al., IEEE TFS 2025)
PAPER_BENCHMARKS = {
    "hospital": {
        "NMI": 0.145, "ARI": 0.071, "F1": 0.453, "FMI": 0.524, "Qhyp": 0.080
    },
    "workspace": {
        "NMI": 0.836, "ARI": 0.849, "F1": 0.924, "FMI": 0.891, "Qhyp": 0.394
    },
    "high_school": {
        "NMI": 0.959, "ARI": 0.957, "F1": 0.980, "FMI": 0.961, "Qhyp": 0.623
    }
}


import contextlib

@contextlib.contextmanager
def cd_context(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def run_fmhmo_single(net_name, gamma=0.5, runs=5, seed=42):
    """Runs original FMHMO on net_name with given gamma and independent runs."""
    np.random.seed(seed)
    with cd_context(FMHMO_PATH):
        network, network_name, groundtruth_path = HG_util.network_set(net=net_name, network_type="real")
        H = HG_func.createHyperGraph(network)
        
        # Create a minimal logger to avoid spamming disk
        dummy_log = logging.getLogger("reproduce_logger")
        dummy_log.setLevel(logging.CRITICAL)
        
        HG_info = HG_util.HG_info_obtain(dummy_log, H)
        WG, WG_nx, G_info = HG_util.HG_to_WG(HG_info, gamma)
        WG_info, Q_info = HG_util.WG_info_obtain(WG, G_info)
        FMHMO_params = HG_util.FMHMO_params_set()
        FMHMO_params['Independent_Runs'] = runs
        real_mem = HG_util.real_cd_obtain(groundtruth_path)
        
        best_results = {"NMI": 0.0, "ARI": 0.0, "F1": 0.0, "FMI": 0.0, "Qhyp": -1.0}
        all_runs = []
        
        for r in range(runs):
            max_QW = 0.0
            best_membership_c = None
            if nx.number_connected_components(WG_nx) == 1:
                ml = community_louvain.best_partition(graph=WG_nx, weight='weight', random_state=seed + r)
                membership_c = [ml[i] for i in range(len(ml))]
                max_QW = WG.modularity(membership=membership_c, weights='weight')
                best_membership_c = membership_c
            else:
                for _ in range(3):
                    ml = WG.community_multilevel(weights="weight", resolution=1.0)
                    PL = [x for x in ml]
                    membership_c = [-1] * Q_info['n']
                    for cno, cnodes in enumerate(PL):
                        for node in cnodes:
                            membership_c[node] = cno
                    X_QW = ml.modularity
                    if X_QW > max_QW:
                        max_QW, best_membership_c = X_QW, membership_c
            
            # Recover hypergraph partition via NCR / NCM
            X_res, X_part, X_NCM_res, X_NCM_part, X_NCR_res, X_NCR_part = HG_util.WGCD_to_HGCD(
                HG_info, WG_info, best_membership_c, FMHMO_params
            )
            
            # Calculate hypergraph modularity Qhyp
            qhyp = sm.modularityH(H, list(X_part.values()))
            
            # Calculate standard crisp metrics
            RITA = {
                k: [] for k in [
                    'Qws', 'MGs', 'SGs', 'Hcuts', 'nmis', 'ARIs', 'F1s', 'FMIs',
                    'Cs', 'Precisions', 'Accuracys', 'Recalls', 'AMIs'
                ]
            }
            ms.MultiIndexCalculation(dummy_log, RITA, HG_info['H'], real_mem, X_res, list(X_part.values()), True)
            
            run_res = {
                "NMI": RITA['nmis'][-1] if RITA['nmis'] else 0.0,
                "ARI": RITA['ARIs'][-1] if RITA['ARIs'] else 0.0,
                "F1": RITA['F1s'][-1] if RITA['F1s'] else 0.0,
                "FMI": RITA['FMIs'][-1] if RITA['FMIs'] else 0.0,
                "Qhyp": qhyp
            }
            all_runs.append(run_res)
            
            if run_res["Qhyp"] > best_results["Qhyp"]:
                best_results = copy.deepcopy(run_res)
                
        # Calculate means and stds
        mean_res = {k: float(np.mean([x[k] for x in all_runs])) for k in best_results.keys()}
        std_res = {k: float(np.std([x[k] for x in all_runs])) for k in best_results.keys()}
        
        return {
            "best": best_results,
            "mean": mean_res,
            "std": std_res,
            "all_runs": all_runs
        }


def main():
    print("=" * 70)
    print("  PHASE 1 (E0): REPRODUCING FMHMO RESULTS ON REAL DATASETS")
    print("=" * 70)
    
    out_dir = os.path.join(PROJECT_ROOT, "results", "E0_reproduction")
    os.makedirs(out_dir, exist_ok=True)
    
    datasets = ["hospital", "workspace", "high_school"]
    gammas = [0.0, 0.5]  # Paper specifies gamma is either 0 or 0.5 for real networks
    
    full_report = {}
    csv_rows = []
    
    for net in datasets:
        print(f"\n[+] Running FMHMO on '{net}'...")
        best_for_net = None
        best_gamma = 0.0
        
        for g in gammas:
            print(f"    Testing gamma = {g:.1f} (5 independent runs)...")
            res = run_fmhmo_single(net, gamma=g, runs=5, seed=123)
            if best_for_net is None or res["best"]["Qhyp"] > best_for_net["best"]["Qhyp"]:
                best_for_net = res
                best_gamma = g
                
        full_report[net] = {
            "best_gamma": best_gamma,
            "metrics": best_for_net
        }
        
        ref = PAPER_BENCHMARKS[net]
        print(f"\n  Result for '{net}' (best gamma = {best_gamma}):")
        print(f"    Metric   | Reproduced (Mean +/- Std) | Paper Table III/IV | Diff")
        print(f"    ---------+---------------------------+--------------------+-------")
        for m in ["NMI", "ARI", "F1", "FMI", "Qhyp"]:
            m_val = best_for_net["mean"][m]
            s_val = best_for_net["std"][m]
            p_val = ref[m]
            diff = m_val - p_val
            print(f"    {m:<8} | {m_val:.3f} +/- {s_val:.3f}         | {p_val:.3f}              | {diff:+.3f}")
            csv_rows.append({
                "Dataset": net,
                "Best_Gamma": best_gamma,
                "Metric": m,
                "Reproduced_Mean": round(m_val, 4),
                "Reproduced_Std": round(s_val, 4),
                "Paper_Value": p_val,
                "Difference": round(diff, 4)
            })
            
    # Save JSON report
    json_path = os.path.join(out_dir, "reproduction_results.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[OK] Full reproduction results saved to: {json_path}")
    
    # Save CSV summary
    csv_path = os.path.join(out_dir, "reproduction_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[OK] Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
