"""
experiments/run_E4.py
Experiment E4: Control Experiment (p_ov = 0.0).
Verifies that on strictly disjoint (crisp) ground-truth hypergraphs,
the methods do not hallucinate false overlaps.
Computes False Positive Overlap Rate (FOR = % predicted overlapping nodes).
Saves CSV table to results/.
"""

import os
import sys
import csv
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from overlap.generate import generate
from overlap.memberships import extract_memberships_from_raw_hyperedges
from overlap.rules import assign
from overlap.stats import compute_overlap_statistics
from overlap.metrics import onmi, omega, overlap_f1


def run_experiment_E4(n=300, c=5, xi_list=None, seeds=None, out_dir=None):
    if xi_list is None:
        xi_list = [0.1, 0.2, 0.3]
    if seeds is None:
        seeds = [42, 101, 202, 303, 404]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
    os.makedirs(out_dir, exist_ok=True)
    
    rules = [
        ("R0", "argmax", None),
        ("R1", "tau=0.25", 0.25),
        ("R1", "tau=0.35", 0.35),
        ("R2", "rho=0.7", 0.7),
        ("R2", "rho=0.85", 0.85),
        ("R3", "top-1", 1),
        ("R3", "top-2", 2)
    ]
    
    records = []
    
    print("=" * 65)
    print("  EXPERIMENT E4: CONTROL EXPERIMENT (PLANTED OVERLAP p_ov = 0.0)")
    print("=" * 65)
    print("Checking whether methods hallucinate false overlap when ground truth is crisp.\n")
    
    for xi in xi_list:
        print(f"[+] Noise level xi = {xi:.1f} (p_ov = 0.0, {len(seeds)} seeds):")
        
        for rule_id, label, param in rules:
            false_overlap_pcts = []
            onmi_scores = []
            omega_scores = []
            
            for seed in seeds:
                hyperedges, truth = generate(n=n, c=c, p_ov=0.0, xi=xi, seed=seed)
                U, crisp, detected_c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=seed)
                
                cov = assign(U, rule=rule_id, param=param)
                stats = compute_overlap_statistics(cov, n)
                
                false_overlap_pcts.append(stats["percent_overlapping"])
                onmi_scores.append(onmi(cov, truth, n))
                omega_scores.append(omega(cov, truth, n))
                
            mean_for = float(np.mean(false_overlap_pcts))
            std_for = float(np.std(false_overlap_pcts))
            mean_onmi = float(np.mean(onmi_scores))
            mean_omega = float(np.mean(omega_scores))
            
            records.append({
                "xi": xi,
                "Rule": rule_id,
                "Config": label,
                "False_Overlap_Rate_pct": round(mean_for, 2),
                "False_Overlap_Std": round(std_for, 2),
                "ONMI_mean": round(mean_onmi, 4),
                "Omega_mean": round(mean_omega, 4)
            })
            
            print(f"    {rule_id:<3} ({label:<10}): False Overlap Rate = {mean_for:5.2f}% ± {std_for:4.2f}% | ONMI = {mean_onmi:.3f} | Omega = {mean_omega:.3f}")
            
    # Save CSV
    csv_file = os.path.join(out_dir, "E4_control_p_ov_zero.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved E4 control results to: {csv_file}")
    return records


if __name__ == "__main__":
    run_experiment_E4()
