"""
experiments/run_E1.py
Experiment E1: Assignment Rule Comparison vs Planted Overlap Fraction (p_ov).
Compares R0 (argmax), R1 (fixed tau=0.2), R2 (relative rho=0.7), and R3 (top-k=2)
across p_ov in {0.0, 0.1, 0.2, 0.3}.
Saves CSV table and plot to results/.
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from overlap.generate import generate
from overlap.memberships import extract_memberships_from_raw_hyperedges
from overlap.rules import assign
from overlap.metrics import onmi, omega, overlap_f1, overlap_node_prf


def run_experiment_E1(n=200, c=4, xi=0.2, p_ov_list=None, seeds=None, out_dir=None):
    if p_ov_list is None:
        p_ov_list = [0.0, 0.1, 0.2, 0.3]
    if seeds is None:
        seeds = [42, 101, 202]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    rules = [
        ("R0", "argmax", None),
        ("R1", "tau=0.25", 0.25),
        ("R2", "rho=0.7", 0.7),
        ("R3", "top-2", 2)
    ]
    
    records = []
    summary = {r_name: {m: [] for m in ["ONMI", "Omega", "F1", "Node_F1"]} for r_name, _, _ in rules}
    
    print("=" * 65)
    print("  EXPERIMENT E1: RULE COMPARISON VS PLANTED OVERLAP (p_ov)")
    print("=" * 65)
    
    for p_ov in p_ov_list:
        print(f"\n[+] Testing planted overlap p_ov = {p_ov:.1f} ({len(seeds)} seeds)...")
        rule_metrics_p_ov = {r[0]: {m: [] for m in ["ONMI", "Omega", "F1", "Node_F1"]} for r in rules}
        
        for seed in seeds:
            hyperedges, truth = generate(n=n, c=c, p_ov=p_ov, xi=xi, seed=seed)
            U, crisp_part, detected_c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=seed)
            
            for rule_id, rule_label, param in rules:
                pred_cov = assign(U, rule=rule_id, param=param)
                
                score_onmi = onmi(pred_cov, truth, n)
                score_omega = omega(pred_cov, truth, n)
                score_f1 = overlap_f1(pred_cov, truth)
                prf = overlap_node_prf(pred_cov, truth, n)
                score_node_f1 = prf["f1"]
                
                rule_metrics_p_ov[rule_id]["ONMI"].append(score_onmi)
                rule_metrics_p_ov[rule_id]["Omega"].append(score_omega)
                rule_metrics_p_ov[rule_id]["F1"].append(score_f1)
                rule_metrics_p_ov[rule_id]["Node_F1"].append(score_node_f1)
                
        # Aggregate over seeds
        for rule_id, rule_label, _ in rules:
            mean_onmi = float(np.mean(rule_metrics_p_ov[rule_id]["ONMI"]))
            std_onmi = float(np.std(rule_metrics_p_ov[rule_id]["ONMI"]))
            mean_omega = float(np.mean(rule_metrics_p_ov[rule_id]["Omega"]))
            std_omega = float(np.std(rule_metrics_p_ov[rule_id]["Omega"]))
            mean_f1 = float(np.mean(rule_metrics_p_ov[rule_id]["F1"]))
            std_f1 = float(np.std(rule_metrics_p_ov[rule_id]["F1"]))
            mean_node_f1 = float(np.mean(rule_metrics_p_ov[rule_id]["Node_F1"]))
            std_node_f1 = float(np.std(rule_metrics_p_ov[rule_id]["Node_F1"]))
            
            records.append({
                "p_ov": p_ov,
                "Rule": rule_id,
                "Label": rule_label,
                "ONMI_mean": round(mean_onmi, 4),
                "ONMI_std": round(std_onmi, 4),
                "Omega_mean": round(mean_omega, 4),
                "Omega_std": round(std_omega, 4),
                "F1_mean": round(mean_f1, 4),
                "F1_std": round(std_f1, 4),
                "Node_F1_mean": round(mean_node_f1, 4),
                "Node_F1_std": round(std_node_f1, 4)
            })
            
            print(f"    {rule_id:<3} ({rule_label:<8}): ONMI={mean_onmi:.3f}±{std_onmi:.3f} | Omega={mean_omega:.3f}±{std_omega:.3f} | F1={mean_f1:.3f}±{std_f1:.3f} | Node-F1={mean_node_f1:.3f}")
            
    # Save CSV
    csv_file = os.path.join(out_dir, "E1_rule_comparison.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved results to: {csv_file}")
    
    # Plot Figure
    plot_E1_figures(records, p_ov_list, rules, fig_dir)
    return records


def plot_E1_figures(records, p_ov_list, rules, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {"R0": "#1f77b4", "R1": "#ff7f0e", "R2": "#2ca02c", "R3": "#d62728"}
    markers = {"R0": "o", "R1": "s", "R2": "^", "R3": "d"}
    
    # Plot 1: ONMI vs p_ov
    ax = axes[0]
    for rule_id, rule_label, _ in rules:
        rule_recs = [r for r in records if r["Rule"] == rule_id]
        x = [r["p_ov"] for r in rule_recs]
        y = [r["ONMI_mean"] for r in rule_recs]
        err = [r["ONMI_std"] for r in rule_recs]
        ax.errorbar(x, y, yerr=err, label=f"{rule_id} ({rule_label})", color=colors[rule_id],
                     marker=markers[rule_id], capsize=3, linewidth=1.8)
    ax.set_title("Overlapping NMI vs Planted Overlap (p_ov)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Planted Overlap Fraction (p_ov)")
    ax.set_ylabel("ONMI")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    # Plot 2: Omega Index vs p_ov
    ax = axes[1]
    for rule_id, rule_label, _ in rules:
        rule_recs = [r for r in records if r["Rule"] == rule_id]
        x = [r["p_ov"] for r in rule_recs]
        y = [r["Omega_mean"] for r in rule_recs]
        err = [r["Omega_std"] for r in rule_recs]
        ax.errorbar(x, y, yerr=err, label=f"{rule_id} ({rule_label})", color=colors[rule_id],
                     marker=markers[rule_id], capsize=3, linewidth=1.8)
    ax.set_title("Omega Index vs Planted Overlap (p_ov)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Planted Overlap Fraction (p_ov)")
    ax.set_ylabel("Omega Index")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "E1_rules_vs_pov.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure to: {plot_path}")


if __name__ == "__main__":
    run_experiment_E1()
