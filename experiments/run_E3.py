"""
experiments/run_E3.py
Experiment E3: Baseline Comparison.
Compares FMHMO-Overlap (R1 and R2) against:
1. Original crisp FMHMO (R0)
2. Clique-Expansion Soft Baseline (Expansion-Overlap)
Evaluates across p_ov in {0.1, 0.2, 0.3}.
Saves CSV table and comparison bar charts to results/.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from overlap.generate import generate
from overlap.memberships import extract_memberships_from_raw_hyperedges
from overlap.rules import assign
from overlap.baselines import expansion_overlap_baseline
from overlap.metrics import onmi, omega, overlap_f1, overlap_node_prf


def run_experiment_E3(n=200, c=4, xi=0.2, p_ov_list=None, seeds=None, out_dir=None):
    if p_ov_list is None:
        p_ov_list = [0.1, 0.2, 0.3]
    if seeds is None:
        seeds = [42, 101, 202]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    methods = [
        "FMHMO (Crisp R0)",
        "FMHMO-Overlap (R1 tau=0.25)",
        "FMHMO-Overlap (R2 rho=0.7)",
        "Expansion-Overlap Baseline"
    ]
    
    records = []
    
    print("=" * 65)
    print("  EXPERIMENT E3: BASELINE COMPARISON ON PLANTED OVERLAP")
    print("=" * 65)
    
    for p_ov in p_ov_list:
        print(f"\n[+] Testing p_ov = {p_ov:.1f} across {len(seeds)} seeds...")
        metrics_by_method = {m: {"onmi": [], "omega": [], "f1": [], "node_f1": []} for m in methods}
        
        for seed in seeds:
            hyperedges, truth = generate(n=n, c=c, p_ov=p_ov, xi=xi, seed=seed)
            U, crisp, detected_c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=seed)
            
            # 1. FMHMO (Crisp R0)
            cov_r0 = assign(U, rule="R0")
            # 2. FMHMO-Overlap R1
            cov_r1 = assign(U, rule="R1", param=0.25)
            # 3. FMHMO-Overlap R2
            cov_r2 = assign(U, rule="R2", param=0.7)
            # 4. Expansion baseline
            cov_exp = expansion_overlap_baseline(hyperedges, n=n, c=c, tau=0.25, seed=seed)
            
            candidates = [
                ("FMHMO (Crisp R0)", cov_r0),
                ("FMHMO-Overlap (R1 tau=0.25)", cov_r1),
                ("FMHMO-Overlap (R2 rho=0.7)", cov_r2),
                ("Expansion-Overlap Baseline", cov_exp)
            ]
            
            for m_name, cov in candidates:
                metrics_by_method[m_name]["onmi"].append(onmi(cov, truth, n))
                metrics_by_method[m_name]["omega"].append(omega(cov, truth, n))
                metrics_by_method[m_name]["f1"].append(overlap_f1(cov, truth))
                prf = overlap_node_prf(cov, truth, n)
                metrics_by_method[m_name]["node_f1"].append(prf["f1"])
                
        for m_name in methods:
            rec = {
                "p_ov": p_ov,
                "Method": m_name,
                "ONMI_mean": round(float(np.mean(metrics_by_method[m_name]["onmi"])), 4),
                "ONMI_std": round(float(np.std(metrics_by_method[m_name]["onmi"])), 4),
                "Omega_mean": round(float(np.mean(metrics_by_method[m_name]["omega"])), 4),
                "Omega_std": round(float(np.std(metrics_by_method[m_name]["omega"])), 4),
                "F1_mean": round(float(np.mean(metrics_by_method[m_name]["f1"])), 4),
                "F1_std": round(float(np.std(metrics_by_method[m_name]["f1"])), 4),
                "Node_F1_mean": round(float(np.mean(metrics_by_method[m_name]["node_f1"])), 4),
                "Node_F1_std": round(float(np.std(metrics_by_method[m_name]["node_f1"])), 4)
            }
            records.append(rec)
            print(f"    {m_name:<28}: ONMI={rec['ONMI_mean']:.3f} | Omega={rec['Omega_mean']:.3f} | F1={rec['F1_mean']:.3f} | Node-F1={rec['Node_F1_mean']:.3f}")
            
    # Save CSV
    csv_file = os.path.join(out_dir, "E3_baseline_comparison.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved baseline comparison CSV to: {csv_file}")
    
    # Plot bar comparison
    plot_E3_figures(records, p_ov_list, methods, fig_dir)
    return records


def plot_E3_figures(records, p_ov_list, methods, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    x = np.arange(len(p_ov_list))
    width = 0.20
    
    # Plot ONMI
    ax = axes[0]
    for idx, m_name in enumerate(methods):
        m_recs = [r for r in records if r["Method"] == m_name]
        y = [r["ONMI_mean"] for r in m_recs]
        ax.bar(x + (idx - 1.5) * width, y, width, label=m_name)
    ax.set_title("ONMI Comparison vs Baseline", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p_ov={p}" for p in p_ov_list])
    ax.set_ylabel("ONMI")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    
    # Plot Omega
    ax = axes[1]
    for idx, m_name in enumerate(methods):
        m_recs = [r for r in records if r["Method"] == m_name]
        y = [r["Omega_mean"] for r in m_recs]
        ax.bar(x + (idx - 1.5) * width, y, width, label=m_name)
    ax.set_title("Omega Index Comparison vs Baseline", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p_ov={p}" for p in p_ov_list])
    ax.set_ylabel("Omega Index")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "E3_baseline_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure to: {plot_path}")


if __name__ == "__main__":
    run_experiment_E3()
