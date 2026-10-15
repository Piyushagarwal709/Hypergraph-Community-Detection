"""
experiments/run_E2.py
Experiment E2: Parameter Sensitivity Curves.
Analyzes how performance varies with:
- Fixed threshold tau in {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40} for Rule R1.
- Relative threshold rho in {0.5, 0.6, 0.7, 0.8, 0.9} for Rule R2.
- Top-k parameter in {1, 2, 3} for Rule R3.
Saves results CSV and sensitivity curves to results/.
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
from overlap.metrics import onmi, omega, overlap_f1
from overlap.stats import compute_overlap_statistics


def run_experiment_E2(n=200, c=4, p_ov=0.2, xi=0.2, seeds=None, out_dir=None):
    if seeds is None:
        seeds = [42, 101, 202]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    tau_list = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    rho_list = [0.50, 0.60, 0.70, 0.80, 0.90]
    k_list = [1, 2, 3]
    
    records = []
    
    print("=" * 65)
    print("  EXPERIMENT E2: PARAMETER SENSITIVITY SWEEPS (R1, R2, R3)")
    print("=" * 65)
    
    # Pre-generate datasets and membership matrices for each seed
    datasets = []
    for seed in seeds:
        hyperedges, truth = generate(n=n, c=c, p_ov=p_ov, xi=xi, seed=seed)
        U, crisp, detected_c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=seed)
        datasets.append((truth, U))
        
    # 1. Sweep R1 tau
    print("\n[+] Sweeping R1 tau in [0.10 .. 0.40]...")
    for tau in tau_list:
        scores = {"onmi": [], "omega": [], "f1": [], "pct_ov": []}
        for truth, U in datasets:
            cov = assign(U, rule="R1", param=tau)
            scores["onmi"].append(onmi(cov, truth, n))
            scores["omega"].append(omega(cov, truth, n))
            scores["f1"].append(overlap_f1(cov, truth))
            stats = compute_overlap_statistics(cov, n)
            scores["pct_ov"].append(stats["percent_overlapping"])
            
        rec = {
            "Rule": "R1",
            "Parameter_Name": "tau",
            "Parameter_Value": tau,
            "ONMI_mean": round(float(np.mean(scores["onmi"])), 4),
            "ONMI_std": round(float(np.std(scores["onmi"])), 4),
            "Omega_mean": round(float(np.mean(scores["omega"])), 4),
            "Omega_std": round(float(np.std(scores["omega"])), 4),
            "F1_mean": round(float(np.mean(scores["f1"])), 4),
            "F1_std": round(float(np.std(scores["f1"])), 4),
            "Percent_Overlap_mean": round(float(np.mean(scores["pct_ov"])), 2)
        }
        records.append(rec)
        print(f"    tau={tau:.2f} -> ONMI={rec['ONMI_mean']:.3f} | Omega={rec['Omega_mean']:.3f} | %Overlap={rec['Percent_Overlap_mean']:.1f}%")

    # 2. Sweep R2 rho
    print("\n[+] Sweeping R2 rho in [0.50 .. 0.90]...")
    for rho in rho_list:
        scores = {"onmi": [], "omega": [], "f1": [], "pct_ov": []}
        for truth, U in datasets:
            cov = assign(U, rule="R2", param=rho)
            scores["onmi"].append(onmi(cov, truth, n))
            scores["omega"].append(omega(cov, truth, n))
            scores["f1"].append(overlap_f1(cov, truth))
            stats = compute_overlap_statistics(cov, n)
            scores["pct_ov"].append(stats["percent_overlapping"])
            
        rec = {
            "Rule": "R2",
            "Parameter_Name": "rho",
            "Parameter_Value": rho,
            "ONMI_mean": round(float(np.mean(scores["onmi"])), 4),
            "ONMI_std": round(float(np.std(scores["onmi"])), 4),
            "Omega_mean": round(float(np.mean(scores["omega"])), 4),
            "Omega_std": round(float(np.std(scores["omega"])), 4),
            "F1_mean": round(float(np.mean(scores["f1"])), 4),
            "F1_std": round(float(np.std(scores["f1"])), 4),
            "Percent_Overlap_mean": round(float(np.mean(scores["pct_ov"])), 2)
        }
        records.append(rec)
        print(f"    rho={rho:.2f} -> ONMI={rec['ONMI_mean']:.3f} | Omega={rec['Omega_mean']:.3f} | %Overlap={rec['Percent_Overlap_mean']:.1f}%")

    # 3. Sweep R3 top-k
    print("\n[+] Sweeping R3 top-k in {1, 2, 3}...")
    for k in k_list:
        scores = {"onmi": [], "omega": [], "f1": [], "pct_ov": []}
        for truth, U in datasets:
            cov = assign(U, rule="R3", param=k)
            scores["onmi"].append(onmi(cov, truth, n))
            scores["omega"].append(omega(cov, truth, n))
            scores["f1"].append(overlap_f1(cov, truth))
            stats = compute_overlap_statistics(cov, n)
            scores["pct_ov"].append(stats["percent_overlapping"])
            
        rec = {
            "Rule": "R3",
            "Parameter_Name": "k",
            "Parameter_Value": k,
            "ONMI_mean": round(float(np.mean(scores["onmi"])), 4),
            "ONMI_std": round(float(np.std(scores["onmi"])), 4),
            "Omega_mean": round(float(np.mean(scores["omega"])), 4),
            "Omega_std": round(float(np.std(scores["omega"])), 4),
            "F1_mean": round(float(np.mean(scores["f1"])), 4),
            "F1_std": round(float(np.std(scores["f1"])), 4),
            "Percent_Overlap_mean": round(float(np.mean(scores["pct_ov"])), 2)
        }
        records.append(rec)
        print(f"    k={k}    -> ONMI={rec['ONMI_mean']:.3f} | Omega={rec['Omega_mean']:.3f} | %Overlap={rec['Percent_Overlap_mean']:.1f}%")

    # Save CSV
    csv_file = os.path.join(out_dir, "E2_parameter_sensitivity.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved sensitivity CSV to: {csv_file}")
    
    # Plot sensitivity curves
    plot_E2_figures(records, fig_dir)
    return records


def plot_E2_figures(records, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. R1 sensitivity
    ax = axes[0]
    r1_recs = [r for r in records if r["Rule"] == "R1"]
    x = [r["Parameter_Value"] for r in r1_recs]
    ax.plot(x, [r["ONMI_mean"] for r in r1_recs], marker="o", label="ONMI", color="#1f77b4")
    ax.plot(x, [r["Omega_mean"] for r in r1_recs], marker="s", label="Omega", color="#2ca02c")
    ax.set_title("Rule R1 Sensitivity (vs tau)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Metric Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # 2. R2 sensitivity
    ax = axes[1]
    r2_recs = [r for r in records if r["Rule"] == "R2"]
    x = [r["Parameter_Value"] for r in r2_recs]
    ax.plot(x, [r["ONMI_mean"] for r in r2_recs], marker="o", label="ONMI", color="#1f77b4")
    ax.plot(x, [r["Omega_mean"] for r in r2_recs], marker="s", label="Omega", color="#2ca02c")
    ax.set_title("Rule R2 Sensitivity (vs rho)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Threshold rho")
    ax.set_ylabel("Metric Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    # 3. R3 sensitivity
    ax = axes[2]
    r3_recs = [r for r in records if r["Rule"] == "R3"]
    x = [r["Parameter_Value"] for r in r3_recs]
    ax.bar([str(val) for val in x], [r["ONMI_mean"] for r in r3_recs], color="#ff7f0e", alpha=0.8, label="ONMI")
    ax.set_title("Rule R3 Sensitivity (vs k)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("ONMI")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(fig_dir, "E2_parameter_sensitivity.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Saved figure to: {plot_path}")


if __name__ == "__main__":
    run_experiment_E2()
