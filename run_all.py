"""
run_all.py
Master runner script for the FMHMO-Overlap project.
Executes the full pipeline end-to-end:
  - E0: Reproduction on real datasets
  - E1: Assignment rule comparison vs planted overlap
  - E2: Parameter sensitivity sweeps
  - E3: Baseline comparisons
  - E4: Control experiment (false-overlap verification at p_ov=0)
  - E5: Real-world hypergraph overlap analysis
  - Pytest unit test suite

Usage:
  python run_all.py --quick   (runs in 1-2 minutes on small settings)
  python run_all.py --full    (runs full sweeps and benchmarks)
"""

import os
import sys
import argparse
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_unit_tests():
    print("\n" + "=" * 70)
    print("  RUNNING PYTEST UNIT TEST SUITE")
    print("=" * 70)
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print("[!] Warning: Some tests failed!")
        return False
    print("[OK] All unit tests passed successfully!")
    return True


def execute_pipeline(quick=True):
    start_time = time.time()
    mode_str = "QUICK MODE" if quick else "FULL MODE"
    print("\n" + "#" * 70)
    print(f"  STARTING FMHMO-OVERLAP PIPELINE ({mode_str})")
    print("#" * 70)
    
    out_dir = os.path.join(PROJECT_ROOT, "results")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Run Unit Tests First
    tests_ok = run_unit_tests()
    
    # 2. Experiment E0: Reproduction
    from experiments.run_E0 import run_fmhmo_single, PAPER_BENCHMARKS
    print("\n" + "=" * 70)
    print("  [Step 1/6] Running E0: Reproduction on Real Datasets")
    print("=" * 70)
    e0_datasets = ["hospital", "workspace"] if quick else ["hospital", "workspace", "high_school"]
    runs_count = 2 if quick else 5
    for net in e0_datasets:
        print(f"  -> Testing '{net}'...")
        res = run_fmhmo_single(net, gamma=0.0, runs=runs_count, seed=42)
        ref = PAPER_BENCHMARKS.get(net, {})
        print(f"     NMI: {res['mean']['NMI']:.3f} (Paper: {ref.get('NMI', 'N/A')}) | Qhyp: {res['mean']['Qhyp']:.3f} (Paper: {ref.get('Qhyp', 'N/A')})")
        
    # 3. Experiment E1: Rule Comparison vs p_ov
    from experiments.run_E1 import run_experiment_E1
    print("\n" + "=" * 70)
    print("  [Step 2/6] Running E1: Assignment Rule Comparison vs p_ov")
    print("=" * 70)
    if quick:
        run_experiment_E1(n=100, c=3, xi=0.2, p_ov_list=[0.0, 0.2], seeds=[42], out_dir=out_dir)
    else:
        run_experiment_E1(n=250, c=4, xi=0.2, p_ov_list=[0.0, 0.1, 0.2, 0.3], seeds=[42, 101, 202], out_dir=out_dir)
        
    # 4. Experiment E2: Parameter Sensitivity
    from experiments.run_E2 import run_experiment_E2
    print("\n" + "=" * 70)
    print("  [Step 3/6] Running E2: Parameter Sensitivity Sweeps")
    print("=" * 70)
    if quick:
        run_experiment_E2(n=100, c=3, p_ov=0.2, xi=0.2, seeds=[42], out_dir=out_dir)
    else:
        run_experiment_E2(n=200, c=4, p_ov=0.2, xi=0.2, seeds=[42, 101, 202], out_dir=out_dir)
        
    # 5. Experiment E3: Baseline Comparison
    from experiments.run_E3 import run_experiment_E3
    print("\n" + "=" * 70)
    print("  [Step 4/6] Running E3: Baseline Comparisons")
    print("=" * 70)
    if quick:
        run_experiment_E3(n=100, c=3, xi=0.2, p_ov_list=[0.1, 0.2], seeds=[42], out_dir=out_dir)
    else:
        run_experiment_E3(n=200, c=4, xi=0.2, p_ov_list=[0.1, 0.2, 0.3], seeds=[42, 101, 202], out_dir=out_dir)
        
    # 6. Experiment E4: Control Experiment (p_ov = 0)
    from experiments.run_E4 import run_experiment_E4
    print("\n" + "=" * 70)
    print("  [Step 5/6] Running E4: Control Experiment (p_ov = 0.0)")
    print("=" * 70)
    if quick:
        run_experiment_E4(n=100, c=3, xi_list=[0.1, 0.2], seeds=[42], out_dir=out_dir)
    else:
        run_experiment_E4(n=250, c=4, xi_list=[0.1, 0.2, 0.3], seeds=[42, 101, 202], out_dir=out_dir)
        
    # 7. Experiment E5: Real-World Overlapping Analysis
    from experiments.run_E5 import run_experiment_E5
    print("\n" + "=" * 70)
    print("  [Step 6/6] Running E5: Real-World Overlapping Analysis")
    print("=" * 70)
    e5_datasets = ["hospital", "workspace"] if quick else ["hospital", "workspace", "senate_committees", "primary_school"]
    run_experiment_E5(datasets=e5_datasets, out_dir=out_dir)
    
    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"  PIPELINE COMPLETE ({mode_str}) in {elapsed:.1f}s")
    print(f"  All results written to: {out_dir}")
    print(f"  Figures written to: {fig_dir}")
    print("#" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="FMHMO-Overlap master execution pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Run fast pipeline on small settings (~1-2 min)")
    group.add_argument("--full", action="store_true", help="Run full sweeps and benchmarks")
    
    args = parser.parse_args()
    
    # Default to quick if neither specified
    is_quick = not args.full
    execute_pipeline(quick=is_quick)


if __name__ == "__main__":
    main()
