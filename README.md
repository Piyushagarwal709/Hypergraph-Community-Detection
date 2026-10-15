# FMHMO-Overlap: Overlapping Hypergraph Community Detection from Fuzzy Memberships

**Team:** Piyush Agarwal, Suraj Kumar Tiwari  
**Course:** Social Computing & Analysis (SCA)  
**Base Paper:** Xiao, Ma, Cao, Xu, *"Hypergraph Community Detection With Fuzzy Memberships"*, **IEEE Transactions on Fuzzy Systems**, vol. 33, no. 9, pp. 3255–3268, September 2025.  
**Base Code Repository:** [lisiran15/FMHMO](https://github.com/lisiran15/FMHMO)

---

## 1. Project Overview & The Core Research Gap

Hypergraphs represent higher-order, multi-way interactions where hyperedges connect arbitrary numbers of nodes. The base paper proposes **FMHMO**, an algorithm that:
1. Reduces hyperedges to a weighted graph via a hybrid similarity parameter $\gamma$ (HS-HR, Eq. 10).
2. Performs modularity optimization on the reduced graph using Louvain/Leiden.
3. Quantifies fine-grained node-to-community fuzzy memberships $u(v, C)$ (Def. 3, Eqs. 4–5).

### The Research Gap:
Despite prominently claiming "Fuzzy Memberships" in its title and mathematical formulation, the original paper's final step applies a simple **`argmax`** on $u(v, C)$, collapsing every node into a single crisp community and **discarding the rich multi-community membership distribution**. Furthermore, the authors never validated their fuzzy memberships against overlapping ground-truth benchmarks.

### Our Contribution (FMHMO-Overlap):
1. **Preserve Continuous Memberships ($U \in \mathbb{R}^{n \times c}$):** Extracts the full membership matrix $U$ without modifying the original authors' codebase.
2. **Four Overlap Assignment Rules (R0–R3):**
   - **R0 (Baseline):** Original crisp argmax partition.
   - **R1 (Fixed Threshold):** Node joins every community where $u(v, C) \ge \tau$ (always including argmax).
   - **R2 (Relative Threshold):** Node joins communities with $u(v, C) \ge \rho \cdot u_{\max}(v)$.
   - **R3 (Top-$k$):** Node joins the top-$k$ communities with non-zero membership.
3. **Planted-Overlap Hypergraph Benchmark:** Evaluates overlapping covers on synthetic hypergraphs with controlled planted overlap fraction $p_{\text{ov}}$ and noise $\xi$.
4. **Overlap-Aware Metrics:** Overlapping NMI (LFK / McDaid), Omega Index (overlapping Rand Index), Overlapping F1, and Overlap-node Precision/Recall/F1.
5. **Interactive Demo App:** Streamlit application visualizing communities, highlighting bridge nodes, and plotting individual membership distributions.

---

## 2. Repository Layout

```
project/
├── FMHMO/                               # Authors' original code (DO NOT MODIFY)
│   ├── FMHMO/                           # Main FMHMO implementation & Cython source
│   │   ├── cython_function.pyx          # Cython module (compiled to .so on Linux)
│   │   ├── data/realNetwork/            # 11 real hypergraphs & ground truth files
│   │   └── data/lfrNetwork/             # Synthetic LFR benchmark datasets
│   └── compareAlgorithm/                # Baseline comparison code (HNNCD)
├── overlap/                             # Core FMHMO-Overlap extension package
│   ├── memberships.py                   # Full membership matrix U extraction & Fig. 1 validation
│   ├── rules.py                         # Assignment rules (R0, R1, R2, R3, R4)
│   ├── stats.py                         # Overlap statistics (% overlapping nodes, average memberships)
│   ├── generate.py                      # Synthetic planted-overlap hypergraph generator
│   ├── metrics.py                       # Overlap-aware metrics (ONMI, Omega, F1, Overlap PRF)
│   └── baselines.py                     # Baseline models (Expansion-Overlap adapter)
├── experiments/                         # Reproducible experimental scripts
│   ├── run_E0.py                        # E0: Paper reproduction on real datasets (hospital, workspace, high_school)
│   ├── run_E1.py                        # E1: Rule comparison vs planted overlap (p_ov)
│   ├── run_E2.py                        # E2: Parameter sensitivity sweeps (tau, rho, k)
│   ├── run_E3.py                        # E3: Baseline comparisons
│   ├── run_E4.py                        # E4: Control experiment (false overlap rate at p_ov = 0)
│   └── run_E5.py                        # E5: Real-world hypergraph overlap & bridge-node case studies
├── demo/                                # Interactive Streamlit application
│   └── app.py                           # Web UI for live network visualization and node inspection
├── tests/                               # Comprehensive Pytest test suite
│   ├── test_memberships.py              # Equation 1-2 & Fig. 1 verification, matrix U invariants
│   ├── test_rules.py                    # R0-R4 hand-computed toy tests
│   ├── test_metrics.py                  # ONMI, Omega, F1, PRF sanity tests
│   └── test_generate.py                 # Planted-overlap generator & control tests
├── docs/                                # Technical documentation
│   └── CODEBASE_MAP.md                  # Detailed code map, equations, and line references
├── results/                             # Saved experimental outputs (CSV, JSON)
│   ├── figures/                         # Generated publication-quality plots (PNG)
│   └── E0_reproduction/                 # Reproduction logs comparing with Paper Table III/IV
├── requirements.txt                     # Pinned Python dependencies
├── run_all.py                           # Master end-to-end execution pipeline (--quick and --full)
├── pytest.ini                           # Test configuration
└── README.md                            # This file
```

---

## 3. Installation & Setup

### Requirements:
- Linux / macOS / Windows
- Python 3.9+ (Tested on Python 3.12 and 3.13)
- GCC compiler (for building the Cython extension)

### Quick Setup:
```bash
# 1. Clone repository and navigate to root
cd project

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Build Cython extension for Linux (inplace in FMHMO/FMHMO)
cd FMHMO/FMHMO
python setup.py build_ext --inplace
cd ../..
```

---

## 4. How to Run

### 4.1 Run Unit Tests
```bash
pytest tests/ -v
```
All 13 unit tests validate:
- Exact Fig. 1 equations ($S(v_0, v_1) = 0.667$, $u(v_1, \cdot) = [0.409, 0.273, 0.318]$).
- Invariants of matrix $U$ (all rows sum to 1.0, non-bridge nodes are one-hot).
- Rule properties (R0 crisp, R1 thresholding, R2 relative, R3 top-k).
- Metric invariants (identical covers yield 1.0, misaligned covers yield $\approx 0$).

### 4.2 Run Master Pipeline
```bash
# Fast pipeline (completes in ~1-2 minutes on smaller sweeps)
python run_all.py --quick

# Full sweeps and benchmarks
python run_all.py --full
```

### 4.3 Run Individual Experiments
```bash
python experiments/run_E0.py   # Paper reproduction (hospital, workspace, high_school)
python experiments/run_E1.py   # Rule comparison vs planted overlap
python experiments/run_E2.py   # Parameter sensitivity sweeps
python experiments/run_E3.py   # Baseline comparisons
python experiments/run_E4.py   # Control experiment (p_ov = 0)
python experiments/run_E5.py   # Real-world hypergraph overlap & bridge nodes
```

### 4.4 Launch Interactive Streamlit Demo
```bash
streamlit run demo/app.py
```
Open `http://localhost:8501` in your browser. The app lets you:
- Select synthetic hypergraphs or real datasets (`hospital`).
- Adjust $\tau, \rho, k$ in real-time via sliders.
- View the 2-section graph with overlapping bridge nodes highlighted in red.
- Inspect any node's membership distribution across all communities.

---

## 5. Summary of Experimental Results

### 5.1 E0: Baseline Reproduction vs Paper Tables III & IV
| Dataset | Metric | Reproduced (Mean $\pm$ Std) | Paper Table III / IV | Difference |
|---|---|---|---|---|
| **`workspace`** | NMI | **0.836 $\pm$ 0.000** | **0.836** | **0.000 (Exact match)** |
| `workspace` | ARI | **0.849 $\pm$ 0.000** | **0.849** | **0.000 (Exact match)** |
| `workspace` | F1 | **0.924 $\pm$ 0.000** | **0.924** | **0.000 (Exact match)** |
| `workspace` | FMI | **0.891 $\pm$ 0.000** | **0.891** | **0.000 (Exact match)** |
| `workspace` | $Q_{\text{hyp}}$ | **0.400 $\pm$ 0.000** | **0.394** | +0.006 |
| **`high_school`** | NMI | 0.937 $\pm$ 0.005 | 0.959 | -0.022 |
| `high_school` | $Q_{\text{hyp}}$ | 0.645 $\pm$ 0.001 | 0.623 | +0.022 |
| **`hospital`** | NMI | 0.119 $\pm$ 0.022 | 0.145 | -0.026 |
| `hospital` | $Q_{\text{hyp}}$ | 0.089 $\pm$ 0.006 | 0.080 | +0.009 |

### 5.2 E1: Rule Comparison on Planted Overlap
On synthetic benchmarks with planted overlap ($p_{\text{ov}} = 0.20$):
- **Crisp R0 (Original FMHMO):** Completely unable to detect overlapping nodes (Overlap-Node F1 = **0.000**).
- **FMHMO-Overlap R1 ($\tau=0.25$):** Achieves Overlap-Node F1 = **0.765**, ONMI = **0.786**, and Omega Index = **0.828**.
- **FMHMO-Overlap R2 ($\rho=0.70$):** High precision overlap recovery with ONMI = **0.721** and Omega = **0.735**.

### 5.3 E4: Control Experiment ($p_{\text{ov}} = 0.0$)
When tested on strictly crisp benchmarks ($p_{\text{ov}} = 0.0$):
- Both R1 ($\tau \ge 0.25$) and R2 ($\rho \ge 0.70$) produce a False Overlap Rate of **0.00%**, proving that FMHMO-Overlap **does not hallucinate artificial overlaps** when true communities are disjoint.

### 5.4 E5: Real-World Overlapping Structure
- **`hospital` ($n=75, m=1825$):** Under R1 ($\tau=0.25$), **36.0%** of nodes exhibit multi-community membership. Node 74 is a prominent bridge node with $u(74, C_5) = 0.542$ and $u(74, C_2) = 0.410$.
- **`workspace` ($n=92, m=788$):** Under R1 ($\tau=0.25$), **10.9%** of nodes overlap. Node 24 is a bridge node connecting community 2 ($u=0.541$) and community 1 ($u=0.395$).

---

## 6. Project Limitations & Ethical Transparency

In accordance with good academic practices, we explicitly document the limitations of this method:
1. **Single-hyperedge nodes cannot overlap:** Nodes participating in only one hyperedge have one-hot memberships and cannot exhibit overlap.
2. **Post-partition calculation:** Memberships are computed after an initial crisp hyperedge partition. If all hyperedges of an overlapping node fall into the same community cluster, the overlap remains undetectable.
3. **Synthetic-only overlap validation:** Real-world benchmark ground-truth labels are crisp (one label per node). We therefore validate overlap accuracy exclusively on planted-overlap synthetic hypergraphs, using real networks only for structural statistics and qualitative case studies.
4. **Heuristic membership scores:** Fuzzy memberships are normalized similarity-based heuristics, not calibrated posterior probabilities.
