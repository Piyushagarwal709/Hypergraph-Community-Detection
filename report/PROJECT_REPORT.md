# Research Project Report: FMHMO-Overlap
## Overlapping Community Detection from Hypergraph Fuzzy Memberships

**Authors:**  
- **Piyush Agarwal** (Methodology, Matrix Extraction, Assignment Rules, Reproduction & Demo)  
- **Suraj Kumar Tiwari** (Evaluation Framework, Synthetic Generator, Metrics, Baseline Comparisons)  

**Institution:** Bennett University / Course: Social Computing & Analysis (SCA)  
**Base Research Paper:** Xiao, Ma, Cao, Xu, *"Hypergraph Community Detection With Fuzzy Memberships"*, **IEEE Transactions on Fuzzy Systems**, vol. 33, no. 9, pp. 3255–3268, September 2025.  
**Repository & Source Code:** `https://github.com/lisiran15/FMHMO` (Original) | `FMHMO-Overlap` (Our Extension)

---

## Abstract

Hypergraphs model complex, higher-order group interactions where each hyperedge can connect an arbitrary number of nodes. In a recent 2025 publication in *IEEE Transactions on Fuzzy Systems*, Xiao et al. proposed **FMHMO**, a hypergraph community detection algorithm that calculates fine-grained node-to-community fuzzy membership grades $u(v, C)$ to assist partition recovery. However, despite the foundational claim of computing "fuzzy memberships", the authors' final step applies a rigid `argmax` operator, discarding all multi-community distribution data and producing solely crisp, disjoint partitions. Furthermore, the paper never validates these fuzzy memberships against overlapping ground-truth benchmarks. 

In this project, we develop **FMHMO-Overlap**, an end-to-end framework that exposes the complete $n \times c$ node-to-community fuzzy membership matrix $U$ and converts it into overlapping community covers via four principled assignment rules (R0 argmax baseline, R1 fixed threshold $\tau$, R2 relative threshold $\rho$, and R3 top-$k$). We design a planted-overlap synthetic hypergraph benchmark and evaluate using overlap-aware metrics: Overlapping NMI (LFK), Omega Index, Overlapping F1, and Overlap-node Precision/Recall/F1. Empirical reproduction on the original paper's datasets replicates reported crisp accuracy exactly on `workspace` (NMI 0.836, ARI 0.849, F1 0.924). On planted-overlap benchmarks ($p_{\text{ov}} = 0.20$), our framework discovers multi-community nodes with an Overlap-Node F1 of **0.765** and Omega Index of **0.828**, whereas the original paper's crisp output achieves **0.000**. On crisp controls ($p_{\text{ov}} = 0.0$), our thresholding rules maintain a False Overlap Rate of **0.00%**, proving absence of artificial hallucination. Finally, on real networks (`hospital` and `workspace`), we identify and visualize concrete bridge nodes connecting functional communities.

---

## 1. Introduction & The Core Research Gap

Hypergraph community detection is pivotal in modeling systems where relationships are inherently multi-way—such as scientific co-authorships, cellular protein complexes, and contact networks. Standard graph projections (like clique expansions) frequently induce severe structural distortion and information loss.

To address this, Xiao et al. (IEEE TFS 2025) proposed the **FMHMO** framework, which operates in three phases:
1. **HS-HR (Hyperedge Similarity-based Reduction):** Projects hyperedges onto a weighted graph where edge weights $w_{ij}$ combine hyperedge topology and average node similarity parameterized by $\gamma$ (Eq. 10).
2. **Modularity Optimization:** Partitions the reduced graph using Louvain/multilevel modularity maximization.
3. **FM-HPR (Partition Recovery):** Maps the partition back to individual nodes. Nodes in multi-community hyperedges (bridge nodes) are resolved by calculating node-to-community fuzzy memberships $u(v, C)$ via Def. 3 (Eqs. 4–5).

### The Three Critical Research Gaps:
1. **L1 (Discarded Multi-Membership Information):** Although the paper defines and computes continuous membership values $u(v, C) \in [0, 1]$, it immediately takes an `argmax`:
   $$X[v] = \arg\max_{C} u(v, C)$$
   All non-maximal membership values are permanently discarded. Thus, despite the title and theoretical derivation, the algorithm outputs a purely crisp partition without overlap.
2. **L2 (Lack of Overlap Ground-Truth Validation):** The authors only evaluated FMHMO on real datasets with single-label (crisp) ground truth and synthetic benchmarks without planted overlap. The validity of the membership grades for true multi-community belonging was never tested.
3. **L3 (Metric Incompatibility):** The paper relied exclusively on crisp classification metrics (standard NMI, ARI, F1, FMI) which are mathematically unable to assess multi-membership accuracy.

---

## 2. Proposed Method: FMHMO-Overlap

Our method preserves the original FMHMO reduction and recovery stages as a sealed, unmodified engine, wrapping it to expose the full continuous membership distribution and mapping it to overlapping communities.

### 2.1 Extraction of the Continuous Membership Matrix $U$
For a hypergraph $G = (V, E)$ with $n = |V|$ nodes and $c$ detected communities, we extract matrix $U \in \mathbb{R}^{n \times c}$ such that:
- For intra-community (non-bridge) nodes, all incident hyperedges reside within a single community $C^*$. Their row is one-hot: $U[v, C^*] = 1.0$ and $U[v, C] = 0.0$ for $C \ne C^*$.
- For bridge nodes, $U[v, :]$ stores the normalized membership grades across all neighbor communities $C$:
  $$u(v, C) = \frac{\text{attr}(v, C)}{\sum_{C' \in \Gamma_C(v)} \text{attr}(v, C')}$$
  where $\text{attr}(v, C) = \sum_{e \in \Gamma_e(v)} u(v, e) \cdot \frac{|e \cap C|}{|e|}$.
- Every row $U[v, :]$ strictly satisfies $\sum_{j=0}^{c-1} U[v, j] = 1.0$ and $U[v, j] \ge 0$.

### 2.2 Overlap Assignment Rules
To convert $U$ into overlapping community sets $\mathcal{C} = \{C_0, C_1, \dots, C_{c-1}\}$, we implement four modular rules:
1. **R0 (Baseline Argmax):**
   $$v \in C_j \iff j = \arg\max_k U[v, k]$$
   Replicates the original paper's crisp partition.
2. **R1 (Fixed Threshold $\tau$):**
   $$v \in C_j \iff U[v, j] \ge \tau \quad \cup \quad \{\arg\max_k U[v, k]\}$$
   Includes any community with non-trivial membership; always retains argmax to prevent orphaned nodes.
3. **R2 (Relative Threshold $\rho$):**
   $$v \in C_j \iff U[v, j] \ge \rho \cdot \max_k U[v, k]$$
   Adapts thresholding to the node's maximum confidence level.
4. **R3 (Top-$k$):**
   Assigns node $v$ to its top $k$ communities satisfying $U[v, j] > 0$.
5. **R4 (Data-Driven Modularity Optimization):**
   Sweeps $\tau$ and selects the partition maximizing an overlapping hypergraph modularity score without requiring ground truth.

---

## 3. Evaluation Methodology

### 3.1 Planted-Overlap Synthetic Hypergraph Generator
To address Gap L2, we built a parameterized synthetic hypergraph generator:
- $n$ nodes partitioned into $c$ base communities.
- A controlled fraction $p_{\text{ov}} \in [0.0, 0.4]$ of nodes is planted into 2 or 3 overlapping communities.
- When $p_{\text{ov}} = 0.0$, the benchmark represents a strictly disjoint partition (used as control).
- Hyperedges of cardinality $d \in [2, 6]$ are generated: with probability $1 - \xi$, nodes are sampled from within a common community; with probability $\xi$, nodes are sampled uniformly at random across all nodes.

### 3.2 Overlap-Aware Evaluation Metrics
To address Gap L3, we evaluate partitions using:
1. **Overlapping NMI (ONMI):** The LFK / McDaid conditional entropy formulation for overlapping covers.
2. **Omega Index ($\Omega$):** The overlapping equivalent of the Adjusted Rand Index, measuring pairwise co-membership agreement against random chance:
   $$\Omega = \frac{P_{\text{obs}} - P_{\text{exp}}}{1 - P_{\text{exp}}}$$
3. **Overlapping F1:** Best-match harmonic mean averaged symmetrically in both directions.
4. **Overlap-Node Precision, Recall, and F1:** Evaluates the binary classification accuracy of identifying which specific nodes belong to multiple communities.

---

## 4. Experimental Results & Verification

All results reported below are generated from direct execution scripts in `experiments/` and logged in `results/`.

### 4.1 Experiment E0: Paper Reproduction
We validated our execution environment against the original paper's reported values (Table III and Table IV in Xiao et al., IEEE TFS 2025):

| Dataset | Nodes $n$ | Hyperedges $m$ | Metric | Reproduced (Ours) | Paper Table III / IV | Difference | Status |
|---|---|---|---|---|---|---|---|
| **`workspace`** | 92 | 788 | **NMI** | **0.836 $\pm$ 0.000** | **0.836** | **0.000** | **Exact Match** |
| `workspace` | 92 | 788 | **ARI** | **0.849 $\pm$ 0.000** | **0.849** | **0.000** | **Exact Match** |
| `workspace` | 92 | 788 | **F1** | **0.924 $\pm$ 0.000** | **0.924** | **0.000** | **Exact Match** |
| `workspace` | 92 | 788 | **FMI** | **0.891 $\pm$ 0.000** | **0.891** | **0.000** | **Exact Match** |
| `workspace` | 92 | 788 | **$Q_{\text{hyp}}$** | **0.400 $\pm$ 0.000** | **0.394** | +0.006 | Consistent |
| **`high_school`** | 327 | 7,818 | **NMI** | 0.937 $\pm$ 0.005 | 0.959 | -0.022 | Consistent |
| `high_school` | 327 | 7,818 | **$Q_{\text{hyp}}$** | 0.645 $\pm$ 0.001 | 0.623 | +0.022 | Consistent |
| **`hospital`** | 75 | 1,825 | **NMI** | 0.119 $\pm$ 0.022 | 0.145 | -0.026 | Consistent |
| `hospital` | 75 | 1,825 | **$Q_{\text{hyp}}$** | 0.089 $\pm$ 0.006 | 0.080 | +0.009 | Consistent |

The exact match on `workspace` confirms the correctness of our wrapper and reduction pipeline.

### 4.2 Experiment E1: Assignment Rule Comparison
Evaluating rules on synthetic hypergraphs ($n=200, c=4, \xi=0.20$) across planted overlap fractions:

| Planted Overlap $p_{\text{ov}}$ | Rule | ONMI | Omega Index | Overlapping F1 | Overlap-Node F1 |
|---|---|---|---|---|---|
| **$p_{\text{ov}} = 0.0$ (Crisp)** | **R0 (Argmax)** | 1.000 | 1.000 | 1.000 | 1.000 |
| | **R1 ($\tau=0.25$)** | 1.000 | 1.000 | 1.000 | 1.000 |
| | **R2 ($\rho=0.70$)** | 1.000 | 1.000 | 1.000 | 1.000 |
| **$p_{\text{ov}} = 0.10$** | R0 (Argmax) | 0.779 | 0.816 | 0.939 | 0.000 |
| | **R1 ($\tau=0.25$)** | **0.875** | **0.912** | **0.973** | **0.824** |
| | R2 ($\rho=0.70$) | 0.829 | 0.871 | 0.959 | 0.571 |
| **$p_{\text{ov}} = 0.20$** | R0 (Argmax) | 0.657 | 0.664 | 0.888 | 0.000 |
| | **R1 ($\tau=0.25$)** | **0.786** | **0.828** | **0.950** | **0.765** |
| | R2 ($\rho=0.70$) | 0.721 | 0.735 | 0.917 | 0.400 |

**Finding:** The original paper's crisp partition (R0) achieves an Overlap-Node F1 of **0.000** because it cannot represent multi-membership. In contrast, R1 ($\tau=0.25$) improves ONMI by **+12.9%** and captures overlapping nodes with **0.765 F1**.

### 4.3 Experiment E4: Control Experiment ($p_{\text{ov}} = 0.0$)
A critical risk of thresholding is hallucinating artificial overlaps on networks that are genuinely crisp. We tested our rules across multiple noise levels $\xi \in \{0.1, 0.2\}$:
- **R0 (Argmax):** False Overlap Rate = **0.00%**
- **R1 ($\tau=0.25$):** False Overlap Rate = **0.00%**
- **R1 ($\tau=0.35$):** False Overlap Rate = **0.00%**
- **R2 ($\rho=0.70$):** False Overlap Rate = **0.00%**
- **R2 ($\rho=0.85$):** False Overlap Rate = **0.00%**
- **R3 ($k=2$):** False Overlap Rate = 52.0% – 68.0%

**Finding:** Rules R1 and R2 exhibit zero false positives on crisp structures because non-bridge nodes naturally have one-hot distributions ($u = 1.0$). In contrast, top-$k$ (R3 with $k=2$) artificially forces multi-membership even when uncalled for. Therefore, **R1 and R2 are the recommended production rules**.

### 4.4 Experiment E5: Real-World Overlap & Case Studies
Applying FMHMO-Overlap to real-world networks reveals significant multi-community bridge topology:
- **`hospital` ($n=75, m=1825$):** Under R1 ($\tau=0.25$), **36.0%** of nodes overlap (average memberships: 1.37).
  - *Case Study (Node 74):* Degree = 11 hyperedges. Membership distribution: **$C_5: 0.542$**, **$C_2: 0.410$**, $C_3: 0.046$. Node 74 serves as an active inter-departmental bridge between medical teams $C_5$ and $C_2$.
- **`workspace` ($n=92, m=788$):** Under R1 ($\tau=0.25$), **10.87%** of nodes overlap (average memberships: 1.11).
  - *Case Study (Node 24):* Degree = 9 hyperedges. Membership distribution: **$C_2: 0.541$**, **$C_1: 0.395$**, $C_3: 0.064$. Represents an individual collaborating across two organizational project groups.

---

## 5. Honest Academic Limitations

In accordance with strict academic integrity, we document the fundamental limitations of our framework:

1. **Theoretical Inability to Detect Single-Hyperedge Overlap:** A node that participates in only one hyperedge has its entire membership mass concentrated on that single hyperedge ($u(v, e) = 1.0$). Consequently, its community membership row is strictly one-hot. Even if such a node represents a planted multi-community entity in ground truth, our method cannot detect its overlap.
2. **Post-Partition Dependency:** Memberships are calculated *after* the reduced weighted graph has been partitioned into crisp hyperedge clusters. If all incident hyperedges of an overlapping node happen to be placed into the same community cluster, the overlap will be masked completely.
3. **Synthetic-Only Overlap Validation:** All standard real-world hypergraph benchmarks (e.g. `hospital`, `senate_committees`, `cora_ca`) possess single-label crisp ground truth. Overlap accuracy can therefore only be validated quantitatively on synthetic benchmarks. Real-world results must be treated as structural and qualitative analyses, not claims of ground-truth overlap precision.
4. **Heuristic Nature of Membership Scores:** The membership values $u(v, C)$ are similarity-based heuristic indicators normalized over neighbor communities, not calibrated Bayesian posterior probabilities.
5. **Computational Complexity:** The algorithm inherits FMHMO's $\mathcal{O}(m^2 + n^2)$ reduction cost, which scales quadratically on very large hypergraphs like `pubmed` ($m = 7,531$).

---

## 6. Conclusion & Future Work

In this project, we successfully addressed the three core limitations of the 2025 IEEE TFS paper by Xiao et al. We extracted the complete continuous fuzzy membership matrix $U$, implemented four overlap assignment rules, designed a planted-overlap synthetic hypergraph benchmark, and evaluated using standardized overlap metrics. Our empirical results demonstrate that thresholding rules R1 and R2 capture multi-community structure effectively (+12.9% ONMI improvement on planted overlap) while remaining fully robust against false overlap hallucination on crisp controls (0.00% false overlap rate).

Future work includes exploring iterative fuzzy updates where node-to-community memberships dynamically feed back into hyperedge community re-assignment, eliminating the dependence on a static initial crisp partition.

---

## References

1. J. Xiao, L. Ma, J. Cao, and X. Xu, *"Hypergraph Community Detection With Fuzzy Memberships"*, IEEE Transactions on Fuzzy Systems, vol. 33, no. 9, pp. 3255–3268, Sept. 2025.
2. A. Lancichinetti, S. Fortunato, and J. Kertész, *"Detecting the overlapping and hierarchical community structure in complex networks"*, New Journal of Physics, vol. 11, no. 3, p. 033015, 2009.
3. A. F. McDaid, D. Greene, and N. Hurley, *"Normalized Mutual Information to evaluate overlapping community findings"*, arXiv:1110.2515, 2011.
4. S. Gregory, *"Fuzzy clustering in complex networks"*, Information Sciences, vol. 181, no. 12, pp. 2487–2496, 2011.
5. N. Ruggeri, M. Contisciani, F. Battiston, and C. De Bacco, *"Community detection in large hypergraphs"*, Science Advances, vol. 9, no. 28, p. eadg9159, 2023.
6. M. Contisciani, F. Battiston, and C. De Bacco, *"Inference of hyperedges and overlapping communities in hypergraphs"*, Nature Communications, vol. 13, no. 1, p. 7229, 2022.
7. B. Kaminski, V. Poulin, P. Pralat, P. Szufel, and F. Théberge, *"Clustering via hypergraph modularity"*, PLoS ONE, vol. 14, no. 11, p. e0224307, 2019.
