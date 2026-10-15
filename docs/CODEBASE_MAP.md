# Codebase Map: FMHMO (Xiao et al., IEEE TFS 2025)

> **Generated for:** FMHMO-Overlap Project  
> **Phase 0:** Discover (Codebase exploration & mapping without modifying original code)  
> **Base Paper:** Xiao, Ma, Cao, Xu, *"Hypergraph Community Detection With Fuzzy Memberships"*, IEEE Trans. Fuzzy Systems, vol. 33, no. 9, 2025.  
> **Original Code Repository:** https://github.com/lisiran15/FMHMO

---

## 1. Directory Tree & Architecture Overview

```
project/
├── FMHMO/                                       # Authors' original code (DO NOT MODIFY)
│   ├── FMHMO/                                   # Main algorithm package
│   │   ├── FMHMO_louvan_real.py                 # Entry point: Real-world dataset execution
│   │   ├── FMHMO_louvan_lfr.py                  # Entry point: Synthetic LFR benchmark execution
│   │   ├── FMHMO_function.py                    # Hypergraph loader, modularity, NCR and NCM procedures
│   │   ├── FMHMO_utils.py                       # Reduction (HG_to_WG), node/hyperedge similarities, WGCD_to_HGCD
│   │   ├── cython_function.pyx                  # Cython source for Eq. 11-12 (mem_func) & hypergraph modularity
│   │   ├── cython_function.c                    # Generated C source from Cython
│   │   ├── cython_function.cp39-win_amd64.pyd   # Precompiled binary for Windows (Python 3.9 64-bit)
│   │   ├── setup.py                             # Cython build script (build_ext --inplace)
│   │   ├── strictModularity.py                  # Hypergraph modularity wrapper
│   │   ├── measure.py                           # Multi-metric evaluation (NMI, ARI, F1, FMI, AMI, etc.)
│   │   ├── measure_util.py / measure_FMI_*.py   # Metric helper utilities
│   │   ├── motif_network_construct.py           # Network construction helpers
│   │   ├── network_analysis_tools.py            # Graph analysis tools
│   │   ├── result_show.py                       # Console & log summary printing
│   │   ├── MyLogging.py                         # Logging utility
│   │   ├── readme.md                            # Basic instructions from authors
│   │   └── data/
│   │       ├── realNetwork/                     # 11 real-world hypergraphs (.txt)
│   │       │   └── real/                        # Ground-truth files (.txt) for all 11 real datasets
│   │       └── lfrNetwork/                      # Synthetic LFR benchmark datasets
│   ├── compareAlgorithm/                        # Baseline comparison algorithms
│   │   └── HNNCD/                               # HNNCD baseline algorithm code & data
│   └── readme.md                                # Root repository readme
├── docs/
│   └── CODEBASE_MAP.md                          # This document
├── final.md                                     # Master project plan & execution guidelines
└── Hypergraph_Community_Detection_With_Fuzzy_Memberships.pdf
```

---

## 2. Language, Environment & Dependencies

- **Primary Language:** Python 3 (Authors specify `Python 3.7+`).
- **Performance Modules:** Cython / C extension (`cython_function.pyx`, `cython_function.c`).
  - *Important observation:* The repository includes a pre-compiled Windows `.pyd` file (`cython_function.cp39-win_amd64.pyd`). On Linux, this extension **must be compiled locally** (`python setup.py build_ext --inplace`) producing a `.so` shared object.
- **Key Dependencies:**
  - `numpy`: Matrix operations, arrays.
  - `networkx`: Graph representations and component checks.
  - `python-igraph` (`igraph`): Louvain / multilevel community detection and shortest paths on reduced graph.
  - `python-louvain` (`community`): Louvain partition on connected weighted graphs.
  - `scikit-learn` (`sklearn.metrics`): NMI, ARI, mutual info metrics.
  - `scipy`: Sparse matrices, statistical routines.
  - `Cython` & `setuptools`: Required for compiling `cython_function`.

---

## 3. Entry Points & Workflow

### 3.1 Entry Points
1. **Real-world networks:** `FMHMO/FMHMO/FMHMO_louvan_real.py`
   - Sweeps parameter $\gamma \in [0.0, 1.0]$ in increments of $0.1$.
   - Loads dataset specified in line 35 (`net='hospital'`).
   - Runs Louvain / multilevel modularity optimization on the reduced graph.
   - Evaluates crisp communities using crisp metrics.
2. **Synthetic LFR networks:** `FMHMO/FMHMO/FMHMO_louvan_lfr.py`
   - Evaluates performance across different mixing parameters $\mu$.

### 3.2 High-Level Workflow of FMHMO
1. **Hypergraph Creation (`createHyperGraph` in `FMHMO_function.py:24`):**
   - Reads edge-list `.txt` file; detects 0-indexed vs 1-indexed.
   - Organizes hyperedges into a dictionary/list grouped by hyperedge cardinality.
2. **Hypergraph Info & Similarity (`HG_info_obtain` in `FMHMO_utils.py:74`):**
   - Computes node incident hyperedges (`node_HE_dict`).
   - Computes node pairwise Jaccard similarities $S(v, v')$ (**Eq. 1**).
   - Computes node-to-hyperedge memberships $u(v, e)$ (**Def. 1, Eq. 2**).
3. **Hyperedge-Scaled Hypergraph Reduction (`HG_to_WG` in `FMHMO_utils.py:162`):**
   - Projects hyperedges as nodes in a weighted graph (`WG` in igraph, `WG_nx` in NetworkX).
   - Weights edges using hybrid similarity parameter $\gamma$ (**Eq. 10**).
4. **Weighted Modularity Optimization (`FMHMO_louvan_real.py:61-78`):**
   - Applies `community_louvain.best_partition` or `WG.community_multilevel` on the reduced graph.
5. **Partition Recovery & Fuzzy Membership Assignment (`WGCD_to_HGCD` in `FMHMO_utils.py:265`):**
   - Maps hyperedge community labels back to nodes.
   - Resolves bridge nodes using **NCR** (Eqs. 11–12) and **NCM** (Eqs. 3–5).
   - **Crucial Step:** Assigns node to a single community via `argmax`.

---

## 4. Input & Output Formats

### 4.1 Input Formats
- **Hypergraph Edge List (`.txt` files):**
  - Found in `FMHMO/FMHMO/data/realNetwork/<name>.txt`.
  - Format: One line per hyperedge, containing node IDs separated by comma or space.
  - Example (`hospital.txt`):
    ```
    1 2
    3 4 5
    ...
    ```
  - Indexing: `createHyperGraph` automatically adjusts: if `'0'` appears, IDs are 0-indexed; otherwise `node_id - 1`.

- **Ground Truth Files (`.txt` files):**
  - Found in `FMHMO/FMHMO/data/realNetwork/real/<name>_groundtruth.txt`.
  - Format: One integer community label per line corresponding to node $0, 1, \dots, n-1$.
  - Read by `real_cd_obtain` (`FMHMO_utils.py:251`). Converts 1-indexed labels to 0-indexed if $0$ is absent.

### 4.2 Available Real-World Datasets & Ground Truth
All 11 real datasets in `FMHMO/FMHMO/data/realNetwork/` have corresponding ground truth files in `real/`:
| Dataset | Edge List Path | Ground Truth Path |
|---|---|---|
| `hospital` | `data/realNetwork/hospital.txt` | `data/realNetwork/real/hospital_groundtruth.txt` |
| `workspace` | `data/realNetwork/workspace.txt` | `data/realNetwork/real/workspace_groundtruth.txt` |
| `high_school` | `data/realNetwork/high_school.txt` | `data/realNetwork/real/high_school_groundtruth.txt` |
| `primary_school` | `data/realNetwork/primary_school.txt` | `data/realNetwork/real/primary_school_groundtruth.txt` |
| `senate_committees` | `data/realNetwork/senate_committees.txt` | `data/realNetwork/real/senate_committees_groundtruth.txt` |
| `house_committees` | `data/realNetwork/house_committees.txt` | `data/realNetwork/real/house_committees_groundtruth.txt` |
| `cora` | `data/realNetwork/cora.txt` | `data/realNetwork/real/cora_groundtruth.txt` |
| `cora_ca` | `data/realNetwork/cora_ca.txt` | `data/realNetwork/real/cora_ca_groundtruth.txt` |
| `enron_email` | `data/realNetwork/enron_email.txt` | `data/realNetwork/real/enron_email_groundtruth.txt` |
| `gene_disease` | `data/realNetwork/gene_disease.txt` | `data/realNetwork/real/gene_disease_groundtruth.txt` |
| `pubmed` | `data/realNetwork/pubmed.txt` | `data/realNetwork/real/pubmed_groundtruth.txt` |

---

## 5. Mathematical Equations: Location & Line References

| Equation in Paper | Mathematical Concept | File & Line Reference | Code Snippet / Implementation Details |
|---|---|---|---|
| **Eq. 1** | Node Pairwise Jaccard Similarity $S(v, v')$ | `FMHMO/FMHMO/FMHMO_utils.py:130-143` | `intersection = HEi & HEj`<br>`Sij_adj[i, j] = len(intersection) / len(union)` |
| **Def. 1, Eq. 2** | Node-to-Hyperedge Membership $u(v, e)$ | `FMHMO/FMHMO/FMHMO_utils.py:145-154` | `Uie[he] = sum([Sij_adj[i, j] for j in he if j!=i]) / (len(he)-1)`<br>Normalized: `Uie[he] /= Uie_sum` |
| **Eq. 10** | Hyperedge-to-Hyperedge Weight $w_{ij}$ (HS-HR reduction with $\gamma$) | `FMHMO/FMHMO/FMHMO_utils.py:180-200` | `wij = gamma * simlary_hei_hej + (1-gamma) * jaccard_simlary` |
| **Def. 4, Eqs. 11–12** | Hyperedge-to-Community Membership on reduced graph ($u(e, C)$ via Dijkstra distance) | `FMHMO/FMHMO/cython_function.pyx:157-200` (`mem_func`) | Average shortest path distance `avgdis`: lines 172-178.<br>Connection strength / `avgdis`: line 187.<br>Normalized `_mem`: line 194. |
| **Def. 2, Eq. 3** | Hyperedge-to-Community Membership in NCM ($u(e, C)$ via node overlap ratio) | `FMHMO/FMHMO/FMHMO_function.py:155` | `rv = heic_num_dict[cno_] / len(he)` |
| **Def. 3, Eqs. 4–5** | Node-to-Community Membership $u(v, C)$ | `FMHMO/FMHMO/FMHMO_function.py:121` (NCR)<br>`FMHMO/FMHMO/FMHMO_function.py:156, 160-173` (NCM) | NCR: `i_jcno_mem[he_cno] += (Uie_dict[i][he] * he_cno_mem)`<br>NCM: `i_jcno_mem[cno_] += (Uie_dict[i][he] * rv)`<br>Normalized over neighbor communities: lines 161, 172. |
| **Argmax Assignment** | **Discards fuzzy memberships and forces crisp partition (The Core Research Gap)** | `FMHMO/FMHMO/FMHMO_function.py:122` (in `NCR`)<br>`FMHMO/FMHMO/FMHMO_function.py:174` (in `NCM`) | `X[i] = max(i_jcno_mem.keys(), key=i_jcno_mem.get)`<br>`i_max_cno = max(i_jcno_mem.keys(), key=i_jcno_mem.get)` |

---

## 6. Detailed Analysis of the Core Research Gap

In `FMHMO_function.py`, inside the `NCM` function (lines 145–174):
```python
# Node v belongs to multiple neighbor communities jcno:
for jcno in j_nodes_setc: i_jcno_mem[jcno] = 0.0

# Memberships u(v, C) are accumulated:
for he in i_hes:
    he_cs = [X[hei] for hei in set(he)-set([i])]
    heic_num_dict = dict(Counter(he_cs))
    for cno_ in heic_num_dict.keys():
        rv = heic_num_dict[cno_]/len(he)
        i_jcno_mem[cno_] += (Uie_dict[i][he] * rv)

# Normalized across neighbor communities:
sum_v = sum(list(i_jcno_mem.values()))
for jcno in i_jcno_mem.keys(): i_jcno_mem[jcno] = i_jcno_mem[jcno]/sum_v

# Node similarity weighting:
...
sum_v = sum(list(i_jcno_mem.values()))
for jcno in i_jcno_mem.keys():
    i_jcno_mem[jcno] = i_jcno_mem[jcno]/sum_v

# THE GAP: Full distribution i_jcno_mem is thrown away, only argmax is retained!
i_max_cno = max(i_jcno_mem.keys(), key=i_jcno_mem.get)
X[i] = i_max_cno
```

**What FMHMO-Overlap will do:**
Instead of computing `i_max_cno` and assigning node $i$ to only one community, we extract the complete distribution `i_jcno_mem` into matrix $U \in \mathbb{R}^{n \times c}$, where $U[i, C] = u(i, C)$. We apply rules R0–R3 to allow nodes with non-trivial multi-community memberships to join multiple communities.

---

## 7. Open Questions & Technical Considerations

1. **Cython Compilation on Linux:**
   - The repository only has a pre-built Windows `.pyd` file (`cython_function.cp39-win_amd64.pyd`).
   - We must compile `cython_function.pyx` on Linux using GCC.
   - Since Python 3.12+ removed `distutils`, we should ensure `setuptools` and `Cython` are available in our virtual environment.
2. **NCR vs NCM in Wrapper:**
   - In `FMHMO_function.py`, `NCR_NCM` runs both `NCR` and `NCM` and checks which partition yields higher hypergraph modularity `modularityH`.
   - In `overlap/memberships.py`, we should extract the membership matrix $U$ corresponding to the final chosen recovery stage (or expose both options cleanly).
3. **Non-Bridge Nodes:**
   - Nodes whose hyperedges all belong to a single community have only 1 neighbor community. In $U$, their row will be one-hot (`1.0` in that community, `0.0` elsewhere).
   - This directly confirms Limitation 1 stated in `final.md` (single-hyperedge or intra-community nodes cannot exhibit overlap).
