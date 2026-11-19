# Project Report: FMHMO-Overlap
## Finding Overlapping Communities in Group Networks (Hypergraphs)

**Course:** Social Computing & Analysis (SCA) — 5th Semester  
**Program:** BCA (Bachelor of Computer Applications), Bennett University  
**Supervisor / Mentor:** Dr. Soumita Das  
**Students:**  
* **Piyush Agarwal** (Enrollment: `S24BCAU0182`) — *Method & Algorithm Architecture Lead*  
* **Suraj Kumar Tiwari** (Enrollment: `S24BCAU0136`) — *Evaluation & Benchmarking Lead*  
**Base Research Paper:** Xiao, Ma, Cao, Xu, *"Hypergraph Community Detection With Fuzzy Memberships"*, **IEEE Transactions on Fuzzy Systems (TFS)**, vol. 33, no. 9, pp. 3255–3268, September 2025.

---

## Executive Summary (The Project in 1 Minute)

In the real world, people and items connect in **groups** (like a project team, a hospital shift, or co-authors on a paper). In computer science, networks that connect groups of any size are called **Hypergraphs**.

In September 2025, a top research paper in *IEEE Transactions on Fuzzy Systems* proposed an algorithm called **FMHMO** to discover communities (departments or clusters) in hypergraphs. The paper calculated continuous "belonging percentages" (fuzzy memberships) for every person. 

**However, the original paper had a major flaw:**  
At the very last line of code, the authors used a rigid `argmax` filter. Even if a doctor spent 51% of their time in Surgery and 49% in Intensive Care, the algorithm forced them into only one department and **threw away the 49% score completely**. Despite having *"Fuzzy Memberships"* in the title, the paper produced strictly single-community, non-overlapping groups!

**What We Did (Our Solution):**  
We built **FMHMO-Overlap**. We kept the authors' underlying calculations intact, but:
1. We saved the **full matrix of belonging percentages** (Matrix $U$) instead of throwing it away.
2. We designed **flexible membership rules** (like a threshold of 25%) so that if someone genuinely belongs to multiple groups, they are allowed to join all of them.
3. We built a **synthetic test generator** to prove our accuracy against ground-truth benchmarks.
4. We created an **interactive web app (Streamlit)** where anyone can adjust sliders and see bridge nodes highlighted in real time.

**Key Result:** On networks with overlapping members, the original paper scored **0.000** (completely blind to overlap), while our method achieved a **0.765 F1-score**. On crisp networks with zero overlap, our method maintained a **0.00% false overlap rate** (no fake groups created).

---

## 1. Introduction: Why Do We Need Hypergraphs and Overlap?

### 1.1 The Difference Between Simple Graphs and Hypergraphs
* **Simple Graph:** Edges connect only **two people at a time** (like two Facebook friends).
* **Hypergraph:** A single connection (called a **Hyperedge**) connects **any number of people together at once**.
  * *Example 1 (Hospital):* An operation theater hyperedge connects $\{1\text{ Lead Surgeon}, 2\text{ Assistant Doctors}, 2\text{ Nurses}\}$.
  * *Example 2 (Academics):* A research paper hyperedge connects $\{4\text{ Co-authors}\}$.
  * *Example 3 (Office):* A weekly sprint meeting hyperedge connects $\{6\text{ Team Members}\}$.

If you try to convert a 5-person group into pairs, you create 10 artificial connections. This creates false density and loses the true group structure. Hypergraphs preserve the exact group boundaries.

### 1.2 The Reality of Overlap
Most community detection algorithms assume that every person belongs to **only one community** (strictly crisp/disjoint). 
In real life, this is almost never true:
* A student belongs to the *Coding Club*, the *Basketball Team*, and their *Classroom Batch*.
* In a hospital, a specialized radiologist works with both the *Oncology Department* and the *Emergency Room*.
* In an office, a manager coordinates between the *Product Design Team* and the *Engineering Team*.

Our goal is to accurately discover these **overlapping bridge individuals**.

---

## 2. The Base Paper & The "Argmax" Paradox

### 2.1 What the Base Paper (Xiao et al., IEEE TFS 2025) Did
The authors proposed **FMHMO**, which operates in 3 main stages:
1. **Stage 1 (Reduction):** Converts hyperedges into a weighted graph using a similarity parameter $\gamma$.
2. **Stage 2 (Clustering):** Uses the well-known Louvain algorithm to group hyperedges into clusters.
3. **Stage 3 (Recovery):** Calculates continuous fuzzy membership values $u(v, C)$ between $0.0$ and $1.0$ showing how strongly node $v$ belongs to community $C$.

### 2.2 The Paradox (The Research Gap)
Here is the exact problem we found in their paper and code:
* The paper title claims: *"Hypergraph Community Detection With **Fuzzy Memberships**"*.
* The math calculates continuous values (e.g., $u(v, C_1) = 0.54$, $u(v, C_2) = 0.41$).
* **BUT in the actual source code (`FMHMO_function.py`, lines 122 & 174):**
  $$\text{Community}[v] = \arg\max u(v, C)$$
  The authors took only the maximum value and collapsed every node into a single group!
* **The 3 Limitations of the Paper:**
  1. **L1 (Wasted Information):** All non-maximum percentages were permanently erased.
  2. **L2 (No Overlap Testing):** The authors never tested their algorithm on networks with known overlapping answers.
  3. **L3 (Incompatible Metrics):** They only used crisp metrics (standard NMI and ARI) that cannot measure overlapping groups.

---

## 3. Our Proposed System: FMHMO-Overlap

Instead of modifying or breaking the base paper's complex code, we wrapped around it like a protective shell.

```
       [ Input Hypergraph ]
                │
                ▼
      [ Base FMHMO Pipeline ] (Kept Untouched)
  - Reduces hypergraph via gamma
  - Clusters hyperedges using Louvain
  - Computes belonging scores u(v, C)
                │
                ▼
  [ Continuous Membership Matrix U ] (Our First Innovation)
  - Dimensions: n nodes × c communities
  - Row Sum = 1.0 (e.g., [0.54, 0.41, 0.05])
  - Non-bridge nodes = [1.0, 0.0, 0.0] (100% pure)
                │
                ▼
     [ 4 Overlap Assignment Rules ] (Our Second Innovation)
  - Rule R0: Argmax (matches original paper)
  - Rule R1: Fixed Threshold (tau = 0.25)  <-- Recommended
  - Rule R2: Relative Threshold (rho = 0.70)
  - Rule R3: Top-k Communities (k = 2)
                │
                ▼
    [ Overlapping Communities Cover ]
  - Members can now belong to 1, 2, or 3 communities
```

### 3.1 The Continuous Membership Matrix $U$
For every person $v$, we store their score across all communities in a table (Matrix $U$).
* **Pure Members (Non-bridge):** If all meetings of a person belong to Department 1, their row is `[1.0, 0.0, 0.0]`.
* **Bridge Members:** If a person attends meetings across multiple departments, their row shows the split: `[0.54, 0.41, 0.05]`.
* **Strict Rule:** Every row always sums up to exactly $1.0$ (100%).

### 3.2 The 4 Assignment Rules (How Scores Become Groups)
How do we decide if a score is high enough to join a community?

1. **Rule R0 (Crisp Argmax Baseline):**  
   Pick only the highest score. This reproduces the original paper's crisp output so we can compare against it.
2. **Rule R1 (Fixed Threshold $\tau$, Default = 0.25):**  
   If your score in a community is $25\%$ or higher, you join it. We always include your highest community so no one is left with zero groups.  
   *Example:* A person with `[0.54, 0.41, 0.05]` joins Community 1 ($54\% \ge 25\%$) AND Community 2 ($41\% \ge 25\%$).
3. **Rule R2 (Relative Threshold $\rho$, Default = 0.70):**  
   You join any community whose score is at least $70\%$ of your highest score. This automatically adapts for people who interact with many small groups.
4. **Rule R3 (Top-$k$, Default $k = 2$):**  
   You simply join your top 2 non-zero communities.

---

## 4. How We Tested It (The Evaluation Framework)

Because real-world datasets only label people with one group, you cannot prove overlap accuracy on real data alone. Therefore, we designed a rigorous testing framework:

### 4.1 Synthetic Planted-Overlap Generator (`generate.py`)
We built a computer generator that creates artificial hypergraphs where we **already know the 100% correct answers**:
* We create a network of $200$ people in $4$ communities.
* We intentionally plant an overlap fraction $p_{\text{ov}}$ (e.g., $10\%$ or $20\%$ of people are chosen to belong to $2$ communities).
* When $p_{\text{ov}} = 0.0$, the network has **zero overlap** (used as a control test).

### 4.2 Overlap-Aware Metrics (`metrics.py`)
To judge overlapping results fairly, we implemented standard research metrics:
1. **Overlapping NMI (ONMI):** Measures how much information the detected groups share with the true groups (Score: $0.0$ to $1.0$).
2. **Omega Index ($\Omega$):** The overlapping version of the famous Adjusted Rand Index. It checks whether pairs of people share the correct number of groups ($0, 1, \text{ or } 2$).
3. **Overlap-Node Precision, Recall & F1:** Specifically checks how accurately we identified the bridge nodes.

---

## 5. Experimental Results: What Did We Discover?

All experiments were executed from dedicated Python scripts in `experiments/` and saved to `results/`.

### 5.1 Experiment E0: Paper Reproduction
First, we verified whether we could reproduce the base paper's published numbers on their own datasets.
* **Dataset: `workspace` (92 office employees, 788 contact hyperedges):**
  * Published NMI in Paper: **0.836** $\to$ Our Reproduced NMI: **0.836** (Exact 3-decimal match!)
  * Published ARI in Paper: **0.849** $\to$ Our Reproduced ARI: **0.849** (Exact match!)
  * Published F1 in Paper: **0.924** $\to$ Our Reproduced F1: **0.924** (Exact match!)
* *Conclusion:* Our environment and execution pipeline are 100% mathematically correct.

### 5.2 Experiment E1: Can We Detect Overlapping Nodes?
We tested our rules on networks with $20\%$ planted overlapping nodes ($p_{\text{ov}} = 0.20$):

| Method / Rule | Overlap-Node F1 | Overlapping NMI | Omega Index | Finding |
|---|:---:|:---:|:---:|---|
| **Original Paper (R0 Argmax)** | **0.000** | 0.657 | 0.664 | **Completely blind to overlap** |
| **FMHMO-Overlap (R1 $\tau=0.25$)** | **0.765** | **0.786** | **0.828** | **Detects overlapping nodes with high accuracy** |
| **FMHMO-Overlap (R2 $\rho=0.70$)** | 0.400 | 0.721 | 0.735 | High-precision, conservative overlap |

* **Key Takeaway:** The original paper's crisp partition achieves **0.000 F1** on overlapping nodes because it is physically incapable of placing someone in two groups. Our Rule R1 achieves **0.765 F1** and boosts overall quality (ONMI) by **+12.9%**.

### 5.3 Experiment E4: The Control Test (Zero False Overlap)
A common criticism of threshold rules is: *"What if the network is purely crisp? Will your rule invent fake overlapping members?"*
* We ran tests on networks with strictly **zero overlap ($p_{\text{ov}} = 0.0$)**:
  * Rule R1 ($\tau = 0.25$): **0.00% False Overlap Rate**
  * Rule R2 ($\rho = 0.70$): **0.00% False Overlap Rate**
  * Rule R3 (Top-2): 52% to 68% False Overlap Rate (Failed)
* *Conclusion:* **Rules R1 and R2 are rock-solid.** If there is no overlap in the network, they output 0.00% fake overlap because pure members naturally have scores of $1.0$ in one group and $0.0$ elsewhere.

### 5.4 Experiment E3: Why Hypergraphs Beat Simple Graphs
We compared FMHMO-Overlap against standard **Clique Expansion** (breaking groups into pairs + Louvain + soft assignment):
* At low overlap ($10\%$), both methods are comparable.
* At higher overlap ($20\%$), **FMHMO-Overlap wins decisively (ONMI: 0.786 vs 0.643, a +22.2% advantage)**.
* *Why?* Turning large groups into pairs creates a giant tangled mess of artificial lines. Preserving genuine hyperedges keeps group boundaries clean.

---

## 6. Real-World Case Studies

We ran our algorithm on real-world contact networks:

### Case Study 1: Hospital Staff Network (`hospital.txt`)
* **Size:** $75$ healthcare workers, $1,825$ interaction hyperedges.
* Under Rule R1 ($\tau = 0.25$), **36.0% of staff members belong to multiple functional communities** (average $1.37$ communities per worker).
* **Key Bridge Individual (Node 74):**
  * Involved in $11$ distinct group interactions across shifts.
  * Membership distribution: **54.2% in Department 5**, **40.9% in Department 2**, and **4.6% in Department 3**.
  * *Interpretation:* Node 74 is a senior doctor/liaison who actively connects Department 5 and Department 2, essential for coordinating patient handovers between wards.

### Case Study 2: Workplace Office Network (`workspace.txt`)
* **Size:** $92$ employees, $788$ meeting hyperedges.
* Under Rule R1, **10.87% of employees overlap**.
* **Key Bridge Individual (Node 24):**
  * Involved in $9$ meeting hyperedges.
  * Membership distribution: **54.1% in Team 2**, **39.5% in Team 1**, and **6.4% in Team 3**.
  * *Interpretation:* A cross-functional project manager coordinating between two primary project groups.

---

## 7. Interactive Web App Demo (`demo/app.py`)

To make our research interactive and accessible, we developed a web dashboard using **Streamlit**:
* **Data Selection:** Users can choose between synthetic networks, the paper's Figure 1 toy example, or the real Hospital dataset.
* **Live Sliders:** Adjust threshold $\tau$, ratio $\rho$, or top-$k$ in real time.
* **Network Visualization:** Renders the network using NetworkX. Members belonging to a single group are colored **Blue**, while multi-community bridge members light up in **Bright Red**.
* **Node Inspector:** Click any person's ID to view a clean bar chart of their exact belonging percentages across every community.

To launch the app:
```bash
streamlit run demo/app.py
```

---

## 8. Honest Academic Limitations (What Our Method Cannot Do)

In real research, it is essential to be honest about where a method reaches its limits:

1. **Single-Meeting Members Cannot Overlap:**  
   If a person participates in only 1 hyperedge, 100% of their score concentrates on that single event ($u = 1.0$). Even if that person belongs to two groups in real life, the hypergraph has no structural way of knowing it.
2. **Dependency on Initial Hyperedge Clustering:**  
   Memberships are calculated *after* hyperedges have been grouped by the Louvain algorithm. If all meetings that a person attends happen to be grouped into the same community cluster, their overlap will be invisible.
3. **Real-World Ground Truth is Crisp:**  
   Standard real-world datasets only provide a single label per person. Therefore, numerical overlap accuracy can only be proven on synthetic benchmarks. Real-world findings must be treated as structural and qualitative analyses.

---

## 9. Work Division: Who Did What?

The work was divided cleanly between the two team members:

### Piyush Agarwal (`S24BCAU0182`) — *Method & Algorithm Architecture Lead*
* Reverse-engineered the authors' original Cython/Python code (`docs/CODEBASE_MAP.md`).
* Built the baseline reproduction harness (`experiments/run_E0.py`) and matched published numbers.
* Created the core extraction module for Matrix $U$ (`overlap/memberships.py`).
* Implemented the 5 assignment rules and overlap statistics (`overlap/rules.py` & `stats.py`).
* Conducted the real-world case studies (`experiments/run_E5.py`).
* Developed the full interactive Streamlit web dashboard (`demo/app.py`).

### Suraj Kumar Tiwari (`S24BCAU0136`) — *Evaluation & Benchmarking Lead*
* Developed the planted-overlap synthetic hypergraph generator (`overlap/generate.py`).
* Formulated and coded the overlap metric suite: ONMI, Omega Index, and Overlap PRF (`overlap/metrics.py`).
* Implemented the Clique Expansion baseline competitor (`overlap/baselines.py`).
* Executed benchmark experiments E1 (rules vs overlap), E2 (sensitivity), E3 (baselines), and E4 (control test).
* Authored the academic LaTeX manuscript (`report/main.tex`) and generated publication figures.

---

## 10. Conclusion

In this project, we successfully identified and resolved the core paradox of a 2025 *IEEE Transactions on Fuzzy Systems* paper. By preserving continuous fuzzy memberships instead of discarding them with `argmax`, **FMHMO-Overlap** recovers realistic multi-community structures (+12.9% ONMI improvement) while remaining completely protected against false overlap hallucinations (0.00% error rate).

The entire codebase is verified with **13 out of 13 unit tests passing**, backed by reproducible experimental scripts, and supported by an intuitive web demo.
