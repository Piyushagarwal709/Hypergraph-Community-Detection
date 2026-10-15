"""
demo/app.py
Interactive Streamlit Application for FMHMO-Overlap.
Allows interactive exploration of:
- Real-world hypergraphs (hospital, workspace, senate_committees) or synthetic planted-overlap hypergraphs.
- Assignment rules (R0 argmax, R1 fixed tau, R2 relative rho, R3 top-k) with real-time parameter tuning.
- Interactive network visualization highlighting overlapping bridge nodes.
- Per-node fuzzy membership grade inspection.
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from overlap.generate import generate
from overlap.memberships import (
    extract_memberships_from_raw_hyperedges,
    get_fig1_toy_hypergraph
)
from overlap.rules import assign
from overlap.stats import compute_overlap_statistics


st.set_page_config(
    page_title="FMHMO-Overlap: Hypergraph Community Detection",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ FMHMO-Overlap: Overlapping Communities from Fuzzy Memberships")
st.markdown(
    "Based on **Xiao et al. (IEEE Trans. Fuzzy Systems 2025)**. "
    "Unlike the original paper which discards fuzzy distributions via argmax, "
    "this system exposes the full membership matrix **$U$** to discover true overlapping communities."
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic Planted Overlap", "Paper Figure 1 Example", "Real Dataset (Hospital)"]
)

if data_source == "Synthetic Planted Overlap":
    n_nodes = st.sidebar.slider("Number of Nodes (n)", 50, 300, 100, step=25)
    c_comms = st.sidebar.slider("Number of Communities (c)", 2, 6, 3)
    p_ov = st.sidebar.slider("Planted Overlap Fraction (p_ov)", 0.0, 0.5, 0.2, step=0.05)
    noise_xi = st.sidebar.slider("Hyperedge Noise (xi)", 0.0, 0.5, 0.15, step=0.05)
    seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    
    hyperedges, truth = generate(n=n_nodes, c=c_comms, p_ov=p_ov, xi=noise_xi, seed=seed)
    
elif data_source == "Paper Figure 1 Example":
    hyperedges = get_fig1_toy_hypergraph()
    n_nodes = 13
    
else:
    # Load hospital dataset
    hosp_file = os.path.join(PROJECT_ROOT, "FMHMO", "FMHMO", "data", "realNetwork", "hospital.txt")
    hyperedges = []
    if os.path.exists(hosp_file):
        with open(hosp_file, "r") as f:
            for line in f:
                parts = line.strip().replace(",", " ").split()
                if parts:
                    hyperedges.append([int(p) - 1 if "0" not in line else int(p) for p in parts])
    n_nodes = max([max(he) for he in hyperedges]) + 1 if hyperedges else 75

# Run FMHMO & extract U
with st.spinner("Computing hypergraph reduction and membership matrix U..."):
    U, crisp_part, detected_c = extract_memberships_from_raw_hyperedges(hyperedges, gamma=0.5, seed=42)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Assignment Rule")

rule_choice = st.sidebar.selectbox(
    "Select Rule",
    ["R1: Fixed Threshold (tau)", "R2: Relative Threshold (rho * u_max)", "R3: Top-k Communities", "R0: Argmax (Original Crisp)"]
)

if rule_choice.startswith("R1"):
    rule_id = "R1"
    param_val = st.sidebar.slider("Threshold (tau)", 0.05, 0.50, 0.25, step=0.05)
elif rule_choice.startswith("R2"):
    rule_id = "R2"
    param_val = st.sidebar.slider("Relative Fraction (rho)", 0.40, 0.95, 0.70, step=0.05)
elif rule_choice.startswith("R3"):
    rule_id = "R3"
    param_val = st.sidebar.slider("Top-k", 1, min(4, detected_c), 2)
else:
    rule_id = "R0"
    param_val = None

# Compute overlapping assignment
communities = assign(U, rule=rule_id, param=param_val)
stats = compute_overlap_statistics(communities, n_nodes)

# ----------------- MAIN DISPLAY -----------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Nodes", stats["num_nodes"])
col2.metric("Hyperedges", len(hyperedges))
col3.metric("Detected Communities", detected_c)
col4.metric("Overlapping Nodes", f"{stats['num_overlapping_nodes']} ({stats['percent_overlapping']}%)")
col5.metric("Avg Memberships / Node", stats["avg_memberships"])

st.markdown("---")

# Layout: Graph visualization & Node Inspector
v_col1, v_col2 = st.columns([3, 2])

with v_col1:
    st.subheader("🌐 Hypergraph 2-Section Network Visualization")
    
    # Build NetworkX 2-section graph for visualization
    G = nx.Graph()
    G.add_nodes_from(range(min(n_nodes, 120))) # limit nodes for responsive plotting
    for he in hyperedges:
        sub_he = [v for v in he if v < 120]
        for i in range(len(sub_he)):
            for j in range(i + 1, len(sub_he)):
                G.add_edge(sub_he[i], sub_he[j])
                
    pos = nx.spring_layout(G, seed=42, k=0.35)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors for communities
    cmap = plt.cm.get_cmap("tab10", detected_c)
    node_colors = [cmap(crisp_part[i] % detected_c) for i in G.nodes()]
    
    # Distinguish single vs overlapping nodes
    overlapping_set = set(stats["overlapping_node_ids"])
    node_sizes = [220 if i in overlapping_set else 80 for i in G.nodes()]
    node_edgecolors = ["red" if i in overlapping_set else "white" for i in G.nodes()]
    node_linewidths = [2.5 if i in overlapping_set else 0.8 for i in G.nodes()]
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="gray")
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edgecolors,
        linewidths=node_linewidths
    )
    
    # Label overlapping nodes
    if len(overlapping_set) <= 30:
        labels = {i: str(i) for i in G.nodes() if i in overlapping_set}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_weight="bold")
        
    ax.set_title(f"Nodes with Red Halos = Overlapping ({stats['percent_overlapping']}%)", fontsize=11)
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

with v_col2:
    st.subheader("🔍 Node Fuzzy Membership Inspector")
    
    inspect_node = st.number_input("Select Node ID to inspect", min_value=0, max_value=n_nodes - 1, value=0)
    
    u_row = U[inspect_node]
    node_comms = [c_idx for c_idx in range(len(communities)) if inspect_node in communities[c_idx]]
    
    st.write(f"**Node {inspect_node} Assigned Communities:** {node_comms}")
    st.write(f"**Multi-community Status:** {'🔥 OVERLAPPING' if len(node_comms) > 1 else 'Single Community'}")
    
    # Bar chart of membership distribution across communities
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
    comm_labels = [f"C{c_idx}" for c_idx in range(detected_c)]
    bars = ax_bar.bar(comm_labels, u_row, color="teal", alpha=0.8)
    
    # Highlight joined communities
    for c_idx in node_comms:
        if c_idx < len(bars):
            bars[c_idx].set_color("coral")
            
    ax_bar.set_ylabel("Membership Grade u(v, C)")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title(f"Membership Distribution for Node {inspect_node}")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)
    st.pyplot(fig_bar)
    plt.close()
    
    # Overlapping Nodes Table
    st.markdown("#### 📋 Overlapping Nodes Summary")
    if stats["overlapping_node_ids"]:
        top_overlaps = stats["overlapping_node_ids"][:10]
        data_rows = []
        for v in top_overlaps:
            comms_joined = [c_idx for c_idx in range(len(communities)) if v in communities[c_idx]]
            top_grades = sorted(U[v], reverse=True)[:2]
            data_rows.append({
                "Node ID": v,
                "Joined": str(comms_joined),
                "Top Grade": round(top_grades[0], 3),
                "2nd Grade": round(top_grades[1], 3) if len(top_grades) > 1 else 0.0
            })
        st.dataframe(data_rows, use_container_width=True)
    else:
        st.info("No nodes currently qualify as overlapping under this rule and threshold.")
