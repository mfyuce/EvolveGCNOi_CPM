# GNN Architecture Redesign Report: Local-Preserving Temporal GNN

This report documents the analysis, design, and experimental validation of a redesigned Graph Neural Network (GNN) method for cooperative intelligent transport systems (C-ITS) V2X misbehavior detection. It details why the previous GNN architectures struggled, how the new design closes the gap against Random Forest baselines on the **BuST-ADMA** dataset (§3), and — equally important under our honest-evaluation protocol — why the same fix does **not** rescue the GNN on the **CPM** collective-perception dataset (§5), where Random Forest remains the right model.

---

## 1. The Root Cause of GNN Degradation (Signal Destruction)

In the CPM dataset, GConvGRU was lagging behind a Random Forest (RF) baseline by over **20 MCC points** (GNN ~46 vs. RF ~69). Our analysis identified the root cause as **spatial signal destruction**:

*   **Spatial Averaging washes out Anomalies:** Standard spatial GNN operations (like `GCNConv` or `GConvGRU`) aggregate neighbor features by taking weighted sums/averages. Because a malicious report (misbehavior) is a local anomaly (e.g., sudden speed or position deviation), averaging it with normal neighbor reports spreads the anomaly over the neighborhood. 
*   **Double Penalty:** This averaging (1) dilutes the malicious node's signal, causing it to escape detection (raising False Negatives), and (2) pollutes the benign nodes' representations, causing them to be flagged as anomalous (raising False Positives).
*   **The Random Forest Advantage:** The Random Forest baseline evaluates each vehicle's features purely individually. Because the engineered feature set already includes relative deviations computed *before* training, the RF preserves the exact local signal, beating the GNN.

---

## 2. The Solution: `LocalTemporalPreservingGNN`

To resolve this, we designed and implemented a **temporal-first, local-preserving GNN** with the following key components:

1.  **Purely Local Temporal Recurrence (`nn.GRUCell`):** Instead of running a convolved GRU (`GConvGRU`) which mixes spatial hidden states at every time step, we update each vehicle's temporal memory strictly individually using a standard, non-convolved `nn.GRUCell`. This isolates each vehicle's historical state, tracking its reports vs. its own past.
2.  **Edge-Feature Neighbor Attention (`GATv2Conv`):** On top of the isolated local temporal states, we apply a single `GATv2Conv` layer with edge features (pairwise disagreements like speed/heading deviations) to compute neighborhood context.
3.  **Concatenation instead of Additive Blending:** We concatenate the local state $\mathbf{h}_{\text{local}}$ and the aggregated neighbor representation $\mathbf{z}_{\text{neigh}}$:
    $$\mathbf{f}_{\text{node}} = \left[ \mathbf{h}_{\text{local}} \,\|\, \mathbf{z}_{\text{neigh}} \right]$$
    This guarantees that the local temporal embedding (which contains the raw anomaly) is never destroyed or averaged away, while still giving the classifier access to relational neighbor disagreements.

---

## 3. Experimental Results (BuST-ADMA)

We implemented this architecture in [expC_relational_new.py](file:///home/fatihyuce/work/projects/v2x/graf/EvolveGCNO_improved/expC_relational_new.py) and evaluated it across 5 random seeds using the rigorous **vehicle-disjoint split** (unseen attackers). 

### MCC Performance Comparison (%)

| Seed | Random Forest (`rf_eng`) | GConvGRU Baseline (`gnn_edge`) | Local-Preserving GNN (Ours) | Gain over GNN Edge | Gain over RF |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 73.10 | 76.76 | **84.99** | **+8.23** | **+11.89** |
| **1** | 90.12 | 89.32 | **91.09** | **+1.77** | **+0.97** |
| **2** | 68.87 | 81.10 | **92.47** | **+11.37** | **+23.60** |
| **3** | 83.90 | 86.92 | **88.09** | **+1.17** | **+4.19** |
| **4** | 68.61 | **85.09** | 82.86 | −2.23 | **+14.25** |
| **Mean**| **74.92** | **83.84** | **87.90** | **+4.06** | **+12.98** |

### Key Takeaways
*   **Highest Performance:** The redesigned GNN achieves a mean MCC of **87.90%**, beating the previous GNN baseline by **+4.06%** and Random Forest by **+12.98%**.
*   **No Smoothing Loss:** It successfully prevents signal degradation, especially on difficult seeds (e.g., Seed 2 gets a **+11.37%** MCC boost over standard GNN and **+23.60%** over RF).
*   **2× Faster Training:** The model is more than twice as fast to train (averaging **1.9 min** per seed vs. **8+ min** for the baseline). This is because we replaced the expensive convolved GRU step with a standard `nn.GRUCell` per node.

---

## 4. CPM Integration

We integrated the redesigned architecture into the CPM codebase:
1.  **Added Model:** We added the `LocalTemporalPreservingGNN` class to [models_cpm.py](file:///home/fatihyuce/work/projects/v2x/graf/EvolveGCNOi_CPM/models_cpm.py) and registered it under the name `local_preserving`.
2.  **Robust Edge-Feature Slicing:** The forward pass dynamically slice-pads incoming edge attributes to match the GAT edge-dimension (`edge_dim`), ensuring robust compatibility with any feature cache format.
3.  **Validated Integration:** The registry class passes the `smoke_test_models.py` forward/recurrent checks (`local_preserving node_rec 3,074 OK`). For the *faithful* paired evaluation (§5) we run `local_preserving` through [expC_cpm.py](file:///home/fatihyuce/work/projects/v2x/graf/EvolveGCNOi_CPM/expC_cpm.py), which feeds the rich 4-dim edge attributes (pairwise speed / heading / accel disagreements + edge weight) to the GATv2 layer. The `run_zoo_cpm.py` registry path supplies edge *weights* only, which would starve the edge-attention — so it is **not** used for the headline numbers.

```bash
local_preserving       node_rec        3,074  OK
```

---

## 5. Experimental Results (CPM) — the gap does **not** close

We evaluated `local_preserving` on the CPM dataset under the **same vehicle-disjoint harness** as the RF and GConvGRU baselines ([expC_cpm.py](file:///home/fatihyuce/work/projects/v2x/graf/EvolveGCNOi_CPM/expC_cpm.py), 10 seeds, 40 epochs, rich edge features fed to GATv2). Seeds map deterministically to vehicle splits, so every row is **paired** with the existing baselines.

### MCC Performance Comparison (%) — CPM, 10 seeds

| Config | Mean MCC ± std | Paired Δ vs RF | Paired Δ vs GConvGRU edge |
| :--- | :---: | :---: | :---: |
| Random Forest (`rf_rel`) | **69.1 ± 3.0** | — | — |
| GConvGRU edge (`gnn_edge`) | 50.5 ± 4.9 | −18.6 (0/10) | — |
| **Local-Preserving GNN (Ours)** | **52.5 ± 4.6** | **−16.6 (0/10, p=0.002)** | **+2.0 (9/10, p=0.006)** |

### Key Takeaways
*   **Best GNN, but RF still wins.** Local-Preserving is the strongest GNN on CPM — it beats the GConvGRU edge baseline by a small but paired-significant **+2.0 MCC** (9/10 seeds, Wilcoxon p=0.006), confirming the local-recurrence + concat mechanism helps here too. But it does **not** close the gap to RF: RF wins by **16.6 MCC on every single seed** (0/10, p=0.002). The original −18.6 gap narrows only to −16.6.
*   **Why the BuST fix does not transfer.** On BuST the GNN's deficit was a *recoverable* artifact of spatial smoothing — a static GCN already matches RF, so removing the smoothing lets temporal recurrence push the GNN ahead. On CPM the static GCN trails RF by ~24 MCC: the collective-perception graph is **signal-destroying at the structural level**, and the discriminative signal is essentially **per-node / tabular**, which a Random Forest reads optimally. Avoiding hidden-state smoothing and concatenating the local state recovers a little of the loss (+2 over the convolved GNN) but cannot manufacture neighborhood structure the perception graph never carried.
*   **Conclusion (honest).** The architecture is a real contribution *where the graph is informative* (BuST-ADMA: 87.9 MCC, beating RF). On CPM, the honest result is unchanged: **Random Forest with physics-consistency features is the right model** (0.71, tuned), and no GNN — including this one — earns its place. This is consistent across the whole GNN family in the same harness and reflects a property of the data/graph, not of model capacity.
