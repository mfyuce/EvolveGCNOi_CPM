"""
generate_gnn_datasets_v2.py
===========================
Produces the v2 CPM GNN dataset from the existing v1 JSON.

Changes vs v1
-------------
1. detection_confidence added as the 10th node feature (F 9 -> 10).
   Per-node value = mean incoming detection_confidence at each timestep
   (0 if the node received no CPM observation that step). Stored RAW; the
   loader standardises features at load time.

Edges (index + weight) are kept IDENTICAL to v1. An earlier draft z-scored
the self-loop edge weights, but the resulting negative weights made GCN
degree normalisation (deg^{-1/2}) produce NaN — edge weights feeding a GCN
must stay non-negative, so v2 changes node features only.

Output
------
  data/v2/burst_adma_with_cpm_multi_sensors999_cls_all_v2.json
  data/v2/burst_adma_with_cpm_multi_sensors999_cls_all_v2.zip
  data/v2/README.txt
"""
import json, zipfile, os, time
import numpy as np

V1_JSON  = "data/burst_adma_with_cpm_multi_sensors_cls/burst_adma_with_cpm_multi_sensors999_cls_all__positive_with_features_as_self_edge.json"
OUT_DIR  = "data/v2"
OUT_JSON = os.path.join(OUT_DIR, "burst_adma_with_cpm_multi_sensors999_cls_all_v2.json")
OUT_ZIP  = os.path.join(OUT_DIR, "burst_adma_with_cpm_multi_sensors999_cls_all_v2.zip")
OUT_README = os.path.join(OUT_DIR, "README.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load v1 ────────────────────────────────────────────────────────────────
t0 = time.time()
print("Loading v1 JSON ...", flush=True)
with open(V1_JSON) as f:
    v1 = json.load(f)

T           = v1["time_periods"]
node_labels = v1["node_labels"]
N           = len(node_labels)
features    = np.array(v1["features"], dtype=np.float32)   # (T, N, 9)
y_detected  = v1["y_detected"]
edge_index  = v1["edge_index"]
edge_weight = v1["edge_weight"]
print(f"  loaded in {time.time()-t0:.1f}s  T={T} N={N} F={features.shape[2]}", flush=True)

# ── 2. Per-node mean detection_confidence → new node feature (dim 9) ─────────
# NOTE: edge weights are kept VERBATIM from v1. An earlier v2 draft z-scored the
# self-loop edge weights to mean 0; the resulting NEGATIVE weights made the GCN
# degree normalisation deg^{-1/2} produce NaN. Edge weights feeding a GCN must
# stay non-negative, so v2 only ADDS a node feature and leaves the graph alone.
print("Building detection_confidence node feature ...", flush=True)
conf_feat = np.zeros((T, N), dtype=np.float32)
for step_str, ei_list in edge_index.items():
    ew_list = edge_weight[step_str]
    t       = int(step_str)
    acc     = np.zeros(N, dtype=np.float64)
    cnt     = np.zeros(N, dtype=np.int32)
    for (src, dst), w in zip(ei_list, ew_list):
        if src != dst:             # proximity edge: w = detection_confidence
            acc[dst] += w
            cnt[dst] += 1
    mask = cnt > 0
    conf_feat[t, mask] = (acc[mask] / cnt[mask]).astype(np.float32)

print(f"  conf_feat: mean={conf_feat[conf_feat>0].mean():.3f}  max={conf_feat.max():.3f}", flush=True)

features_v2 = np.concatenate([features, conf_feat[:, :, np.newaxis]], axis=2)  # (T,N,10)

# ── 3. Edge weights: kept verbatim from v1 (must stay non-negative for GCN) ──
edge_weight_v2 = edge_weight

# ── 4. Write v2 JSON ──────────────────────────────────────────────────────────
print("Writing v2 JSON ...", flush=True)
t1 = time.time()
v2_out = {
    "version":      2,
    "time_periods": T,
    "node_labels":  node_labels,
    "features":     features_v2.tolist(),
    "y_detected":   y_detected,
    "edge_index":   edge_index,
    "edge_weight":  edge_weight_v2,
    "note": "v2 = v1 with a 10th node feature (mean incoming detection_confidence). "
            "Edges (index + weight) are identical to v1. Node features are RAW; "
            "standardise at load time (BurstAdmaDatasetLoader z-scores per feature)."
}
with open(OUT_JSON, "w") as f:
    json.dump(v2_out, f)
print(f"  written {os.path.getsize(OUT_JSON)/1e6:.0f} MB in {time.time()-t1:.1f}s", flush=True)

# ── 7. Zip ────────────────────────────────────────────────────────────────────
print("Zipping ...", flush=True)
t2 = time.time()
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(OUT_JSON, os.path.basename(OUT_JSON))
print(f"  ZIP {os.path.getsize(OUT_ZIP)/1e6:.0f} MB in {time.time()-t2:.1f}s", flush=True)

# ── 8. README ─────────────────────────────────────────────────────────────────
readme = """\
BurST-ADMA + CPM for GNN — temporal-graph dataset v2
======================================================

Temporal-graph dataset for V2X misbehavior detection with Collective
Perception Messages (CPM), PyTorch Geometric Temporal format.

Reference paper
---------------
  "State of the Art and First Synthetic Dataset for Misbehavior Detection
   with Collective Perception in [C-]V2X Networks"
  Yuce, Erturk, Aydin — IEEE PIMRC 2025

Changes vs v1
-------------
1. Node feature vector extended: F 9 -> 10. Index 9 = mean incoming
   detection_confidence for each node at each timestep (0 if no CPM
   observation received that step). Stored RAW; standardise at load time.

Edges (index + weight) are IDENTICAL to v1. (An earlier draft z-scored the
self-loop edge weights, but negative weights break GCN deg^{-1/2}
normalisation, so v2 changes node features only.)

Files
-----
  burst_adma_with_cpm_multi_sensors999_cls_all_v2.json / .zip

JSON schema
-----------
Key            Shape / Type         Description
----------     ------------------   -------------------------------------------
version        int                  2
time_periods   int                  T = 1000
node_labels    list[str]            730 vehicle ids; defines node index order
features       (T, N, 10) list      Per-node CPM feature vector
                                    Index 0  x_detected         (m, UTM easting)
                                    Index 1  y_detected         (m, UTM northing)
                                    Index 2  heading_detected   (degrees)
                                    Index 3  speed_detected     (m/s)
                                    Index 4  acceleration_det.  (m/s²)
                                    Index 5  sensor_range       (m)
                                    Index 6  visibility_range   (m)
                                    Index 7  sensor_type        (int-encoded)
                                    Index 8  weather_conditions (0=clear,1=foggy,
                                                                 2=rainy,3=snowy)
                                    Index 9  mean_det_conf      mean incoming
                                             detection_confidence (0-1); 0 if
                                             not detected this step  [NEW in v2]
y_detected     (T, N) list          Misbehavior label per node per step
                                    0 = benign; 1..7 = attack type
edge_index     {"t": [[s,d]]}       Proximity + self-loop edges per timestep
                                    (identical to v1)
edge_weight    {"t": [w]}           Aligned weights (identical to v1):
                                    prox edges  : detection_confidence (0-1)
                                    self-loops  : the node's own raw feature
                                                  values (features-as-self-edge)

Usage
-----
  from BurstAdmaDatasetLoader import BurstAdmaDatasetLoader
  lb = BurstAdmaDatasetLoader(
      json_path="data/v2/burst_adma_with_cpm_multi_sensors999_cls_all_v2.json",
      features_as_self_edge=True, binary=True)
  ds = lb.get_dataset(lags=1)
  # n_node_features == 10
"""
with open(OUT_README, "w") as f:
    f.write(readme)

print(f"\nDone in {time.time()-t0:.0f}s")
print(f"  JSON : {os.path.getsize(OUT_JSON)/1e6:.0f} MB")
print(f"  ZIP  : {os.path.getsize(OUT_ZIP)/1e6:.0f} MB")
print(f"  Dir  : {OUT_DIR}/")
