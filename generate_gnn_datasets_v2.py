"""
generate_gnn_datasets_v2.py
===========================
Produces the v2 CPM GNN dataset from the existing v1 JSON.

Changes vs v1
-------------
1. Self-loop edge weights normalised: z-score per feature.
   Edge structure: every detection event produces 7 consecutive self-loops
   for the detecting node carrying its features in order:
     [x, y, heading, speed, acceleration, sensor_range, visibility_range]
   These raw values had a huge scale mismatch (x~145, y~-37) against the
   proximity edge weights which are detection_confidence in [0,1].
   v2 z-scores each feature slot using global mean/std computed across all
   non-zero nodes and timesteps.

2. detection_confidence added as 10th node feature (F 9 -> 10).
   Per-node value = mean incoming detection_confidence at each timestep
   (0 if the node received no CPM observation that step).

3. "feat_norm" key in JSON stores the normalisation statistics so loaders
   can invert the transform if needed.

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
SELF_FEAT_DIM = 7   # [x, y, heading, speed, acc, sensor_range, vis_range]

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

# ── 2. Normalisation stats from node feature array ───────────────────────────
# features[:, :, :7] = [x, y, heading, speed, acc, sensor_range, vis_range]
# Use only non-zero (active) nodes.
print("Computing normalisation stats ...", flush=True)
feat7       = features[:, :, :SELF_FEAT_DIM]           # (T, N, 7)
active_mask = (features.sum(axis=2) != 0)              # (T, N)
feat7_flat  = feat7[active_mask]                        # (M, 7)
feat_mean   = feat7_flat.mean(axis=0)                   # (7,)
feat_std    = feat7_flat.std(axis=0)                    # (7,)
feat_std[feat_std == 0] = 1.0

print(f"  mean: {np.round(feat_mean, 2)}", flush=True)
print(f"  std:  {np.round(feat_std,  2)}", flush=True)

# ── 3. Per-node mean detection_confidence → new node feature (dim 9) ─────────
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

# ── 4. Normalise self-loop edge weights ───────────────────────────────────────
# Pattern per timestep: each node may have multiple groups of exactly 7
# consecutive self-loops (one group per detection event it participates in).
# feat_idx cycles 0..6 within each group; track per-node count.
print("Normalising self-loop edge weights ...", flush=True)
edge_weight_v2 = {}
for step_str, ei_list in edge_index.items():
    ew_list    = edge_weight[step_str]
    ew_v2      = []
    self_count = {}      # node → cumulative self-loop count this step
    for (src, dst), w in zip(ei_list, ew_list):
        if src != dst:
            ew_v2.append(float(w))    # prox: detection_confidence unchanged
        else:
            cnt      = self_count.get(src, 0)
            feat_idx = cnt % SELF_FEAT_DIM
            norm_w   = (w - float(feat_mean[feat_idx])) / float(feat_std[feat_idx])
            ew_v2.append(float(norm_w))
            self_count[src] = cnt + 1
    edge_weight_v2[step_str] = ew_v2

# ── 5. Sanity check ──────────────────────────────────────────────────────────
step0_ew_self = [w for (s,d), w in zip(edge_index['0'], edge_weight_v2['0']) if s == d]
print(f"  step0 self ew sample (should be ~N(0,1)): {[round(w,3) for w in step0_ew_self[:7]]}", flush=True)

# ── 6. Write v2 JSON ──────────────────────────────────────────────────────────
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
    "feat_norm": {
        "self_loop_mean": feat_mean.tolist(),
        "self_loop_std":  feat_std.tolist(),
        "feature_order":  ["x","y","heading","speed","acceleration","sensor_range","visibility_range"],
        "note":           "self-loop ew are z-scored; prox ew = detection_confidence (unchanged)"
    }
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
1. Self-loop edge weights z-score normalised per feature (7 dimensions:
   x, y, heading, speed, acceleration, sensor_range, visibility_range).
   v1 stored raw coordinate/speed values causing large scale mismatch
   against the 0-1 detection_confidence proximity weights.
2. Node feature vector extended: F 9 -> 10.
   Index 9 = mean incoming detection_confidence for each node at each
   timestep (0 if no CPM observation received that step).
3. JSON includes "feat_norm" key with mean/std used for self-loop
   normalisation (for reference / inverse transform).

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
edge_weight    {"t": [w]}           Aligned weights:
                                    prox edges  : detection_confidence (0-1)
                                    self-loops  : z-score normalised feature
                                                  (normalisation in feat_norm)
feat_norm      dict                 Normalisation stats for self-loop ew:
                                    self_loop_mean, self_loop_std (len 7)

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
