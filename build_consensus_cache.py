"""
build_consensus_cache.py  (run on .100 where the JSON + cache live)

Appends the COLLECTIVE-PERCEPTION CONSENSUS features that the per-node cache
discarded: for each vehicle V at step t, how its SELF-report (own BSM) differs
from the CONSENSUS of OTHER vehicles' observations of V.

  mismatch_pos   = || self(x,y) - mean_observers(x,y) ||     (key spoofing cue)
  mismatch_speed = | self_speed - mean_observers_speed |
  spread_pos     = std of observers' reported positions       (observer disagreement)

Inputs (small, exported locally from the raw DB):  self_agg.csv, cons_agg.csv
Ground-truth label + node order come from the cache / JSON (no re-derivation).

Regenerate the two aggregate CSVs from the raw DB (sqlite-side GROUP BY, low memory):
  sqlite3 -csv -header imported_data/burst_adma.db "SELECT CAST(timestep AS INTEGER) t,
    vehicle_id V, AVG(CAST(x_detected AS REAL)) sx, AVG(CAST(y_detected AS REAL)) sy,
    AVG(CAST(speed_detected AS REAL)) ss, AVG(CAST(heading_detected AS REAL)) sh,
    AVG(CAST(acceleration_detected AS REAL)) sa
    FROM burst_adma_with_cpm_multi_sensors999_cls_all
    WHERE is_self='1' AND vehicle_id!='' GROUP BY t,V" > self_agg.csv
  sqlite3 -csv -header imported_data/burst_adma.db "SELECT CAST(timestep AS INTEGER) t,
    detected_vehicle_id V, COUNT(*) nobs, AVG(CAST(x_detected AS REAL)) cx,
    AVG(CAST(y_detected AS REAL)) cy, AVG(CAST(speed_detected AS REAL)) cs,
    AVG(CAST(x_detected AS REAL)*CAST(x_detected AS REAL)) cxx,
    AVG(CAST(y_detected AS REAL)*CAST(y_detected AS REAL)) cyy
    FROM burst_adma_with_cpm_multi_sensors999_cls_all
    WHERE is_self='0' AND detected_vehicle_id!='' GROUP BY t,V" > cons_agg.csv

Output: data/cache_v1_consensus.pt  (= cache_v1_rich.pt with K consensus feats appended)
Validates node-order alignment by cross-checking JSON y_detected vs cache Y.
"""
import time, json
import numpy as np
import pandas as pd
import torch

JSON  = "data/burst_adma_with_cpm_multi_sensors999_cls_all__positive_with_features_as_self_edge.json"
CACHE = "data/cache_v1_rich.pt"
OUT   = "data/cache_v1_consensus.pt"
FEATS = ["mismatch_pos", "mismatch_speed", "spread_pos"]


def main():
    t0 = time.time()
    print("loading cache ...", flush=True)
    c = torch.load(CACHE)
    X, active, Y = c["X"], c["active"], c["Y"]
    n_nodes, T, lags, in_f = c["n_nodes"], c["T"], c["lags"], c["in_f"]
    n_steps = T - lags
    print(f"  cache: T={T} N={n_nodes} F={in_f} steps={n_steps} ({time.time()-t0:.0f}s)", flush=True)

    print("loading JSON (node_labels + y_detected) ...", flush=True)
    with open(JSON) as f:
        j = json.load(f)
    node_labels = j["node_labels"]
    y_det = np.array(j["y_detected"])          # (T, N)
    del j
    vid2idx = {v: i for i, v in enumerate(node_labels)}
    print(f"  node_labels={len(node_labels)} e.g. {node_labels[:4]}  y_detected={y_det.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- validate node-order / label alignment: cache.Y[i] == (y_detected[i+lags] != 0) ----
    mism = 0
    for i in range(0, n_steps, max(1, n_steps // 20)):
        a = Y[i].numpy().astype(bool)
        b = (y_det[i + lags] != 0)
        mism += int((a != b).sum())
    print(f"  [validate] label mismatch over sampled steps: {mism} (0 => aligned)", flush=True)

    # ---- consensus features from the DB aggregates ----
    self_df = pd.read_csv("self_agg.csv")
    cons_df = pd.read_csv("cons_agg.csv")
    m = self_df.merge(cons_df, on=["t", "V"], how="left")
    m["mismatch_pos"] = np.sqrt((m.sx - m.cx) ** 2 + (m.sy - m.cy) ** 2)
    m["mismatch_speed"] = (m.ss - m.cs).abs()
    var = (m.cxx - m.cx ** 2) + (m.cyy - m.cy ** 2)
    m["spread_pos"] = np.sqrt(var.clip(lower=0))
    no_cons = m["cx"].isna()
    for col in FEATS:
        m.loc[no_cons, col] = 0.0
    m["idx"] = m["V"].map(vid2idx)
    m = m.dropna(subset=["idx"])
    m["idx"] = m["idx"].astype(int)

    K = len(FEATS)
    cons = np.zeros((n_steps, n_nodes, K), dtype=np.float32)
    tt, ii = m["t"].values.astype(int), m["idx"].values
    ok = (tt >= 0) & (tt < n_steps) & (ii >= 0) & (ii < n_nodes)
    for k, col in enumerate(FEATS):
        cons[tt[ok], ii[ok], k] = m[col].values[ok].astype(np.float32)
    print(f"  placed {ok.sum()} consensus rows into ({n_steps},{n_nodes},{K})", flush=True)

    # ---- direct signal check with the CORRECT label ----
    amask = np.stack([active[t].numpy() for t in range(n_steps)])      # (n_steps,N)
    ymask = np.stack([Y[t].numpy() for t in range(n_steps)]).astype(bool)
    mp = cons[:, :, 0]
    att = mp[amask & ymask]; ben = mp[amask & ~ymask]
    print(f"  [signal] mismatch_pos  attacker={att.mean():.3f}+/-{att.std():.3f}  "
          f"benign={ben.mean():.3f}+/-{ben.std():.3f}  ratio={att.mean()/(ben.mean()+1e-9):.2f}", flush=True)
    ms = cons[:, :, 1]
    att2 = ms[amask & ymask]; ben2 = ms[amask & ~ymask]
    print(f"  [signal] mismatch_speed attacker={att2.mean():.3f}  benign={ben2.mean():.3f}", flush=True)

    # ---- z-score over active node-steps, zero inactive, append ----
    flat = cons[amask]
    mu, sd = flat.mean(0), flat.std(0); sd[sd == 0] = 1.0
    consz = (cons - mu) / sd
    for t in range(n_steps):
        consz[t][~active[t].numpy()] = 0.0
    Xnew = [torch.cat([X[t], torch.tensor(consz[t])], dim=1) for t in range(n_steps)]

    c_out = dict(c)
    c_out["X"] = Xnew
    c_out["in_f"] = in_f + K
    torch.save(c_out, OUT)
    print(f"saved {OUT}  in_f {in_f}->{in_f + K}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
