"""
build_consensus_cpm.py — Stage 1 EXPLORATORY (self-contained, superseded).

⚠️ LABEL TRAP: this self-contained version derives the label from `cls_detected`,
which is the OBJECT CLASS (84% positive), NOT the misbehaviour label. The real
per-node label is the JSON `y_detected` (== raw `label_detected`, 4.2% positive).
Use the AUTHORITATIVE pipeline instead — it takes the label from the validated cache:
  build_consensus_cache.py  ->  run_consensus_rf.py / run_consensus_diag.py
This file is kept only for the SQL aggregation logic and as a record of the trap.

build_consensus_cpm.py — Stage 1 of the CPM consensus-signal test.

The per-node CPM cache collapsed every vehicle to its OWN reported kinematics +
in_degree, discarding the collective-perception cross-check: how a vehicle's
SELF-report (is_self=1) compares to what OTHER vehicles OBSERVE about it
(is_self=0 rows whose detected_vehicle_id is that vehicle). A vehicle that spoofs
its own BSM still gets sensed truthfully by neighbours -> self-vs-consensus
mismatch. That mismatch is the canonical CPM misbehaviour cue and is NOT in the
current features.

This script rebuilds a self-contained tabular dataset DIRECTLY from the raw DB
(sqlite-side GROUP BY, low memory) and runs a vehicle-disjoint RF in three
feature sets, to answer ONE question: does the consensus mismatch add signal
on top of the per-node physics features that already reach ~0.69-0.71 MCC?

  A) physics-only            (recomputed from self trajectory; in_degree included)
  B) physics + consensus     (adds self-vs-observers mismatch + observer spread)
  C) consensus-only          (how much standalone signal the mismatch carries)

If B >> A, the discarded collective-perception structure is recoverable signal
and a target-centric GNN (Stage 2) is motivated. If B ~ A, "RF wins" is final.

Usage:  python build_consensus_cpm.py [db_path] [n_seeds]
"""
import sys, time, sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score

DB = sys.argv[1] if len(sys.argv) > 1 else "imported_data/burst_adma.db"
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TABLE = "burst_adma_with_cpm_multi_sensors999_cls_all"


def circ_abs_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def load_from_db():
    t0 = time.time()
    con = sqlite3.connect(DB)
    print("[db] aggregating SELF rows (own BSM)...", flush=True)
    self_df = pd.read_sql_query(f"""
        SELECT CAST(timestep AS REAL) t, vehicle_id AS V,
               AVG(CAST(x_detected AS REAL))            sx,
               AVG(CAST(y_detected AS REAL))            sy,
               AVG(CAST(speed_detected AS REAL))        ss,
               AVG(CAST(heading_detected AS REAL))      sh,
               AVG(CAST(acceleration_detected AS REAL)) sa,
               MAX(CAST(cls_detected AS INTEGER))       cls
        FROM {TABLE}
        WHERE is_self='1' AND vehicle_id != ''
        GROUP BY t, V
    """, con)
    print(f"[db]   self rows: {len(self_df)} ({time.time()-t0:.0f}s)", flush=True)

    print("[db] aggregating CONSENSUS rows (others' observations per target)...", flush=True)
    cons_df = pd.read_sql_query(f"""
        SELECT CAST(timestep AS REAL) t, detected_vehicle_id AS V,
               COUNT(*)                                 nobs,
               AVG(CAST(x_detected AS REAL))            cx,
               AVG(CAST(y_detected AS REAL))            cy,
               AVG(CAST(speed_detected AS REAL))        cs,
               AVG(CAST(x_detected AS REAL)*CAST(x_detected AS REAL)) cxx,
               AVG(CAST(y_detected AS REAL)*CAST(y_detected AS REAL)) cyy
        FROM {TABLE}
        WHERE is_self='0' AND detected_vehicle_id != ''
        GROUP BY t, V
    """, con)
    con.close()
    print(f"[db]   consensus rows: {len(cons_df)} ({time.time()-t0:.0f}s)", flush=True)
    return self_df, cons_df


def build_features(self_df, cons_df):
    df = self_df.sort_values(["V", "t"]).reset_index(drop=True)
    # ---- per-node physics from the self trajectory (matches build_features_cpm_rich) ----
    g = df.groupby("V", sort=False)
    px, py = g["sx"].shift(1), g["sy"].shift(1)
    ps, ph, pa = g["ss"].shift(1), g["sh"].shift(1), g["sa"].shift(1)
    df["dx"] = (df.sx - px).abs()
    df["dy"] = (df.sy - py).abs()
    df["disp"] = np.sqrt(df.dx**2 + df.dy**2)
    df["speed_resid"] = (df.disp - df.ss).abs()          # displacement vs reported speed
    df["dspeed"] = (df.ss - ps).abs()
    df["dheading"] = circ_abs_deg(df.sh, ph)
    df["daccel"] = (df.sa - pa).abs()
    df["accel_resid"] = ((df.ss - ps) - df.sa).abs()     # delta-speed vs reported accel
    first = px.isna()                                     # no previous appearance
    for c in ["dx", "dy", "disp", "speed_resid", "dspeed", "dheading", "daccel", "accel_resid"]:
        df.loc[first, c] = 0.0

    # ---- consensus: self vs others' observations of self ----
    df = df.merge(cons_df, on=["t", "V"], how="left")
    df["nobs"] = df["nobs"].fillna(0.0)
    df["mismatch_pos"] = np.sqrt((df.sx - df.cx)**2 + (df.sy - df.cy)**2)
    df["mismatch_speed"] = (df.ss - df.cs).abs()
    var = (df.cxx - df.cx**2) + (df.cyy - df.cy**2)
    df["spread_pos"] = np.sqrt(var.clip(lower=0))
    no_cons = df["cx"].isna()                             # nobody observed this vehicle
    for c in ["mismatch_pos", "mismatch_speed", "spread_pos"]:
        df.loc[no_cons, c] = 0.0

    df["y"] = (df["cls"].fillna(0) != 0).astype(int)
    return df


PHYS = ["dx", "dy", "disp", "speed_resid", "dspeed", "dheading", "daccel",
        "accel_resid", "nobs", "ss"]
CONS = ["mismatch_pos", "mismatch_speed", "spread_pos"]


def bal_sw(y):
    w = np.ones(len(y))
    for c in (0, 1):
        n = (y == c).sum()
        if n > 0:
            w[y == c] = len(y) / (2.0 * n)
    return w


def best_mcc(y, p):
    bm, bt = -1.0, 0.5
    for thr in np.arange(0.05, 1.0, 0.025):
        pred = (p >= thr).astype(int)
        if pred.sum() == 0:
            continue
        m = matthews_corrcoef(y, pred)
        if m > bm:
            bm, bt = m, thr
    pred = (p >= bt).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return bm * 100, auc, f1_score(y, pred, average="macro", zero_division=0) * 100


def run(df, feats, seed):
    vehicles = df["V"].unique()
    vidx = {v: i for i, v in enumerate(vehicles)}
    node_lab = np.zeros(len(vehicles), dtype=int)
    for v, sub in df.groupby("V"):
        node_lab[vidx[v]] = int(sub["y"].max() > 0)
    rng = np.random.default_rng(seed)
    trm = np.zeros(len(vehicles), dtype=bool)
    for c in (0, 1):
        idx = np.where(node_lab == c)[0]; rng.shuffle(idx)
        trm[idx[:int(0.7 * len(idx))]] = True
    is_tr = df["V"].map(lambda v: trm[vidx[v]]).values
    X = df[feats].fillna(0.0).values
    y = df["y"].values
    rf = RandomForestClassifier(n_estimators=300, max_features=0.5, n_jobs=-1, random_state=seed)
    rf.fit(X[is_tr], y[is_tr], sample_weight=bal_sw(y[is_tr]))
    return best_mcc(y[~is_tr], rf.predict_proba(X[~is_tr])[:, 1])


def main():
    t0 = time.time()
    self_df, cons_df = load_from_db()
    df = build_features(self_df, cons_df)
    n_active = len(df); n_pos = int(df["y"].sum())
    print(f"[feat] {n_active} node-steps, {n_pos} positive ({100*n_pos/n_active:.2f}%), "
          f"{df['V'].nunique()} vehicles, consensus-present {100*(df['nobs']>0).mean():.1f}%", flush=True)
    print(f"[feat] mean mismatch_pos: attacker={df.loc[df.y==1,'mismatch_pos'].mean():.3f} "
          f"benign={df.loc[df.y==0,'mismatch_pos'].mean():.3f}", flush=True)

    sets = {"A_physics": PHYS, "B_phys+cons": PHYS + CONS, "C_cons_only": CONS}
    agg = {k: [] for k in sets}
    for s in range(NSEED):
        for name, feats in sets.items():
            mcc, auc, macro = run(df, feats, s)
            agg[name].append(mcc)
            print(f"RESULT consensus set={name} seed={s} mcc={mcc:.2f} auc={auc:.4f} macrof1={macro:.2f}", flush=True)
    print("\n=== CPM consensus-feature RF (vehicle-disjoint, {} seeds) ===".format(NSEED), flush=True)
    for name in sets:
        v = np.array(agg[name])
        print(f"{name:14s} {v.mean():5.1f} +/- {v.std():4.1f}", flush=True)
    a, b = np.array(agg["A_physics"]), np.array(agg["B_phys+cons"])
    print(f"\nconsensus lift (B - A): {(b-a).mean():+.1f}  ({(b>a).sum()}/{len(b)} seeds>0)", flush=True)
    print(f"(total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
